import os
import re
import cv2
import torch
import base64
import json
import numpy as np
import pandas as pd
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from legal_base import LEGAL_ARTICLES_RK
from cv_triggers import PoseEventDetector

def normalize_name(s):
    return str(s).strip().replace('Х', 'X').replace('х', 'x').lower()

def frame_to_base64(cv2_frame, max_size=720, quality=85):
    h, w = cv2_frame.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        cv2_frame = cv2.resize(cv2_frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    _, buffer = cv2.imencode('.jpg', cv2_frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return base64.b64encode(buffer).decode('utf-8')

SYSTEM_RULES_PROMPT = """
Ты — эксперт службы безопасности и кассового аудита ForteBank (Техническое задание Datathon 26).
Твоя задача — проверить серию кадров видеонаблюдения кассовой зоны вокруг подозрительного момента на соответствие правилам кассовой дисциплины (Постановления Правления Национального Банка РК № 231 и № 224, adilet.zan.kz).

СПИСОК ПРАВИЛ И НАРУШЕНИЙ КЕЙСА FORTEBANK:
1. UNLOCKED_SAFE: "Нарушение режима хранения ценностей / незапертый сейф или оставленный ключ" (Постановление НБРК № 224, п. 31) — сейф открыт, кассир не заперла дверцу, ключ торчит в замке.
2. LEAVE_WORKPLACE: "Оставление рабочего места / незаблокированный ноутбук / открытые ценности" (Постановление НБРК № 231, п. 38) — кассир встала/ушла, не заблокировав ноутбук, или оставила деньги/документы без присмотра.
3. UNLOCKED_DOOR: "Незакрытая входная дверь кассовой кабины" (Постановление НБРК № 224, п. 12) — кассир или клиент вошли/вышли и оставили входную дверь открытой.
4. NO_ID_VERIFICATION: "Непроведение идентификации клиента перед началом операции" (Постановление НБРК № 231, п. 18) — кассир начала обслуживание без проверки удостоверения личности клиента.
5. THIRD_PERSON_INTRUSION: "Присутствие посторонних лиц в кассовой кабине / подсматривание" (Постановление НБРК № 231, п. 12) — в зоне кассы находится третий человек или посторонний подсматривает за кассиром.
6. COUNTING_AWAY_FROM_CLIENT: "Пересчет наличности вне поля зрения клиента" (Постановление НБРК № 231, п. 23) — кассир пересчитывает деньги после того, как клиент уже отошел от кассы.
7. NO_COUNTING_MACHINE: "Нарушение порядка пересчета, проверки подлинности и сортировки банкнот" (Постановление НБРК № 231, п. 22-24) — не пересчитала на машинке, или соединила отбракованные/фальшивые купюры с нормальными.
8. MONEY_LEFT_BETWEEN_CLIENTS: "Наличие остатков наличности от предыдущей операции при обслуживании следующего клиента" (Постановление НБРК № 231, п. 27) — кассир начала обслуживание нового клиента, не убрав деньги прошлого клиента в кассу.
9. FOOD_DRINK_AT_DESK: "Прием пищи / напитков на рабочем месте в кассовом узле" (Постановление НБРК № 231, п. 15, 56) — кассир пьет воду/чай, принимает пищу или на столе лежит инкассаторский мешок/баул/личные вещи.
10. CLIENT_DESK_ACCESS: "Неполная выдача денег / нарушение передачи через лоток" (Постановление НБРК № 231, п. 25) — не вернула часть сдачи или передала деньги мимо лотка.

ИНСТРУКЦИЯ ПО ОЦЕНКЕ:
- Большинство штатных действий кассира (разговор, перекладывание чистых бланков, опора на руку) НЕ ЯВЛЯЮТСЯ нарушениями.
- Если нарушений нет, ответь:
  {"verdict": "no_violation", "is_violation": false, "type": "ШТАТНО", "description": "Действия кассира и клиента соответствуют регламенту.", "legal_source": "-"}
- Если на кадрах зафиксировано нарушение из списка, укажи:
  {"verdict": "violation", "is_violation": true, "type": "<Точное название нарушения>", "description": "<Что именно видно на кадрах и почему это нарушение>", "legal_source": "<Статья и пункт Постановления НБРК>"}
"""

class BankViolationDetector:
    def __init__(self, model_name="Qwen/Qwen2-VL-2B-Instruct"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        
        print(f"[BankViolationDetector] Загрузка VLM на холостые ({model_name})...")
        try:
            self.processor = AutoProcessor.from_pretrained(model_name, local_files_only=True)
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=self.dtype,
                device_map=self.device,
                local_files_only=True
            )
        except Exception:
            print(f"[BankViolationDetector] Загрузка {model_name} из Hugging Face Hub...")
            self.processor = AutoProcessor.from_pretrained(model_name)
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=self.dtype,
                device_map=self.device
            )
        self.model.eval()
        vram = torch.cuda.memory_allocated() / (1024**2) if self.device == "cuda" else 0
        print(f"[BankViolationDetector] VLM готова на холостых! Занято VRAM: {vram:.0f} MB")
        
        # Инициализация детектора событий на базе YOLO Pose (Temporal Stability Tracker)
        self.pose_detector = PoseEventDetector()
        print("[BankViolationDetector] PoseEventDetector готов к быстрому сканированию видео!")

    def analyze_file(self, video_path, filename=None):
        """
        Анализирует видеофайл и возвращает результат в строгом формате ТЗ Datathon 26:
        {
          "verdict": "no_violation" | "violation",
          "violations": [
             {
               "type": "...",
               "timecode_sec": 44.2,
               "description": "...",
               "evidence_frames": [
                 { "timestamp_sec": 44.2, "image_base64": "..." }
               ]
             }
          ]
        }
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Невозможно открыть видео: {video_path}")
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <= 0 or total_frames <= 0:
            cap.release()
            raise ValueError("Некорректный FPS или число кадров в видеофайле")
            
        duration = total_frames / fps

        # 1. Быстрое извлечение ключевых событий кассовой зоны через YOLO Pose
        events = self.pose_detector.detect_video_events(video_path)
        if not events:
            cap.release()
            return {
                "verdict": "no_violation",
                "violations": []
            }

        violations = []

        # 2. Инспекция ключевых кадров каждого события через подгруженную VLM
        for ev in events:
            center_t = ev["timecode_sec"]
            start_t = ev.get("window_start_sec", max(0.0, center_t - 3.0))
            end_t = ev.get("window_end_sec", min(duration, center_t + 4.0))
            trigger_type = ev.get("trigger", "EVENT")
            prompt_focus = ev.get("prompt_focus", trigger_type)

            # Выбор 5-6 ключевых кадров под специфику события
            if trigger_type == "CASHIER_STANDUP":
                offsets = [-1.5, 0.0, 2.0, 4.0, 6.0]
            elif trigger_type == "HAND_TO_MOUTH":
                offsets = [-1.5, -0.5, 0.0, 1.0, 2.5]
            elif trigger_type == "HANDOVER":
                offsets = [-2.5, -1.0, 0.0, 1.5, 3.0]
            elif trigger_type == "CLIENT_DEPARTURE":
                offsets = [-1.0, 0.0, 2.0, 4.0, 6.0]
            else:
                offsets = [-2.0, 0.0, 2.0, 4.0]

            target_secs = [max(0.0, min(duration, center_t + dt)) for dt in offsets]
            target_secs = sorted(list(set(target_secs)))

            frames_pil = []
            frames_cv2 = []

            for t in target_secs:
                cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
                ret, frame = cap.read()
                if ret:
                    frames_cv2.append((t, frame))
                    h, w = frame.shape[:2]
                    scale = 448 / max(h, w)
                    resized = cv2.resize(frame, (int(w * scale), int(h * scale)))
                    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                    frames_pil.append(Image.fromarray(rgb))

            if not frames_pil:
                continue

            # Промпт по правилам кейса ForteBank
            content = []
            for img in frames_pil:
                content.append({"type": "image", "image": img})

            user_query = (
                f"{SYSTEM_RULES_PROMPT}\n\n"
                f"Серия кадров видеонаблюдения вокруг таймкода {center_t:.1f}с.\n"
                f"Тип обнаруженного события: [{trigger_type}]. Фокус контроля: {prompt_focus}\n"
                "Определи, есть ли нарушение кассовой дисциплины, и ответь СТРОГО в формате JSON:"
            )
            content.append({"type": "text", "text": user_query})

            messages = [{"role": "user", "content": content}]
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=160,
                    do_sample=False
                )
            trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
            vlm_text = self.processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()

            # Парсинг ответа VLM
            is_viol, v_type, v_desc, v_source = self._parse_vlm_response(vlm_text, trigger_type)

            if is_viol:
                # 1-2 кадра доказательства (evidence frames)
                ev_frames = []
                idx_mid = len(frames_cv2) // 2
                best_t, best_cv = frames_cv2[idx_mid]
                ev_frames.append({
                    "timestamp_sec": round(float(best_t), 1),
                    "image_base64": frame_to_base64(best_cv)
                })

                full_description = f"{v_desc} Источник: {v_source}"

                violations.append({
                    "type": v_type,
                    "timecode_sec": round(float(center_t), 1),
                    "description": full_description,
                    "evidence_frames": ev_frames
                })

        cap.release()

        return {
            "verdict": "violation" if len(violations) > 0 else "no_violation",
            "violations": violations
        }

    def _parse_vlm_response(self, text, trigger_type):
        """
        Надежный парсер JSON-ответа от VLM с fallback на юридическую базу НПА РК
        """
        try:
            m = re.search(r'\{.*\}', text, re.DOTALL)
            if m:
                data = json.loads(m.group(0))
                verdict = str(data.get("verdict", "")).lower()
                is_viol = data.get("is_violation", False) or (verdict == "violation")
                if not is_viol or str(data.get("type", "")).upper() == "ШТАТНО":
                    return False, "ШТАТНО", "Штатная работа кассы.", "-"
                
                v_type = data.get("type", "Нарушение регламента кассы")
                v_desc = data.get("description", "Зафиксировано отклонение от кассовой дисциплины.")
                v_source = data.get("legal_source", "Постановление Правления НБРК № 231")
                return True, v_type, v_desc, v_source
        except Exception:
            pass

        # Fallback по ключевым словам и типу триггера
        t_low = text.lower()
        if "штатно" in t_low or "нет нарушения" in t_low:
            return False, "ШТАТНО", "Штатная работа кассы.", "-"

        legal_key = self._classify_violation(text, trigger_type)
        legal_data = LEGAL_ARTICLES_RK.get(legal_key, LEGAL_ARTICLES_RK["LEAVE_WORKPLACE"])
        return True, legal_data["type"], legal_data["law_quote"], legal_data["law_source"]

    def _classify_violation(self, text, trigger_type):
        t = text.lower()
        if "клиент" in t and ("взял" in t or "стола" in t or "сдач" in t or "лоток" in t):
            return "CLIENT_DESK_ACCESS"
        if "посторонн" in t or "третий" in t or "подсматр" in t or trigger_type == "THIRD_PERSON":
            return "THIRD_PERSON_INTRUSION"
        if "пьет" in t or "вода" in t or "напиток" in t or "ест" in t or trigger_type == "HAND_TO_MOUTH":
            return "FOOD_DRINK_AT_DESK"
        if "после" in t and ("ушел" in t or "ухода" in t or "пересчит" in t) or trigger_type == "CLIENT_DEPARTURE":
            return "COUNTING_AWAY_FROM_CLIENT"
        if "сейф" in t or "замок" in t or "ключ" in t:
            return "UNLOCKED_SAFE"
        if "дверь" in t and ("не закры" in t or "открыт" in t):
            return "UNLOCKED_DOOR"
        if "удостов" in t or "паспорт" in t or "личность" in t:
            return "NO_ID_VERIFICATION"
        if "прошл" in t and "деньги" in t:
            return "MONEY_LEFT_BETWEEN_CLIENTS"
        if "баул" in t or "сумк" in t or "инкассат" in t:
            return "UNAUTHORIZED_ITEMS"
        if "машинк" in t or "счет" in t or "фальшив" in t:
            return "NO_COUNTING_MACHINE"
        if trigger_type == "CASHIER_STANDUP":
            return "LEAVE_WORKPLACE"
        return "LEAVE_WORKPLACE"
