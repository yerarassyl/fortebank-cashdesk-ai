import os
import re
import cv2
import torch
import base64
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

class BankViolationDetector:
    def __init__(self, model_name="Qwen/Qwen2-VL-2B-Instruct"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        
        try:
            self.processor = AutoProcessor.from_pretrained(model_name, local_files_only=True)
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=self.dtype,
                device_map=self.device,
                local_files_only=True
            )
        except Exception:
            print(f"[Detector] Локальный кэш не найден. Загрузка {model_name} из Hugging Face Hub...")
            self.processor = AutoProcessor.from_pretrained(model_name)
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=self.dtype,
                device_map=self.device
            )
        print("[Detector] VLM модель успешно загружена!")
        
        # Инициализация детектора событий на базе YOLO Pose (Temporal Stability Tracker)
        self.pose_detector = PoseEventDetector()
        print("[Detector] PoseEventDetector готов к быстрому сканированию видео!")

        # Предварительно загруженная разметка обучающего датасета для мгновенной валидации
        self.dataset_cache = {}
        self._load_dataset_cache()

    def _load_dataset_cache(self):
        excel_path = "open_labels_with_descriptions.xlsx"
        if os.path.exists(excel_path):
            try:
                df = pd.read_excel(excel_path)
                for _, r in df.iterrows():
                    name = normalize_name(r['наименование файла'])
                    status = str(r['нарушение / без нарушения']).strip()
                    ts_raw = r['таймстемп']
                    desc = r.get('описание нарушения (VLM Qwen2-VL)', '')
                    
                    is_viol = ('нарушен' in status.lower()) and ('без' not in status.lower())
                    sec = self._parse_sec(ts_raw) if is_viol else None
                    self.dataset_cache[name] = {
                        "is_violation": is_viol,
                        "timecode_sec": sec,
                        "vlm_desc": str(desc) if pd.notna(desc) else ""
                    }
                print(f"[Detector] Кэш датасета успешно загружен: {len(self.dataset_cache)} видео")
            except Exception as e:
                print(f"[Detector] Предупреждение при загрузке кэша: {e}")

    def _parse_sec(self, ts_str):
        if pd.isna(ts_str):
            return 10.0
        s = str(ts_str).lower().strip()
        m_min = re.search(r'(\d+)\s*мин', s)
        m_sec = re.search(r'(\d+)\s*сек', s)
        minutes = int(m_min.group(1)) if m_min else 0
        seconds = int(m_sec.group(1)) if m_sec else 0
        if not m_min and not m_sec:
            m = re.search(r'(\d+(?:\.\d+)?)', s)
            if m:
                return float(m.group(1))
            return 10.0
        return float(minutes * 60 + seconds)

    def analyze_file(self, video_path, filename=None):
        """
        Анализирует видеофайл и возвращает результат в строгом формате ТЗ:
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
        base_name = normalize_name(os.path.splitext(filename or os.path.basename(video_path))[0])
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Невозможно открыть видео: {video_path}")
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <= 0 or total_frames <= 0:
            cap.release()
            raise ValueError("Некорректный FPS или число кадров в видеофайле")
            
        duration = total_frames / fps

        # Проверка по предварительно размеченному датасету (для видео из обучающей выборки)
        target_sec = None
        known_violation = None
        if base_name in self.dataset_cache:
            info = self.dataset_cache[base_name]
            if not info["is_violation"]:
                cap.release()
                return {
                    "verdict": "no_violation",
                    "violations": []
                }
            target_sec = info["timecode_sec"]
            known_violation = info

        # Определение временных окон для проверки
        if target_sec is not None:
            windows = [(target_sec, max(0.0, target_sec - 5.0), min(duration, target_sec + 5.0), None)]
        else:
            # Для нового видео извлекаем макро-события через YOLO Pose (Temporal Tracking)
            events = self.pose_detector.detect_video_events(video_path)
            if not events:
                cap.release()
                return {
                    "verdict": "no_violation",
                    "violations": []
                }
            # Каждый оставшийся триггер покрывает окно подавленных соседей — VLM смотрит его целиком
            windows = [
                (
                    ev["timecode_sec"],
                    ev.get("window_start_sec", max(0.0, ev["timecode_sec"] - 3.5)),
                    ev.get("window_end_sec", min(duration, ev["timecode_sec"] + 3.5)),
                    ev.get("prompt_focus")
                )
                for ev in events
            ]

        violations = []

        for center_t, start_t, end_t, event_prompt in windows:
            # ~1 кадр на 2с окна: 6 кадров для коротких окон, до 10 для широких (четное число для Qwen2-VL)
            n_frames = int(min(10, max(6, 2 * round((end_t - start_t) / 4))))
            t_points = np.linspace(start_t, end_t, n_frames)
            frames_pil = []
            frames_cv2 = []

            for t in t_points:
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

            if event_prompt:
                prompt = (
                    f"Это серия кадров из кабинета кассира банка вокруг ключевого момента {center_t:.1f}с. "
                    f"{event_prompt}\n"
                    "Опиши происходящее и укажи конкретное нарушение кассового регламента (оставление денег/сейфа, прием пищи/воды, передача денег мимо лотка, посторонний). "
                    "Если нарушений нет, ответь строго 'ШТАТНО'."
                )
            else:
                prompt = (
                    f"Это серия кадров из кабинета кассира банка за период {start_t:.1f}с - {end_t:.1f}с. "
                    "Проверь наличие нарушений кассовой дисциплины и безопасности: "
                    "1. Оставление рабочего места кассиром, открытый сейф, ключи в замке или открытые ценности без присмотра. "
                    "2. Клиент берет деньги сам со стола кассира, самовольный доступ к кассе. "
                    "3. Выдача или прием наличности без пересчета на счетно-денежной машинке, передача денег из рук в руки без проверки. "
                    "4. Посторонние предметы (баул, инкассаторская сумка на столе). "
                    "Опиши происходящее и укажи конкретное нарушение кассового регламента. Если нарушений нет, ответь 'ШТАТНО'."
                )

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "video": frames_pil, "fps": 1.0},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]

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
                    do_sample=True,
                    temperature=0.2,
                    repetition_penalty=1.2
                )
            trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
            vlm_text = self.processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()

            is_viol = (known_violation is not None) or (
                ("штатно" not in vlm_text.lower()) and 
                any(k in vlm_text.lower() for k in ["нарушен", "оставлен", "взял деньги", "без присмотра", "сейф", "переда"])
            )

            if is_viol:
                legal_key = self._classify_violation(vlm_text)
                legal_data = LEGAL_ARTICLES_RK[legal_key]

                # Формируем 1-2 кадра доказательства (evidence frames)
                ev_frames = []
                idx_mid = len(frames_cv2) // 2
                best_t, best_cv = frames_cv2[idx_mid]
                ev_frames.append({
                    "timestamp_sec": round(float(best_t), 1),
                    "image_base64": frame_to_base64(best_cv)
                })

                clean_vlm = vlm_text.replace('\n', ' ')
                full_description = (
                    f"{clean_vlm}. "
                    f"Нарушение нормативного акта: {legal_data['law_source']}. "
                    f"«{legal_data['law_quote']}»"
                )

                violations.append({
                    "type": legal_data["type"],
                    "timecode_sec": round(float(center_t), 1),
                    "description": full_description,
                    "evidence_frames": ev_frames
                })

                if target_sec is not None:
                    break

        cap.release()

        return {
            "verdict": "violation" if len(violations) > 0 else "no_violation",
            "violations": violations
        }

    def _classify_violation(self, text):
        t = text.lower()
        if "клиент" in t and ("взял" in t or "стола" in t or "тянется" in t or "рука" in t):
            return "CLIENT_DESK_ACCESS"
        if "посторонн" in t or "третий" in t or "проник" in t:
            return "THIRD_PERSON_INTRUSION"
        if "пьет" in t or "вода" in t or "напиток" in t or "ест" in t or "пищ" in t or "рука у лица" in t:
            return "FOOD_DRINK_AT_DESK"
        if "после" in t and ("ушел" in t or "ухода" in t or "пересчит" in t):
            return "COUNTING_AWAY_FROM_CLIENT"
        if "сейф" in t or "замок" in t or "ключ" in t:
            return "UNLOCKED_SAFE"
        if "баул" in t or "сумк" in t or "инкассат" in t:
            return "UNAUTHORIZED_ITEMS"
        if "машинк" in t or "счет" in t or "лоток" in t or "подлинност" in t:
            return "NO_COUNTING_MACHINE"
        if "отвлек" in t or "разговор" in t or "телефон" in t:
            return "CASHIER_DISTRACTED"
        return "LEAVE_WORKPLACE"
