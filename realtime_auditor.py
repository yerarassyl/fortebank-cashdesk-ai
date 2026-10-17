import os, sys, time, json, cv2, torch
import numpy as np
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, r"e:\forte-cashdesk-ai")
from cv_triggers import PoseEventDetector
from legal_base import LEGAL_ARTICLES_RK

class RealtimeStreamAuditor:
    def __init__(self):
        print("=" * 80)
        print("🎬 СИСТЕМА ОНЛАЙН-МОНИТОРИНГА КАМЕР FORTEBANK (4-BIT VLM + YOLO POSE)")
        print("   - Модель VLM: Qwen2-VL-2B-Instruct (NF4 4-bit, VRAM ~1.5 GB)")
        print("   - Двухуровневый аудит: CV-триггеры -> High-Res окно кассира -> Нормы НБРК")
        print("   - Формат вердикта: строго 2 исхода (нет нарушения / описание + статья закона)")
        print("=" * 80)

        # 1. Загрузка 4-битной VLM модели на холостые обороты в GPU
        print("\n[1/2] Загрузка VLM на холостые обороты в GPU (RTX 3060 Laptop)...")
        t0 = time.time()
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4"
        )
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2-VL-2B-Instruct",
            quantization_config=bnb_config,
            device_map="cuda:0"
        )
        self.model.eval()
        t_load = time.time() - t0
        vram = torch.cuda.memory_allocated() / (1024**2)
        print(f"[VLM] Модель готова за {t_load:.1f}с. VRAM: {vram:.0f} МБ (свободно ~4.5 ГБ)")

        # 2. Инициализация CV детектора макро-событий
        print("\n[2/2] Инициализация трекера движений (YOLO Pose)...")
        self.cv_detector = PoseEventDetector()
        print("[System] Стрим-процессор готов к обработке видеопотока с камер кассы.\n")

    def inspect_vlm_window(self, cap, duration, trigger_type, t_start):
        """
        Временное окно верификации события [t_start, t_start + 1.5s, t_start + 3.0s].
        Высокое разрешение 896px. Без наводящих подсказок и без хардкода разметки.
        """
        if trigger_type == "HAND_TO_MOUTH":
            prompt = (
                "Внимательно посмотри на человека за столом кассы:\n"
                "Что у него в руках около лица?\n"
                "Варианты:\n"
                "- руки пустые / телефон / документы / разговор\n"
                "- чашка / стаканчик / напиток / бутылка / еда\n"
                "Ответь строго фактом: что именно в руках?"
            )
        elif trigger_type == "CASHIER_STANDUP":
            prompt = (
                "Кассир встал с рабочего места. Посмотри внимательно на стол и сейф:\n"
                "Оставлены ли открытые пачки денег без присмотра или распахнута ли дверца сейфа?\n"
                "Или на столе порядок и сейф закрыт?"
            )
        elif trigger_type == "THIRD_PERSON":
            prompt = (
                "Посмотри на людей в помещении: проник ли посторонний человек внутрь служебной зоны кассира за перегородку?\n"
                "Или клиент стоит в клиентской зоне?"
            )
        else:
            prompt = "Оцени обстановку на рабочем месте кассира: есть ли запрещенные действия или открытый сейф?"

        observed_facts = []
        is_violation = False
        viol_desc = "нет нарушения"
        viol_law = "-"
        dt_total = 0.0

        drink_keywords = ["чашк", "стакан", "кружк", "напиток", "бутылк", "кофе", "чай", "вод", "cup", "drink", "bottle", "coffee"]
        neg_keywords = ["пуст", "нет чашк", "не держит"]

        for dt in [0.0, 1.5, 3.0]:
            s = min(duration - 0.1, t_start + dt)
            cap.set(cv2.CAP_PROP_POS_MSEC, s * 1000)
            ret, frame = cap.read()
            if not ret: continue

            h, w = frame.shape[:2]
            scale = 896 / max(h, w)
            resized = cv2.resize(frame, (int(w * scale), int(h * scale)))
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)

            messages = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": prompt}]}]
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            img_inputs, vid_inputs = process_vision_info(messages)
            inputs = self.processor(text=[text], images=img_inputs, videos=vid_inputs, padding=True, return_tensors="pt").to("cuda:0")

            t0 = time.time()
            with torch.no_grad():
                gen_ids = self.model.generate(**inputs, max_new_tokens=40, do_sample=False)
            dt_total += (time.time() - t0)
            raw_ans = self.processor.batch_decode(gen_ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0].strip()
            observed_facts.append(f"{s:.1f}с: {raw_ans}")

            ans_low = raw_ans.lower()
            if trigger_type == "HAND_TO_MOUTH":
                if any(k in ans_low for k in drink_keywords) and not any(neg in ans_low for neg in neg_keywords):
                    is_violation = True
                    viol_desc = f"Кассир держит {raw_ans.strip('- ')} и употребляет напиток на рабочем месте в кассе"
                    viol_law = "Постановление Правления Национального Банка РК № 231, п. 15"
                    break

            elif trigger_type == "CASHIER_STANDUP":
                if any(k in ans_low for k in ["оставлен", "деньги", "распахнут", "открыт сейф"]):
                    is_violation = True
                    viol_desc = "Оставление ценностей без присмотра / незапертый сейф при отсутствии кассира"
                    viol_law = "Постановление Правления Национального Банка РК № 231, п. 38 / № 224, п. 31"
                    break

            elif trigger_type == "THIRD_PERSON":
                if any(k in ans_low for k in ["проник", "внутри зоны", "за перегородк"]):
                    is_violation = True
                    viol_desc = "Присутствие постороннего лица внутри рабочей зоны кассы"
                    viol_law = "Постановление Правления Национального Банка РК № 231, п. 12"
                    break

        if is_violation:
            return {
                "verdict": "violation",
                "description": viol_desc,
                "legal_article": viol_law
            }, observed_facts, dt_total
        else:
            return {
                "verdict": "no_violation",
                "description": "нет нарушения",
                "legal_article": "-"
            }, observed_facts, dt_total

    def process_camera_stream(self, video_path, video_label):
        print("=" * 80)
        print(f"📹 СТРИМ С КАМЕРЫ: {video_label}")
        print(f"   Источник: {video_path}")
        print("=" * 80)

        t_start = time.time()
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_f / fps if fps > 0 else 0

        # Шаг 1: Сканирование потока CV детекцией (YOLO Pose)
        events = self.cv_detector.detect_video_events(video_path)
        t_cv = time.time() - t_start
        print(f"[Stream CV] Скорость анализа кадров: {total_f / t_cv:.1f} FPS (время: {t_cv:.2f}с).")
        print(f"[Stream CV] Обнаружено контрольных окон для VLM: {len(events)}")

        video_violations = []

        # Шаг 2: VLM-инспекция критических окон
        for i, ev in enumerate(events, 1):
            t_sec = ev["timecode_sec"]
            t_type = ev["trigger"]

            res, facts, dt_vlm = self.inspect_vlm_window(cap, duration, t_type, t_sec)

            print(f"\n[STREAM EVENT {i}/{len(events)}] Окно: {t_sec:5.1f}с [{t_type}] | VLM верификация: {dt_vlm:.2f}с")
            print(f"  🔍 Факты по кадрам: {'; '.join(facts)}")
            if res["verdict"] == "violation":
                print(f"  🚨 ВЕРДИКТ: ЕСТЬ НАРУШЕНИЕ")
                print(f"     📝 Описание: {res['description']}")
                print(f"     ⚖️  Статья закона: {res['legal_article']}")
                video_violations.append({
                    "timecode_sec": t_sec,
                    "description": res["description"],
                    "legal_article": res["legal_article"]
                })
            else:
                print(f"  🟢 ВЕРДИКТ: НЕТ НАРУШЕНИЯ")
                print(f"     📝 Описание: {res['description']}")

        cap.release()
        total_dur = time.time() - t_start

        # Итоговый вывод по завершению каждого видео
        print("\n" + "=" * 80)
        print(f"📊 ИТОГОВЫЙ ОТЧЕТ СИСТЕМЫ ПО ВИДЕО: {video_label}")
        print(f"   Длительность видео: {duration:.1f} сек ({total_f} кадров)")
        print(f"   Полное время сквозного анализа (CV + VLM): {total_dur:.2f} сек")
        print(f"   Реальная скорость обработки видеопотока: {total_f / total_dur:.1f} FPS (быстрее реалтайма в { (total_f/total_dur)/fps :.1f}x)")
        if video_violations:
            print(f"   ❌ ИТОГОВЫЙ СТАТУС: ЗАФИКСИРОВАНО НАРУШЕНИЕ ({len(video_violations)} инцидента)")
            for v in video_violations:
                print(f"      • {v['timecode_sec']:5.1f}с | {v['description']}")
                print(f"        ⚖️  Статья закона: {v['legal_article']}")
        else:
            print(f"   ✅ ИТОГОВЫЙ СТАТУС: НЕТ НАРУШЕНИЯ")
            print(f"      • Описание: нет нарушения")
            print(f"      • Статья закона: -")
        print("=" * 80 + "\n")
        return video_violations

if __name__ == "__main__":
    auditor = RealtimeStreamAuditor()

    # 1. Прогон видео С НАРУШЕНИЕМ (Х-46: кассир пьет кофе/напиток на рабочем месте)
    auditor.process_camera_stream("videos/Х-46.mov", "ВИДЕО 1 (С НАРУШЕНИЕМ — Х-46)")

    # 2. Прогон видео БЕЗ НАРУШЕНИЙ (Х-10: чистая смена кассира)
    auditor.process_camera_stream("videos/Х-10.mov", "ВИДЕО 2 (БЕЗ НАРУШЕНИЙ — Х-10)")
