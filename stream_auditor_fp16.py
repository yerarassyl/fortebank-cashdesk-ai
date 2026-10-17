import os, sys, time, json, cv2, torch
import numpy as np
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, r"e:\forte-cashdesk-ai")
from cv_triggers import PoseEventDetector

class FP16CameraAuditor:
    def __init__(self):
        print("=" * 80, flush=True)
        print("🚀 REAL-TIME АУДИТ ПОТОКА В ПОЛНОЙ ТОЧНОСТИ FP16 (X-1 и X-2)", flush=True)
        print("   - Модель VLM: Qwen2-VL-2B-Instruct (torch.float16, VRAM ~4.2 GB)", flush=True)
        print("   - Модель CV: YOLO Pose 11m (отсечение фонового коридора x < 28%)", flush=True)
        print("   - Формат вердикта: строго 2 исхода (нет нарушения / описание + статья закона)", flush=True)
        print("=" * 80, flush=True)

        print("\n[1/2] Загрузка VLM в FP16 на холостые обороты в GPU...", flush=True)
        t0 = time.time()
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2-VL-2B-Instruct",
            torch_dtype=torch.float16,
            device_map="cuda:0"
        )
        self.model.eval()
        t_vlm = time.time() - t0
        vram = torch.cuda.memory_allocated() / (1024**2)
        print(f"[VLM FP16] Модель подгружена за {t_vlm:.1f}с. VRAM: {vram:.0f} МБ (свободно ~1.8 ГБ)", flush=True)

        print("\n[2/2] Инициализация трекера движений (YOLO Pose)...", flush=True)
        self.cv_detector = PoseEventDetector()
        print("[System] Стрим-процессор готов к обработке видеопотоков.\n", flush=True)

    def inspect_vlm_window(self, cap, duration, trigger_type, t_start):
        drink_keywords = ["чашк", "стакан", "кружк", "напиток", "бутылк", "кофе", "чай", "вод", "cup", "drink", "bottle", "coffee"]
        neg_keywords = ["пуст", "нет чашк", "не держит", "пустые", "нет бутылк", "телефон", "разговор"]

        observed_facts = []
        is_violation = False
        viol_desc = "нет нарушения"
        viol_law = "-"
        dt_total = 0.0

        if trigger_type == "HAND_TO_MOUTH":
            prompt = (
                "Внимательно посмотри на руки кассира: держит ли он чашку, стаканчик, кружку или бутылку с напитком?\n"
                "Или руки пустые / держит телефон / занят работой?\n"
                "Ответь строго фактом: что именно в руках?"
            )
            offsets = [0.0, 1.5]
        elif trigger_type == "CLIENT_ARRIVAL":
            prompt = (
                "В кассу подошел клиент. Посмотри на действия: происходит ли обычное обслуживание клиента?\n"
                "Или есть открытый сейф / оставленные без присмотра деньги?"
            )
            offsets = [0.0]
        elif trigger_type == "CASHIER_STANDUP":
            prompt = (
                "Кассир встал с места. Оставлены ли открытые пачки денег без присмотра или распахнута ли дверца сейфа?\n"
                "Ответь строго: 'ДА' или 'НЕТ'."
            )
            offsets = [0.0]
        else:
            prompt = "Оцени обстановку на рабочем месте: есть ли открытый сейф или явные нарушения? Ответь: 'ДА' или 'НЕТ'."
            offsets = [0.0]

        for dt in offsets:
            s = min(max(0.0, duration - 0.1), t_start + dt)
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
                    item = raw_ans.strip('- *')
                    viol_desc = f"Кассир держит {item} и употребляет напиток на рабочем месте в кассе"
                    viol_law = "Постановление Правления Национального Банка РК № 231, п. 15"
                    break

            elif trigger_type in ["CASHIER_STANDUP", "INITIAL_DESK_CHECK"]:
                if ans_low.startswith("да") or "распахнут" in ans_low:
                    is_violation = True
                    viol_desc = "Оставление ценностей без присмотра / незапертый сейф при отсутствии кассира"
                    viol_law = "Постановление Правления Национального Банка РК № 231, п. 38 / № 224, п. 31"
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

    def process_video_stream(self, video_path, video_label):
        print("=" * 80, flush=True)
        print(f"📹 СТРИМ С КАМЕРЫ: {video_label}")
        print(f"   Файл: {video_path}", flush=True)
        print("=" * 80, flush=True)

        t_start = time.time()
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_f / fps if fps > 0 else 0

        # Сканирование CV потока
        t_cv_0 = time.time()
        events = self.cv_detector.detect_video_events(video_path)
        t_cv = time.time() - t_cv_0
        cv_fps = total_f / t_cv if t_cv > 0 else 0
        print(f"[Stream CV] Просканировано {total_f} кадров со скоростью {cv_fps:.1f} FPS (время: {t_cv:.2f}с).", flush=True)
        print(f"[Stream CV] Обнаружено контрольных окон для VLM: {len(events)}", flush=True)

        video_violations = []

        # VLM верификация контрольных окон
        for i, ev in enumerate(events, 1):
            t_sec = ev["timecode_sec"]
            t_type = ev["trigger"]

            res, facts, dt_vlm = self.inspect_vlm_window(cap, duration, t_type, t_sec)

            print(f"\n[STREAM EVENT {i}/{len(events)}] Окно: {t_sec:5.1f}с [{t_type}] | VLM отклик: {dt_vlm:.2f}с", flush=True)
            print(f"  🔍 Наблюдение VLM: {'; '.join(facts)}", flush=True)
            if res["verdict"] == "violation":
                print(f"  🚨 ВЕРДИКТ: ЕСТЬ НАРУШЕНИЕ", flush=True)
                print(f"     📝 Описание: {res['description']}", flush=True)
                print(f"     ⚖️  Статья закона: {res['legal_article']}", flush=True)
                video_violations.append({
                    "timecode_sec": t_sec,
                    "trigger": t_type,
                    "description": res["description"],
                    "legal_article": res["legal_article"]
                })
            else:
                print(f"  🟢 ВЕРДИКТ: НЕТ НАРУШЕНИЯ (Штатно)", flush=True)

        cap.release()
        total_time = time.time() - t_start
        effective_fps = total_f / total_time if total_time > 0 else 0
        realtime_speedup = (effective_fps / fps) if fps > 0 else 0

        # Итоговый вывод по завершению видео
        print("\n" + "=" * 80, flush=True)
        print(f"📊 ИТОГОВЫЙ ОТЧЕТ СИСТЕМЫ ПО ВИДЕО: {video_label}", flush=True)
        print(f"   Длительность видеопотока: {duration:.1f} сек ({total_f} кадров)", flush=True)
        print(f"   Полное время сквозного анализа (CV + VLM FP16): {total_time:.2f} сек", flush=True)
        print(f"   Реальная скорость обработки видеопотока: {effective_fps:.1f} FPS (быстрее реалтайма в {realtime_speedup:.1f}x)", flush=True)
        if video_violations:
            print(f"   ❌ СТАТУС СМЕНЫ: ЗАФИКСИРОВАНО НАРУШЕНИЕ ({len(video_violations)} инцидентов)", flush=True)
            for v in video_violations:
                print(f"      • {v['timecode_sec']:5.1f}с [{v['trigger']}] | {v['description']}", flush=True)
                print(f"        ⚖️  Закон: {v['legal_article']}", flush=True)
        else:
            print(f"   ✅ СТАТУС СМЕНЫ: НЕТ НАРУШЕНИЯ (Штатная работа кассы)", flush=True)
            print(f"      • Описание: нет нарушения", flush=True)
            print(f"      • Статья закона: -", flush=True)
        print("=" * 80 + "\n", flush=True)
        return video_violations

if __name__ == "__main__":
    auditor = FP16CameraAuditor()

    # Прогон 1-го видео
    auditor.process_video_stream("videos/X-1.mov", "ВИДЕО 1 (X-1)")

    # Прогон 2-го видео
    auditor.process_video_stream("videos/X-2.mov", "ВИДЕО 2 (X-2)")
