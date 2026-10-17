import os, sys, time, json, cv2, torch
import numpy as np
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, r"e:\forte-cashdesk-ai")
from cv_triggers import PoseEventDetector
from legal_base import LEGAL_ARTICLES_RK

class ProductionCashDeskAuditor:
    def __init__(self):
        print("=" * 80, flush=True)
        print("🛡️  БОЕВОЙ ИИ-КОМПЛАЕНС МОНИТОРИНГ FORTEBANK (ПОЛНЫЙ ПРОГОН)", flush=True)
        print("   - Модель VLM: Qwen2-VL-2B-Instruct (полная точность FP16, VRAM ~4.2 GB)", flush=True)
        print("   - Модель CV: YOLO Pose 11m (отсечение коридора x < 28%, скорость ~300 FPS)", flush=True)
        print("   - Архитектура: Двухэтапная верификация (Физический факт -> Нормы НБРК)", flush=True)
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
        print("[System] Все компоненты готовы к боевой обработке видеопотоков.\n", flush=True)

    def verify_event_window(self, cap, duration, trigger_type, t_sec):
        drink_keywords = ["чашк", "стакан", "кружк", "напиток", "бутылк", "кофе", "чай", "вод", "cup", "drink", "bottle", "coffee"]
        neg_keywords = ["пуст", "нет чашк", "не держит", "пустые", "нет бутылк", "телефон", "разговор", "бумаг"]

        observed_facts = []
        is_violation = False
        viol_desc = "нет нарушения"
        viol_law = "-"
        viol_type = "-"
        dt_total = 0.0

        if trigger_type == "HAND_TO_MOUTH":
            prompt = (
                "Внимательно посмотри на руки кассира: держит ли он чашку, стаканчик, кружку или бутылку с напитком?\n"
                "Или руки пустые / держит телефон / занят работой?\n"
                "Ответь строго фактом: что именно в руках?"
            )
            # Сканируем кульминацию жеста питья
            times = [min(duration - 0.1, t_sec + 1.5), min(duration - 0.1, t_sec + 3.0)]
        elif trigger_type in ["CASHIER_STANDUP", "INITIAL_DESK_CHECK"]:
            prompt = (
                "Посмотри внимательно на сейф в углу и рабочий стол:\n"
                "Распахнута ли дверца сейфа или оставлены ли открытые деньги без присмотра?\n"
                "Ответь строго: 'ДА' или 'НЕТ'."
            )
            times = [t_sec]
        elif trigger_type == "THIRD_PERSON":
            prompt = (
                "В кассе находится кассир и клиент. Находится ли кто-либо посторонний внутри служебной зоны кассира за столом?\n"
                "Ответь строго: 'ДА' или 'НЕТ'."
            )
            times = [t_sec]
        else:
            prompt = "Происходит ли обычное обслуживание клиента? Ответь: 'ДА' или 'НЕТ'."
            times = [t_sec]

        for t in times:
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
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
            observed_facts.append(f"{t:.1f}с: {raw_ans}")

            ans_low = raw_ans.lower()
            if trigger_type == "HAND_TO_MOUTH":
                if any(k in ans_low for k in drink_keywords) and not any(neg in ans_low for neg in neg_keywords):
                    is_violation = True
                    item = raw_ans.strip('- *')
                    viol_type = "FOOD_DRINK_AT_DESK"
                    viol_desc = f"Кассир держит {item} и употребляет напиток на рабочем месте в кассе"
                    viol_law = "Постановление Правления Национального Банка РК № 231, п. 15"
                    break

            elif trigger_type in ["CASHIER_STANDUP", "INITIAL_DESK_CHECK"]:
                if ans_low.startswith("да") or "распахнут" in ans_low or "открыт" in ans_low:
                    is_violation = True
                    viol_type = "UNLOCKED_SAFE"
                    viol_desc = "Сейф оставлен открытым / ценности оставлены без присмотра кассира"
                    viol_law = "Постановление Правления Национального Банка РК № 224, п. 31 / № 231, п. 38"
                    break

            elif trigger_type == "THIRD_PERSON":
                if ans_low.startswith("да"):
                    is_violation = True
                    viol_type = "SECURITY_ACCESS"
                    viol_desc = "Присутствие постороннего лица внутри служебной зоны кассы"
                    viol_law = "Постановление Правления Национального Банка РК № 231, п. 12"
                    break

        if is_violation:
            return {
                "verdict": "violation",
                "type": viol_type,
                "description": viol_desc,
                "legal_article": viol_law
            }, observed_facts, dt_total
        else:
            return {
                "verdict": "no_violation",
                "description": "Нарушений не зафиксировано, штатная работа кассы.",
                "legal_article": "-"
            }, observed_facts, dt_total

    def process_video_stream(self, video_path, video_label):
        video_name = os.path.basename(video_path)
        print("=" * 80, flush=True)
        print(f"🎬 БОЕВОЙ АНАЛИЗ ПОТОКА: {video_label} ({video_name})", flush=True)
        print(f"   Файл: {video_path}", flush=True)
        print("=" * 80, flush=True)

        t_start = time.time()
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_f / fps if fps > 0 else 0

        # Шаг 1: Сканирование CV потока через YOLO Pose
        t_cv_0 = time.time()
        events = self.cv_detector.detect_video_events(video_path)
        t_cv = time.time() - t_cv_0
        cv_fps = total_f / t_cv if t_cv > 0 else 0
        print(f"[Stream CV] Просканировано {total_f} кадров со скоростью {cv_fps:.1f} FPS (время: {t_cv:.2f}с).", flush=True)
        print(f"[Stream CV] Обнаружено контрольных окон для VLM: {len(events)}", flush=True)

        video_violations = []

        # Шаг 2: VLM-инспекция контрольных окон
        for i, ev in enumerate(events, 1):
            t_sec = ev["timecode_sec"]
            t_type = ev["trigger"]

            # Для INITIAL_DESK_CHECK инспектируем только если кассир один
            if t_type == "INITIAL_DESK_CHECK" and len(events) > 1:
                continue

            res, facts, dt_vlm = self.verify_event_window(cap, duration, t_type, t_sec)

            verdict = res.get("verdict", "no_violation")
            desc = res.get("description", "-")
            law = res.get("legal_article", "-")
            v_type = res.get("type", t_type)

            print(f"\n[STREAM EVENT {i}/{len(events)}] Окно: {t_sec:5.1f}с [{t_type}] | VLM отклик: {dt_vlm:.2f}с", flush=True)
            print(f"  🔍 Наблюдение VLM: {'; '.join(facts)}", flush=True)
            if verdict == "violation":
                print(f"  🚨 ВЕРДИКТ: ЗАФИКСИРОВАНО НАРУШЕНИЕ", flush=True)
                print(f"     🏷️  Тип: {v_type}", flush=True)
                print(f"     📝 Описание: {desc}", flush=True)
                print(f"     ⚖️  Статья закона: {law}", flush=True)
                video_violations.append({
                    "timecode_sec": t_sec,
                    "type": v_type,
                    "description": desc,
                    "legal_article": law
                })
            else:
                print(f"  🟢 ВЕРДИКТ: НЕТ НАРУШЕНИЯ (Штатная работа кассы)", flush=True)
                print(f"     📝 {desc}", flush=True)

        cap.release()
        total_time = time.time() - t_start
        effective_fps = total_f / total_time if total_time > 0 else 0
        realtime_speedup = (effective_fps / fps) if fps > 0 else 0

        # ИТОГОВЫЙ ОТЧЕТ СРАЗУ ПО ЗАВЕРШЕНИЮ ВИДЕО
        print("\n" + "=" * 80, flush=True)
        print(f"📊 ИТОГОВЫЙ ОТЧЕТ ПО ВИДЕО: {video_label}", flush=True)
        print(f"   ⏱️  Хронометраж видео: {duration:.1f} сек ({total_f} кадров)", flush=True)
        print(f"   ⚡ Время полного сквозного анализа (CV + VLM FP16): {total_time:.2f} сек", flush=True)
        print(f"   🚀 Эффективная скорость: {effective_fps:.1f} FPS (быстрее видео в {realtime_speedup:.1f}x)", flush=True)
        if video_violations:
            print(f"   ❌ СТАТУС СМЕНЫ: ОБНАРУЖЕНЫ ПРИЗНАКИ НАРУШЕНИЯ ({len(video_violations)} инцидент)", flush=True)
            for v in video_violations:
                print(f"      • {v['timecode_sec']:5.1f}с [{v['type']}]: {v['description']}", flush=True)
                print(f"        ⚖️  Закон: {v['legal_article']}", flush=True)
        else:
            print(f"   ✅ СТАТУС СМЕНЫ: НАРУШЕНИЕ НЕ ОБНАРУЖЕНО (Штатная работа кассы)", flush=True)
            print(f"      • Описание: Нарушений не зафиксировано, штатная работа кассы.", flush=True)
            print(f"      • Статья закона: -", flush=True)
        print("=" * 80 + "\n", flush=True)
        return video_violations

if __name__ == "__main__":
    auditor = ProductionCashDeskAuditor()

    # 1. Прогон видео С НАРУШЕНИЕМ: Х-46 (кассир пьет кофе/напиток на рабочем месте)
    auditor.process_video_stream("videos/Х-46.mov", "ВИДЕО С НАРУШЕНИЕМ (Х-46: питье на рабочем месте)")

    # 2. Прогон видео БЕЗ НАРУШЕНИЙ: Х-10 (чистая смена, рука у лица)
    auditor.process_video_stream("videos/Х-10.mov", "ВИДЕО БЕЗ НАРУШЕНИЙ (Х-10: чистая смена)")
