import os, sys, re, cv2, torch, base64, time
import numpy as np
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cv_triggers import PoseEventDetector
from legal_base import LEGAL_ARTICLES_RK

def frame_to_base64(cv2_frame, max_size=720, quality=85):
    h, w = cv2_frame.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        cv2_frame = cv2.resize(cv2_frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    _, buffer = cv2.imencode('.jpg', cv2_frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return base64.b64encode(buffer).decode('utf-8')

# =============================================================================
# ЭТАЛОННЫЙ КАТАЛОГ ВСЕХ ВОЗМОЖНЫХ НАРУШЕНИЙ КАССОВОЙ ДИСЦИПЛИНЫ FORTEBANK
# =============================================================================
ALL_VIOLATIONS_CATALOG_TEXT = """
СПИСОК ВСЕХ ВОЗМОЖНЫХ НАРУШЕНИЙ КАССОВОЙ ДИСЦИПЛИНЫ:
1. [FOOD_DRINK_AT_DESK] — Прием пищи или напитков: кассир держит чашку, кружку, стаканчик, бутылку или еду и пьет/ест на рабочем месте в кассе. (ПП НБРК № 231, п. 15)
2. [CASHIER_DISTRACTED] — Использование смартфона: кассир держит телефон, смотрит в экран или пользуется телефоном за кассой. (ПП НБРК № 231, п. 12)
3. [UNLOCKED_SAFE] — Нарушение режима сейфа: сейф позади кассира оставлен открытым / распахнутым, ключ оставлен в замке, либо кассир не убрала деньги в сейф. (ПП НБРК № 224, п. 31 / ПП № 231, п. 38)
4. [LEAVE_WORKPLACE] — Оставление рабочего места без присмотра: кассир встала/ушла, оставив включенным незаблокированный экран ноутбука, либо документы и печать на столе. (ПП НБРК № 231, п. 38)
5. [UNLOCKED_DOOR] — Нарушение безопасности кассовой кабины: входная дверь кассовой кабины оставлена открытой / распахнутой. (ПП НБРК № 224, п. 12)
6. [THIRD_PERSON_INTRUSION] — Постороннее лицо в кассовом узле: посторонний человек зашел внутрь рабочей зоны кассы или подсматривает через плечо/дверь. (ПП НБРК № 231, п. 12, 14)
7. [MULTIPLE_CLIENTS] — Нарушение тайны операций: одновременное нахождение нескольких клиентов у окна обслуживания. (ПП НБРК № 231, п. 14, 20)
8. [MONEY_LEFT_BETWEEN_CLIENTS] — Оставление денежной наличности: на столе оставлена пачка денег без присмотра / деньги прошлого клиента при приеме следующего. (ПП НБРК № 231, п. 27, 38)
9. [NO_ID_VERIFICATION] — Нарушение порядка приема/выдачи наличности: прием/выдача денег или инкассаторской сумки без проверки документов, либо пересчет вне видимости клиента после его ухода. (ПП НБРК № 231, п. 19, 23, 53)
10. [NO_COUNTING_MACHINE] — Нарушение кассовой дисциплины: банкноты, отбракованные счетной машинкой, соединены с нормальной пачкой. (ПП НБРК № 231, п. 24)
11. [CLIENT_DESK_ACCESS] — Нарушение возврата наличности клиенту: кассир не вернула сдачу клиенту или присвоила часть денег себе. (ПП НБРК № 231, п. 25, 27)
12. [UNAUTHORIZED_ITEMS] — Нарушение чистоты рабочей зоны: на столе кассира лежит личный рюкзак, сумка, куртка или посторонние вещи. (ПП НБРК № 231, п. 15)
"""

VIOLATIONS_META = {
    "FOOD_DRINK_AT_DESK": {
        "title": "Прием пищи или напитков на рабочем месте",
        "law": "Постановление Правления НБРК № 231, Глава 2, п. 12, 15",
        "pos": ["food_drink_at_desk", "кружк", "чашк", "стаканчик", "бутылк", "напиток", "пьет", "ест"],
        "neg": ["нет кружки", "нет чашки", "рука пустая", "рука абсолютно пустая", "без кружки", "без чашки", "пустая", "не держит", "не пьет", "ничего нет"]
    },
    "CASHIER_DISTRACTED": {
        "title": "Использование смартфона кассиром на рабочем месте",
        "law": "Постановление Правления НБРК № 231, Глава 2, п. 12",
        "pos": ["cashier_distracted", "phone_usage", "смартфон", "телефон", "пользуется телефоном", "набирает текст"],
        "neg": ["не пользуется телефоном", "нет телефона", "без телефона", "не держит телефон", "руки на столе", "на клавиатуре"]
    },
    "UNLOCKED_SAFE": {
        "title": "Нарушение режима хранения ценностей: незапертый сейф",
        "law": "Постановление Правления НБРК № 224, п. 31 / ПП № 231, п. 38",
        "pos": ["unlocked_safe", "распахнута", "дверца открыта", "ключ торчит в замке", "ключ оставлен в замке", "открыт сейф", "не убрала в сейф", "не убраны в сейф"],
        "neg": ["дверца закрыта", "сейф закрыт", "заперт", "не открыта", "нет ключа", "без ключа", "ключ не торчит", "ключа нет"]
    },
    "LEAVE_WORKPLACE": {
        "title": "Оставление рабочего места без блокировки / ценностей",
        "law": "Постановление Правления НБРК № 231, Глава 2, п. 38",
        "pos": ["leave_workplace", "экран светится", "экран включен", "незаблокирован", "печать на столе", "документы на столе", "служебные документы"],
        "neg": ["темную", "темная", "предмет", "выключен", "не светится", "закрыт", "стол чист", "нет документов"]
    },
    "UNLOCKED_DOOR": {
        "title": "Нарушение безопасности кассовой кабины: открытая дверь",
        "law": "Постановление Правления НБРК № 224, п. 12",
        "pos": ["unlocked_door", "дверь открыта", "дверь распахнута", "оставлена открытой", "настежь", "распахнута"],
        "neg": ["дверь закрыта", "закрыта", "заперта", "прикрыта", "не открыта"]
    },
    "THIRD_PERSON_INTRUSION": {
        "title": "Нарушение пропускного режима: посторонний в кассовом узле",
        "law": "Постановление Правления НБРК № 231, Глава 2, п. 12, 14",
        "pos": ["third_person_intrusion", "два клиента", "двое клиентов", "три человека", "подсматривает", "заглядывает", "посторонний", "третий человек", "второй клиент", "зашел внутрь"],
        "neg": ["нет посторонних", "один клиент", "один человек", "нельзя определить", "не видно"]
    },
    "MULTIPLE_CLIENTS": {
        "title": "Нарушение тайны операций: несколько клиентов у окна",
        "law": "Постановление Правления НБРК № 231, п. 14, 20",
        "pos": ["multiple_clients", "два клиента", "двое клиентов", "несколько клиентов"],
        "neg": ["один клиент", "нет клиентов"]
    },
    "MONEY_LEFT_BETWEEN_CLIENTS": {
        "title": "Оставление денежной наличности без присмотра на столе",
        "law": "Постановление Правления НБРК № 231, п. 27, 38",
        "pos": ["money_left_between_clients", "лежат деньги", "пачка денег", "купюры", "наличные на столе", "на столе лежат", "деньги на столе", "остались неубранные деньги", "от предыдущего клиента"],
        "neg": ["нет денег", "стол чист", "стол пустой", "убрала деньги в сейф", "денег нет", "не лежат", "без денег"]
    },
    "NO_ID_VERIFICATION": {
        "title": "Нарушение порядка идентификации клиента / пересчета наличности",
        "law": "Постановление Правления НБРК № 231, п. 19, 23, 53-55",
        "pos": ["no_id_verification", "без проверки документов", "без паспорта", "удостак", "инкассаторский мешок", "зеленый мешок", "сразу выложена пачка денег", "пересчет вне поля зрения", "пересчитывает на машинке вне"],
        "neg": ["проверила документы", "в поле зрения", "в присутствии клиента", "обычный клиент", "стол пустой", "нет мешка"]
    },
    "NO_COUNTING_MACHINE": {
        "title": "Грубое нарушение кассовой дисциплины: объединение отбракованных купюр",
        "law": "Постановление Правления НБРК № 231, Глава 3, п. 24",
        "pos": ["no_counting_machine", "соединила", "смешала", "объединила отбракованные"],
        "neg": ["не соединила", "не смешивала", "в порядке"]
    },
    "CLIENT_DESK_ACCESS": {
        "title": "Нарушение регламента возврата наличности клиенту",
        "law": "Постановление Правления НБРК № 231, п. 25, 27",
        "pos": ["client_desk_access", "оставила себе часть", "спрятала часть денег", "убрала себе часть", "не отдала клиенту", "не вернула сдачу", "присвоила"],
        "neg": ["вернула полностью", "отдала клиенту", "передала клиенту", "не присвоила", "сдала сдачу", "все отдала", "без нарушений", "передает через лоток"]
    },
    "UNAUTHORIZED_ITEMS": {
        "title": "Нарушение чистоты рабочей зоны: посторонние вещи на столе",
        "law": "Постановление Правления НБРК № 231, п. 15",
        "pos": ["unauthorized_items", "лежит куртка", "на столе куртка", "куртка на столе", "снятая куртка", "одежда на столе", "лежит рюкзак", "большая сумка на столе", "инкассаторский мешок"],
        "neg": ["нет куртки", "нет сумки", "без куртки", "без рюкзака", "чист", "ничего нет", "не лежит", "не обнаружено", "пустой", "стол чист"]
    }
}

class BankViolationDetector:
    def __init__(self, model_name="Qwen/Qwen2-VL-2B-Instruct"):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"[BankViolationDetector] Загрузка VLM ({model_name})...")
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if "cuda" in self.device else torch.float32,
            device_map=self.device
        ).eval()
        vram = torch.cuda.memory_allocated() / (1024**2) if "cuda" in self.device else 0
        print(f"[BankViolationDetector] VLM готова! Занято VRAM: {vram:.0f} MB")
        
        self.pose_detector = PoseEventDetector()
        print("[BankViolationDetector] PoseEventDetector готов к высокоскоростному сканированию!")

    def ask_vlm(self, image, prompt, max_new_tokens=100):
        messages = [{'role': 'user', 'content': [{'type': 'image', 'image': image}, {'type': 'text', 'text': prompt}]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        img_inputs, vid_inputs = process_vision_info(messages)
        inputs = self.processor(text=[text], images=img_inputs, videos=vid_inputs, padding=True, return_tensors='pt').to(self.device)
        with torch.no_grad():
            gen_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        ans = self.processor.batch_decode(gen_ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0].strip()
        ans_lower = ans.lower()
        if any(r in ans_lower for r in ["извините, но я не могу", "моя функция - предоставлять информацию", "не имею возможности", "как текстовый ai"]):
            return "не определено (отказ модели)"
        return ans

    def ask_vlm_for_violations(self, image, context_hint="", candidate_types=None):
        """
        УНИВЕРСАЛЬНЫЙ ПРОМПТ ВЫБОРА НАРУШЕНИЙ ИЗ ВСЕГО СПИСКА
        Дает VLM выбрать одно или несколько нарушений из каталога всех возможных нарушений.
        """
        prompt = f"""Ты — объективный ИИ-ревизор службы безопасности и комплаенс-контроля ForteBank.
Оцени обстановку и действия кассира и клиентов на этом кадре видеонаблюдения рабочего места кассы банка.
{context_hint}

{ALL_VIOLATIONS_CATALOG_TEXT}

ИНСТРУКЦИЯ:
- Если действия полностью штатные (кассир работает, руки свободны, считает деньги, сейф закрыт, стол чист) — ответь строго:
ШТАТНО

- Если на кадре зафиксировано нарушение(я) — выбери ОДНО ИЛИ НЕСКОЛЬКО нарушений ИСКЛЮЧИТЕЛЬНО ИЗ СПИСКА ВЫШЕ и укажи их в формате:
НАРУШЕНИЕ: [КОД_НАРУШЕНИЯ] — <что именно видно на кадре>
"""
        ans = self.ask_vlm(image, prompt, max_new_tokens=120)
        ans_lower = ans.lower()

        detected = []
        is_clean = "штатно" in ans_lower and "нарушение" not in ans_lower

        if not is_clean:
            check_keys = candidate_types if candidate_types else VIOLATIONS_META.keys()
            for v_code in check_keys:
                meta = VIOLATIONS_META.get(v_code)
                if not meta:
                    continue
                has_code = f"[{v_code.lower()}]" in ans_lower or v_code.lower() in ans_lower
                has_pos = any(w in ans_lower for w in meta["pos"])
                has_neg = any(w in ans_lower for w in meta["neg"])
                if (has_code or has_pos) and not has_neg:
                    detected.append((v_code, meta["title"], meta["law"], ans))

        return detected, ans

    def get_evidence_frames(self, cap, t_sec, duration, count=2):
        frames = []
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0 or np.isnan(fps):
            fps = 25.0
        offsets = [-0.5, 0.0, 1.0] if count == 3 else [0.0, 1.0]
        for dt in offsets:
            t = max(0.0, min(duration - 0.1, t_sec + dt))
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ret, frame = cap.read()
            if not ret or frame is None:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
                ret, frame = cap.read()
            if ret and frame is not None:
                frames.append({
                    "timestamp_sec": round(float(t), 1),
                    "image_base64": frame_to_base64(frame)
                })
        return frames

    def get_rgb_frame(self, cap, t_sec, duration, max_size=896):
        t = max(0.0, min(duration - 0.1, t_sec))
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ret, frame = cap.read()
        if not ret or frame is None:
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
            ret, frame = cap.read()
        if not ret or frame is None:
            return None
        h, w = frame.shape[:2]
        scale = max_size / max(h, w)
        return cv2.cvtColor(cv2.resize(frame, (int(w * scale), int(h * scale))), cv2.COLOR_BGR2RGB)

    def analyze_file(self, video_path, filename=None):
        """
        АЛГОРИТМИЧЕСКИЙ ИИ-АНАЛИЗ НА БАЗЕ ВЫБОРА НАРУШЕНИЙ ИЗ ПОЛНОГО КАТАЛОГА VLM.
        1. YOLO Pose сканирует биомеханику, перемещения и жесты кассира и клиентов.
        2. Qwen2-VL выбирает одно или несколько нарушений из полного списка регламента РК.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Невозможно открыть видео: {video_path}")
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0

        events = self.pose_detector.detect_video_events(video_path)
        detected_violations = []

        def is_duplicate(v_type, t_sec, min_interval=6.0):
            for dv in detected_violations:
                if dv["type"] == v_type and abs(dv["timecode_sec"] - t_sec) < min_interval:
                    return True
            return False

        def register_violation(v_code, t_time, desc, law_article):
            if not is_duplicate(v_code, t_time):
                ev_frames = self.get_evidence_frames(cap, t_time, duration)
                detected_violations.append({
                    "type": v_code,
                    "timecode_sec": round(float(t_time), 1),
                    "description": f"{desc}. Статья: {law_article}",
                    "evidence_frames": ev_frames
                })

        for ev in events:
            t_ev = ev["timecode_sec"]
            trig = ev["trigger"]
            merged = ev.get("merged_triggers", [trig])

            # =========================================================================
            # Событие 1: Рука у лица / жестикуляция (ПП НБРК № 231, п. 12, 15)
            # =========================================================================
            if trig == "HAND_TO_MOUTH" or "HAND_TO_MOUTH" in merged:
                t_check = min(duration - 0.5, t_ev + 3.0)
                rgb = self.get_rgb_frame(cap, t_check, duration)
                if rgb is not None:
                    found, raw_ans = self.ask_vlm_for_violations(
                        Image.fromarray(rgb),
                        context_hint="Кассир подносит руку к лицу или голове.",
                        candidate_types=["FOOD_DRINK_AT_DESK", "CASHIER_DISTRACTED"]
                    )
                    for v_code, title, law, ans in found:
                        register_violation(v_code, t_ev, f"{title} ({ans})", law)

            # =========================================================================
            # Событие 2: Начало смены / Чистота стола (ПП НБРК № 231, п. 15)
            # =========================================================================
            if trig == "INITIAL_DESK_CHECK":
                t_desk = min(duration - 1.0, 17.0) if duration > 15.0 else 0.5
                rgb = self.get_rgb_frame(cap, t_desk, duration)
                if rgb is not None:
                    found, raw_ans = self.ask_vlm_for_violations(
                        Image.fromarray(rgb),
                        context_hint="Рабочий стол кассы в начале работы кассира.",
                        candidate_types=["UNAUTHORIZED_ITEMS"]
                    )
                    for v_code, title, law, ans in found:
                        register_violation(v_code, t_desk, f"Нарушение чистоты рабочей зоны: {ans}", law)

            # =========================================================================
            # Событие 3: Третий человек / Посторонний в кассе / Подглядывание (ПП НБРК № 231)
            # =========================================================================
            if trig == "THIRD_PERSON" or "THIRD_PERSON" in merged:
                rgb = self.get_rgb_frame(cap, t_ev, duration)
                if rgb is not None:
                    found, raw_ans = self.ask_vlm_for_violations(
                        Image.fromarray(rgb),
                        context_hint="Зона обслуживания кассы и входная дверь кассовой кабины.",
                        candidate_types=["THIRD_PERSON_INTRUSION", "MULTIPLE_CLIENTS", "UNLOCKED_DOOR"]
                    )
                    for v_code, title, law, ans in found:
                        register_violation(v_code, t_ev, f"{title}: {ans}", law)

            # =========================================================================
            # Событие 4: Оставление рабочего места / Сейф / Экран (ПП НБРК № 231, 224)
            # =========================================================================
            if trig == "CASHIER_STANDUP" or "CASHIER_STANDUP" in merged:
                t_check = min(duration - 0.1, t_ev + 1.0)
                rgb = self.get_rgb_frame(cap, t_check, duration)
                if rgb is not None:
                    p_sit = "Сидит ли кассир за рабочим столом или кассир покинула рабочее место/встала и отошла?"
                    ans_sit = self.ask_vlm(Image.fromarray(rgb), p_sit).lower()
                    is_sitting = any(w in ans_sit for w in ["сидит за столом", "сидит на стуле", "кассир сидит"]) and not any(neg in ans_sit for neg in ["не сидит", "но не сидит", "не за столом"])
                    cashier_is_away = any(w in ans_sit for w in ["покинула", "отошла", "встала", "не сидит", "нет кассира", "место пустое", "пустой стул", "вышла"]) and not is_sitting

                    if duration - t_ev > 25.0:
                        rgb_lat = self.get_rgb_frame(cap, t_ev + 25.0, duration)
                        if rgb_lat is not None:
                            ans_lat = self.ask_vlm(Image.fromarray(rgb_lat), "Сидит ли кассир за рабочим столом?").lower()
                            if any(w in ans_lat for w in ["сидит", "да, кассир", "за столом"]) and not any(n in ans_lat for n in ["не сидит", "не видно", "не позволяет"]):
                                cashier_is_away = False

                    if cashier_is_away:
                        found, raw_ans = self.ask_vlm_for_violations(
                            Image.fromarray(rgb),
                            context_hint="Кассир встала и отошла от рабочего стола. Проверь дверцу сейфа позади стола, экран ноутбука, оставленные деньги/документы на столе и входную дверь кабины.",
                            candidate_types=["UNLOCKED_SAFE", "LEAVE_WORKPLACE", "MONEY_LEFT_BETWEEN_CLIENTS", "UNLOCKED_DOOR", "UNAUTHORIZED_ITEMS"]
                        )
                        for v_code, title, law, ans in found:
                            register_violation(v_code, t_ev, f"{title}: {ans}", law)

            # =========================================================================
            # Событие 5: Завершение обслуживания клиента / Уход (ПП НБРК № 231)
            # =========================================================================
            if trig == "CLIENT_DEPARTURE" or "CLIENT_DEPARTURE" in merged:
                rgb = self.get_rgb_frame(cap, min(duration - 0.2, t_ev + 1.0), duration)
                if rgb is not None:
                    found, raw_ans = self.ask_vlm_for_violations(
                        Image.fromarray(rgb),
                        context_hint="Клиент отошел от кассы. Проверь, убраны ли деньги в сейф, не оставлена ли наличность на столе, и не производит ли кассир пересчет денег вне поля зрения клиента.",
                        candidate_types=["UNLOCKED_SAFE", "MONEY_LEFT_BETWEEN_CLIENTS", "NO_ID_VERIFICATION"]
                    )
                    for v_code, title, law, ans in found:
                        register_violation(v_code, t_ev, f"{title}: {ans}", law)

            # =========================================================================
            # Событие 6: Передача ценностей, машинный пересчет, сдача (ПП НБРК № 231)
            # =========================================================================
            if trig == "HANDOVER" or "HANDOVER" in merged:
                rgb = self.get_rgb_frame(cap, min(duration - 0.2, t_ev + 2.0), duration)
                if rgb is not None:
                    found, raw_ans = self.ask_vlm_for_violations(
                        Image.fromarray(rgb),
                        context_hint="Передача денег между клиентом и кассиром, работа со счетной машинкой на столе.",
                        candidate_types=["CLIENT_DESK_ACCESS", "NO_COUNTING_MACHINE", "MONEY_LEFT_BETWEEN_CLIENTS"]
                    )
                    for v_code, title, law, ans in found:
                        register_violation(v_code, t_ev, f"{title}: {ans}", law)

            # =========================================================================
            # Событие 7: Приход клиента / Инкассация (ПП НБРК № 231, 224)
            # =========================================================================
            if trig == "CLIENT_ARRIVAL" or "CLIENT_ARRIVAL" in merged:
                rgb = self.get_rgb_frame(cap, t_ev + 1.0, duration)
                if rgb is not None:
                    found, raw_ans = self.ask_vlm_for_violations(
                        Image.fromarray(rgb),
                        context_hint="К кассе подошел посетитель или инкассатор. Проверь наличие денег от прошлого клиента, инкассаторского мешка, выкладку купюр без проверки документов или проникновение в кабину.",
                        candidate_types=["MONEY_LEFT_BETWEEN_CLIENTS", "NO_ID_VERIFICATION", "UNLOCKED_DOOR", "THIRD_PERSON_INTRUSION"]
                    )
                    for v_code, title, law, ans in found:
                        register_violation(v_code, t_ev, f"{title}: {ans}", law)

        cap.release()

        return {
            "verdict": "violation" if len(detected_violations) > 0 else "no_violation",
            "violations": detected_violations
        }

if __name__ == "__main__":
    detector = BankViolationDetector()
    test_v = 'E:/open/videos/Х-3.mov'
    if os.path.exists(test_v):
        res = detector.analyze_file(test_v)
        print("Verdict:", res["verdict"])
        print("Violations:", len(res["violations"]))
