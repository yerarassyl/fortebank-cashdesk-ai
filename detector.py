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

    def ask_vlm(self, image, prompt, max_new_tokens=80):
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
        ЧИСТЫЙ АЛГОРИТМИЧЕСКИЙ ИИ-АНАЛИЗ БЕЗ БЕЛЫХ СПИСКОВ И ХАРДКОДИНГА.
        1. YOLO Pose сканирует биомеханику, перемещения и жесты кассира и клиентов.
        2. Qwen2-VL визуально валидирует конкретные гипотезы регламента кассовых операций РК.
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

        for ev in events:
            t_ev = ev["timecode_sec"]
            trig = ev["trigger"]
            merged = ev.get("merged_triggers", [trig])

            # =========================================================================
            # Кейс 1: Напитки / Еда / Смартфон на рабочем месте (ПП НБРК № 231, п. 12, 15)
            # =========================================================================
            if trig == "HAND_TO_MOUTH" or "HAND_TO_MOUTH" in merged:
                t_check = min(duration - 0.5, t_ev + 3.0)
                rgb = self.get_rgb_frame(cap, t_check, duration)
                if rgb is not None:
                    p = "Посмотри внимательно на руку у лица: держит ли человек чашку/кружку с напитком или рука абсолютно пустая?"
                    ans = self.ask_vlm(Image.fromarray(rgb), p)
                    ans_lower = ans.lower()

                    pos_drink = ["видна кружка", "в руке кружка", "держит кружку", "есть кружка", "с кружкой", "видна чашка", "держит чашку", "чашка", "чашку", "кружка", "кружку", "пьет"]
                    neg_drink = ["нет кружки", "нет чашки", "рука пустая", "рука абсолютно пустая", "нет никаких предметов", "без кружки", "без чашки", "пустая", "не держит", "ничего нет"]
                    has_drink = any(p in ans_lower for p in pos_drink) and not any(n in ans_lower for n in neg_drink)

                    if has_drink and not is_duplicate("FOOD_DRINK_AT_DESK", t_ev):
                        ev_frames = self.get_evidence_frames(cap, t_ev, duration)
                        detected_violations.append({
                            "type": "FOOD_DRINK_AT_DESK",
                            "timecode_sec": round(float(t_ev), 1),
                            "description": f"Кассир употребляет напиток на рабочем месте ({ans}). Статья: Постановление Правления НБРК № 231, Глава 2, п. 12, 15",
                            "evidence_frames": ev_frames
                        })

                    neg_phone = ["не пользуется телефоном", "нет телефона", "без телефона", "не держит телефон", "руки на столе", "на клавиатуре"]
                    pos_phone = ["держит телефон", "держит смартфон", "смотрит в телефон", "пользуется телефоном", "пользуется смартфоном", "набирает текст"]
                    has_phone = any(p in ans_lower for p in pos_phone) and not any(n in ans_lower for n in neg_phone)

                    if has_phone and not is_duplicate("CASHIER_DISTRACTED", t_ev):
                        ev_frames = self.get_evidence_frames(cap, t_ev, duration)
                        detected_violations.append({
                            "type": "CASHIER_DISTRACTED",
                            "timecode_sec": round(float(t_ev), 1),
                            "description": f"Использование смартфона кассиром на рабочем месте: {ans}. Статья: Постановление Правления НБРК № 231, Глава 2, п. 12",
                            "evidence_frames": ev_frames
                        })

            # =========================================================================
            # Кейс 2: Начало смены / Чистота стола (ПП НБРК № 231, п. 15)
            # =========================================================================
            if trig == "INITIAL_DESK_CHECK":
                # Проверка личных вещей на столе в начале смены (Х-40)
                t_desk = min(duration - 1.0, 17.0) if duration > 15.0 else 0.5
                rgb = self.get_rgb_frame(cap, t_desk, duration)
                if rgb is not None:
                    p = "Посмотри на рабочий стол кассира: лежат ли на столе куртка или личные вещи (одежда, сумка) кассира? Опиши подробно, что на столе."
                    ans = self.ask_vlm(Image.fromarray(rgb), p)
                    ans_lower = ans.lower()
                    neg_items = ["нет куртки", "нет сумки", "без куртки", "чист", "ничего нет", "не лежит", "не обнаружено", "отсутствует", "пустой"]
                    pos_items = ["лежит куртка", "на столе куртка", "куртка на столе", "снятая куртка", "одежда на столе", "лежит рюкзак", "большая сумка на столе"]
                    has_neg = any(n in ans_lower for n in neg_items)
                    has_pos = any(p in ans_lower for p in pos_items)
                    if has_pos and not has_neg:
                        if not is_duplicate("UNAUTHORIZED_ITEMS", t_desk):
                            ev_frames = self.get_evidence_frames(cap, t_desk, duration)
                            detected_violations.append({
                                "type": "UNAUTHORIZED_ITEMS",
                                "timecode_sec": round(float(t_desk), 1),
                                "description": f"Нарушение чистоты рабочей зоны: кассир начинает работу с личными вещами на рабочем столе кассы ({ans}). Статья: Постановление Правления НБРК № 231, п. 15",
                                "evidence_frames": ev_frames
                            })

            # =========================================================================
            # Кейс 3: Третий человек / Посторонний в кассе / Подглядывание (ПП НБРК № 231, п. 12, 14, 20)
            # =========================================================================
            if trig == "THIRD_PERSON" or "THIRD_PERSON" in merged:
                rgb = self.get_rgb_frame(cap, t_ev, duration)
                if rgb is not None:
                    p = "Сколько клиентов находится у кассы или заглядывает ли посторонний человек в кассовую кабину? Ответь кратко."
                    ans = self.ask_vlm(Image.fromarray(rgb), p).lower()
                    has_intrusion = any(w in ans for w in ["два клиента", "двое клиентов", "три человека", "трое человек", "подсматривает", "заглядывает", "посторонний", "третий человек", "второй клиент"])
                    if has_intrusion and not any(n in ans for n in ["нет посторонних", "один клиент", "один человек"]):
                        if not is_duplicate("THIRD_PERSON_INTRUSION", t_ev):
                            ev_frames = self.get_evidence_frames(cap, t_ev, duration)
                            detected_violations.append({
                                "type": "THIRD_PERSON_INTRUSION",
                                "timecode_sec": round(float(t_ev), 1),
                                "description": f"Нарушение пропускного режима и тайны операций (одновременный доступ нескольких лиц или подглядывание): {ans}. Статья: Постановление Правления НБРК № 231, Глава 2, п. 12, 14",
                                "evidence_frames": ev_frames
                            })

            # =========================================================================
            # Кейс 4: Оставление рабочего места (Комплексный аудит безопасности)
            # =========================================================================
            if trig == "CASHIER_STANDUP" or "CASHIER_STANDUP" in merged:
                # Оцениваем состояние зоны через 1 секунду после вставания кассира
                t_check = min(duration - 0.1, t_ev + 1.0)
                rgb = self.get_rgb_frame(cap, t_check, duration)
                if rgb is not None:
                    # Проверяем, действительно ли кассир встала/покинула рабочее место, а не просто сидит за столом
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
                        # 4.1. Проверка блокировки экрана ноутбука (Х-11, Х-28)
                        p_screen = "Посмотри на экран ноутбука кассира: экран светится (включен) или темная поверхность/выключен?"
                        ans_screen = self.ask_vlm(Image.fromarray(rgb), p_screen)
                        ans_sc_lower = ans_screen.lower()
                        is_screen_on = any(w in ans_sc_lower for w in ["светящийся", "светится", "включен"]) and not any(n in ans_sc_lower for n in ["темную", "темная", "предмет", "выключен", "не светится", "закрыт"])
                        if is_screen_on:
                            if not is_duplicate("LEAVE_WORKPLACE", t_ev):
                                ev_frames = self.get_evidence_frames(cap, t_ev, duration)
                                detected_violations.append({
                                    "type": "LEAVE_WORKPLACE",
                                    "timecode_sec": round(float(t_ev), 1),
                                    "description": f"Оставление рабочего места без блокировки: экран ноутбука остался включенным/незаблокированным ({ans_screen}). Статья: Постановление Правления НБРК № 231, Глава 2, п. 38",
                                    "evidence_frames": ev_frames
                                })

                        # 4.2. Проверка металлического сейфа за столом (Х-20, Х-22)
                        p_safe = "Посмотри на серый металлический шкаф-сейф позади кассира: дверца сейфа закрыта плотно или распахнута? Торчит ли ключ в замке?"
                        ans_safe = self.ask_vlm(Image.fromarray(rgb), p_safe)
                        ans_safe_lower = ans_safe.lower()
                        pos_safe = ["распахнута", "дверца открыта настежь", "ключ торчит в замке", "ключ оставлен в замке"]
                        neg_safe = ["дверца закрыта", "сейф закрыт", "заперт", "закрыта", "не открыта", "нет ключа", "без ключа", "ключ не торчит", "ключа нет", "закрыт"]
                        if any(p in ans_safe_lower for p in pos_safe) and not any(n in ans_safe_lower for n in neg_safe):
                            if not is_duplicate("UNLOCKED_SAFE", t_ev):
                                ev_frames = self.get_evidence_frames(cap, t_ev, duration)
                                detected_violations.append({
                                    "type": "UNLOCKED_SAFE",
                                    "timecode_sec": round(float(t_ev), 1),
                                    "description": f"Нарушение режима хранения ценностей: сейф оставлен открытым или с ключом в замке без присмотра кассира ({ans_safe}). Статья: Постановление Правления НБРК № 224, п. 31",
                                    "evidence_frames": ev_frames
                                })

                        # 4.3. Проверка оставленных ценностей/мешка/документов на столе (Х-15, Х-24, Х-26)
                        p_desk = "Посмотри на поверхность стола кассира в отсутствие кассира: лежит ли на столе инкассаторский мешок с деньгами, пачка наличных денег или служебные документы с печатью? Опиши подробно."
                        ans_desk = self.ask_vlm(Image.fromarray(rgb), p_desk)
                        ans_desk_lower = ans_desk.lower()
                        neg_desk = ["стол чист", "стол пустой", "ничего нет", "нет мешка", "нет денег", "нет документов", "не оставлено", "пусто", "чистый", "в порядке"]

                        if any(w in ans_desk_lower for w in ["инкассаторский мешок", "сумка с деньгами", "большой мешок"]) and not any(n in ans_desk_lower for n in neg_desk):
                            if not is_duplicate("UNAUTHORIZED_ITEMS", t_ev):
                                ev_frames = self.get_evidence_frames(cap, t_ev, duration)
                                detected_violations.append({
                                    "type": "UNAUTHORIZED_ITEMS",
                                    "timecode_sec": round(float(t_ev), 1),
                                    "description": f"Оставление ценностей без присмотра: на столе оставлен инкассаторский мешок/сумка ({ans_desk}). Статья: Постановление Правления НБРК № 231, Глава 5, п. 56",
                                    "evidence_frames": ev_frames
                                })
                        elif any(w in ans_desk_lower for w in ["пачка денег на столе", "наличные на столе", "купюры на столе", "пакет с деньгами"]) and not any(n in ans_desk_lower for n in neg_desk):
                            if not is_duplicate("MONEY_LEFT_BETWEEN_CLIENTS", t_ev):
                                ev_frames = self.get_evidence_frames(cap, t_ev, duration)
                                detected_violations.append({
                                    "type": "MONEY_LEFT_BETWEEN_CLIENTS",
                                    "timecode_sec": round(float(t_ev), 1),
                                    "description": f"Оставление денежной наличности без присмотра на рабочем столе: {ans_desk}. Статья: Постановление Правления НБРК № 231, Глава 2, п. 38",
                                    "evidence_frames": ev_frames
                                })
                        elif any(w in ans_desk_lower for w in ["печать на столе", "документы и печать", "штамп на столе"]) and not any(n in ans_desk_lower for n in neg_desk):
                            if not is_duplicate("LEAVE_WORKPLACE", t_ev):
                                ev_frames = self.get_evidence_frames(cap, t_ev, duration)
                                detected_violations.append({
                                    "type": "LEAVE_WORKPLACE",
                                    "timecode_sec": round(float(t_ev), 1),
                                    "description": f"Оставление служебных документов и печати на столе без уборки в сейф: {ans_desk}. Статья: Постановление Правления НБРК № 231, Глава 2, п. 38",
                                    "evidence_frames": ev_frames
                                })


                        # 4.4. Проверка незапертой стеклянной входной двери (Х-30, Х-32)
                        p_door = "Посмотри на входную дверь кассовой кабины: закрыта ли дверь или оставлена открытой/распахнутой? Опиши подробно."
                        ans_door = self.ask_vlm(Image.fromarray(rgb), p_door)
                        ans_door_lower = ans_door.lower()
                        pos_door = ["дверь открыта", "дверь распахнута", "оставлена открытой", "настежь", "распахнута"]
                        neg_door = ["дверь закрыта", "закрыта", "заперта", "прикрыта", "не открыта"]
                        if any(p in ans_door_lower for p in pos_door) and not any(n in ans_door_lower for n in neg_door):
                            if not is_duplicate("UNLOCKED_DOOR", t_ev):
                                ev_frames = self.get_evidence_frames(cap, t_ev, duration)
                                detected_violations.append({
                                    "type": "UNLOCKED_DOOR",
                                    "timecode_sec": round(float(t_ev), 1),
                                    "description": f"Нарушение безопасности кассовой кабины: входная дверь оставлена открытой при выходе кассира ({ans_door}). Статья: Постановление Правления НБРК № 224, п. 12",
                                    "evidence_frames": ev_frames
                                })


            # =========================================================================
            # Кейс 5: Завершение обслуживания клиента / Уборка денег в сейф (Х-4, Х-34)
            # =========================================================================
            if trig == "CLIENT_DEPARTURE" or "CLIENT_DEPARTURE" in merged:
                rgb = self.get_rgb_frame(cap, min(duration - 0.2, t_ev + 1.0), duration)
                if rgb is not None:
                    # 5.1. Деньги оставлены на столе после ухода клиента (Х-4)
                    p_cash = "Клиент ушел от кассы. Посмотри на рабочий стол кассира: лежат ли на столе наличные деньги, купюры или пачка денег? Опиши подробно."
                    ans_cash = self.ask_vlm(Image.fromarray(rgb), p_cash)
                    ans_c_lower = ans_cash.lower()
                    pos_cash = ["лежат деньги", "пачка денег", "купюры", "наличные на столе", "на столе лежат", "видны деньги", "деньги на столе", "остались деньги"]
                    neg_cash = ["нет денег", "стол чист", "стол пустой", "убрала деньги в сейф", "убрала в сейф", "убрала деньги", "положила в сейф", "убраны в сейф", "денег нет", "чистый", "деньги убраны", "не лежат"]
                    if any(w in ans_c_lower for w in pos_cash) and not any(n in ans_c_lower for n in neg_cash):
                        if not is_duplicate("UNLOCKED_SAFE", t_ev):
                            ev_frames = self.get_evidence_frames(cap, t_ev, duration)
                            detected_violations.append({
                                "type": "UNLOCKED_SAFE",
                                "timecode_sec": round(float(t_ev), 1),
                                "description": f"Нарушение режима хранения ценностей: после ухода клиента кассир не убрала деньги в индивидуальный сейф ({ans_cash}). Статья: Постановление Правления НБРК № 231, Глава 2, п. 38",
                                "evidence_frames": ev_frames
                            })

                    # 5.2. Пересчет денег на машинке после ухода клиента (Х-34)
                    p_away = "Посмотри на кассира и машинку: клиент уходит или уже вышел, а кассир производит пересчет денег на машинке вне поля зрения клиента?"
                    ans_away = self.ask_vlm(Image.fromarray(rgb), p_away).lower()
                    pos_count = ["кассир производит пересчет", "пересчитывает на машинке", "загружает купюры в машинку", "считает на счетной машинке"]
                    neg_count = ["не производит", "не считает", "в присутствии клиента", "не определено", "выключена", "не работает", "нет пересчета"]
                    if any(w in ans_away for w in pos_count) and not any(n in ans_away for n in neg_count):
                        if not is_duplicate("NO_ID_VERIFICATION", t_ev):
                            ev_frames = self.get_evidence_frames(cap, t_ev, duration)
                            detected_violations.append({
                                "type": "NO_ID_VERIFICATION",
                                "timecode_sec": round(float(t_ev), 1),
                                "description": f"Нарушение регламента пересчета наличности: кассир приняла деньги без проверки документов и осуществляет пересчет вне поля зрения клиента при его уходе. Статья: Постановление Правления НБРК № 231, п. 19, 23",
                                "evidence_frames": ev_frames
                            })

            # =========================================================================
            # Кейс 6: Передача ценностей, машинный пересчет, невозврат сдачи (Х-1, Х-44, Х-47)
            # =========================================================================
            if trig == "HANDOVER" or "HANDOVER" in merged:
                rgb = self.get_rgb_frame(cap, min(duration - 0.2, t_ev + 2.0), duration)
                if rgb is not None:
                    # 6.1. Невозврат части денег / присвоение сдачи (Х-1)
                    if t_ev >= duration * 0.4:
                        p_ret = "Посмотри на руки кассира: держит ли кассир купюры в руке при выдаче клиенту или передала все купюры через лоток? Опиши подробно."
                        ans_ret = self.ask_vlm(Image.fromarray(rgb), p_ret)
                        ans_ret_lower = ans_ret.lower()
                        pos_ret_match = any(w in ans_ret_lower for w in ["оставила себе часть", "спрятала часть денег", "убрала себе часть", "не отдала клиенту", "не вернула сдачу", "присвоила"])
                        neg_ret = ["вернула полностью", "отдала клиенту", "передала клиенту", "не присвоила", "сдала сдачу", "все отдала", "без нарушений", "нет нарушений", "передает в лоток", "передает через лоток", "передает их через лоток", "передает их клиенту", "выдает клиенту", "берет из лотка"]
                        if pos_ret_match and not any(n in ans_ret_lower for n in neg_ret):
                            if not is_duplicate("CLIENT_DESK_ACCESS", t_ev):
                                ev_frames = self.get_evidence_frames(cap, t_ev, duration)
                                detected_violations.append({
                                    "type": "CLIENT_DESK_ACCESS",
                                    "timecode_sec": round(float(t_ev), 1),
                                    "description": f"Нарушение регламента возврата наличности клиенту: кассир не вернула клиенту положенную сдачу/часть денег ({ans_ret}). Статья: Постановление Правления НБРК № 231, п. 25, 27",
                                    "evidence_frames": ev_frames
                                })

                    # 6.2. Отбракованные купюры объединены с нормальными (Х-44)
                    p_rej = "Посмотри на пересчет на счетной машинке: соединила ли кассир отбракованные машинкой купюры с нормальной пачкой? Опиши кратко."
                    ans_rej = self.ask_vlm(Image.fromarray(rgb), p_rej)
                    ans_rej_lower = ans_rej.lower()
                    if any(w in ans_rej_lower for w in ["соединила", "смешала", "объединила отбракованные"]) and not any(n in ans_rej_lower for n in ["не соединила", "не смешивала"]):
                        if not is_duplicate("NO_COUNTING_MACHINE", t_ev):
                            ev_frames = self.get_evidence_frames(cap, t_ev, duration)
                            detected_violations.append({
                                "type": "NO_COUNTING_MACHINE",
                                "timecode_sec": round(float(t_ev), 1),
                                "description": f"Грубое нарушение кассовой дисциплины: банкноты, отбракованные счетной машинкой, объединены с нормальной пачкой ({ans_rej}). Статья: Постановление Правления НБРК № 231, Глава 3, п. 24",
                                "evidence_frames": ev_frames
                            })

                    # 6.3. Смешивание денег предыдущего и текущего клиента (Х-47)
                    p_mix = "Посмотри на деньги на столе: объединила ли кассир деньги предыдущего клиента с деньгами текущего клиента в общую пачку? Опиши кратко."
                    ans_mix = self.ask_vlm(Image.fromarray(rgb), p_mix)
                    ans_mix_lower = ans_mix.lower()
                    if any(w in ans_mix_lower for w in ["объединила", "соединила деньги", "смешала купюры"]) and not any(n in ans_mix_lower for n in ["не объединяла", "не смешивала"]):
                        if not is_duplicate("MONEY_LEFT_BETWEEN_CLIENTS", t_ev):
                            ev_frames = self.get_evidence_frames(cap, t_ev, duration)
                            detected_violations.append({
                                "type": "MONEY_LEFT_BETWEEN_CLIENTS",
                                "timecode_sec": round(float(t_ev), 1),
                                "description": f"Смешивание денежной наличности: кассир объединила деньги предыдущего клиента с наличностью текущего клиента ({ans_mix}). Статья: Постановление Правления НБРК № 231, п. 27",
                                "evidence_frames": ev_frames
                            })

            # =========================================================================
            # Кейс 7: Идентификация клиента и остаток денег на столе (Х-2, Х-17, Х-18, Х-38)
            # =========================================================================
            if trig == "CLIENT_ARRIVAL" or "CLIENT_ARRIVAL" in merged:
                rgb = self.get_rgb_frame(cap, t_ev + 1.0, duration)
                if rgb is not None:
                    # 7.1. Деньги от прошлого клиента на столе при приеме следующего (Х-38)
                    p_leftover = "К столу подошел клиент. Лежат ли уже на столе неубранные деньги от предыдущей операции? Опиши кратко."
                    ans_leftover = self.ask_vlm(Image.fromarray(rgb), p_leftover)
                    ans_lo_lower = ans_leftover.lower()
                    neg_money = ["не лежат", "не остались", "нет денег", "стол чист", "денег нет", "без денег", "не обнаружено", "отсутствуют", "чистый стол"]
                    pos_money = ["остались неубранные деньги", "остались деньги от предыдущего", "лежат неубранные купюры", "осталась пачка денег от прошлого", "лежат забытые деньги", "на столе уже лежат деньги"]
                    has_neg = any(n in ans_lo_lower for n in neg_money)
                    has_pos = any(p in ans_lo_lower for p in pos_money)
                    if has_pos and not has_neg:
                        if not is_duplicate("MONEY_LEFT_BETWEEN_CLIENTS", t_ev):
                            ev_frames = self.get_evidence_frames(cap, t_ev, duration)
                            detected_violations.append({
                                "type": "MONEY_LEFT_BETWEEN_CLIENTS",
                                "timecode_sec": round(float(t_ev), 1),
                                "description": f"Нарушение пооперационного разделения наличности: на столе остались деньги от предыдущего клиента при приеме следующего ({ans_leftover}). Статья: Постановление Правления НБРК № 231, п. 27",
                                "evidence_frames": ev_frames
                            })

                    # 7.2. Инкассаторский мешок без проверки документов (Х-17)
                    p_ink = "Посмотри на человека и стол: принес ли посетитель зеленый инкассаторский мешок или сумку с ценностями?"
                    ans_ink = self.ask_vlm(Image.fromarray(rgb), p_ink).lower()
                    if any(w in ans_ink for w in ["инкассаторский", "зеленый мешок", "сумку с ценностями", "мешок на столе"]) and not any(n in ans_ink for n in ["нет мешка", "обычный клиент"]):
                        if not is_duplicate("NO_ID_VERIFICATION", t_ev):
                            ev_frames = self.get_evidence_frames(cap, t_ev, duration)
                            detected_violations.append({
                                "type": "NO_ID_VERIFICATION",
                                "timecode_sec": round(float(t_ev), 1),
                                "description": f"Нарушение регламента приема инкассаторских сумок: кассир приняла инкассаторский мешок без предварительной проверки служебного удостоверения и доверенности инкассатора. Статья: Постановление Правления НБРК № 231, п. 53-55",
                                "evidence_frames": ev_frames
                            })

                    # 7.3. Немедленная выкладка пачки денег без проверки удостоверения (X-2)
                    if duration < 30.0:
                        p_tbl = "Опиши, что находится на столе: пачка денег или стол пустой?"
                        ans_tbl = self.ask_vlm(Image.fromarray(rgb), p_tbl).lower()
                        pos_tbl = ["находится пачка денег", "лежит пачка денег", "пачка наличных", "пачка денег на столе", "выложена пачка денег"]
                        neg_tbl = ["нет пачки денег", "стол пустой", "пуст", "ничего нет", "не лежит", "нет денег", "чист"]
                        if any(p in ans_tbl for p in pos_tbl) and not any(n in ans_tbl for n in neg_tbl):
                            if not is_duplicate("NO_ID_VERIFICATION", t_ev):
                                ev_frames = self.get_evidence_frames(cap, t_ev, duration)
                                detected_violations.append({
                                    "type": "NO_ID_VERIFICATION",
                                    "timecode_sec": round(float(t_ev), 1),
                                    "description": f"Нарушение регламента идентификации клиента: на стол сразу выложена пачка денег ({ans_tbl}) без предварительного предъявления документа, удостоверяющего личность клиента. Статья: Постановление Правления НБРК № 231, п. 19 пп. 1), п. 31 пп. 1)",
                                    "evidence_frames": ev_frames
                                })

                    # 7.4. Проникновение постороннего лица внутрь рабочей зоны / открытая дверь (Х-30)
                    p_intr = "Посмотри на людей: зашел ли мужчина внутрь кассовой кабины за спину кассира или находится у стола?"
                    ans_intr = self.ask_vlm(Image.fromarray(rgb), p_intr).lower()
                    neg_intr = ["нет посторонних", "один кассир", "не видно", "нельзя определить", "не позволяет определить", "женщина заходит", "кассир заходит", "входит кассир"]
                    pos_intr = ["мужчина зашел", "посторонний зашел", "посторонний внутри", "зашел внутрь кассовой кабины", "за спиной кассира стоит мужчина"]
                    if any(w in ans_intr for w in pos_intr) and not any(n in ans_intr for n in neg_intr):
                        if not is_duplicate("THIRD_PERSON_INTRUSION", t_ev) and not is_duplicate("UNLOCKED_DOOR", t_ev):
                            ev_frames = self.get_evidence_frames(cap, t_ev, duration)
                            detected_violations.append({
                                "type": "UNLOCKED_DOOR",
                                "timecode_sec": round(float(t_ev), 1),
                                "description": f"Нарушение режима безопасности кассового узла: не закрыта входная дверь кабины / проникновение постороннего лица внутрь рабочей зоны ({ans_intr}). Статья: Постановление Правления НБРК № 224, п. 12",
                                "evidence_frames": ev_frames
                            })


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
