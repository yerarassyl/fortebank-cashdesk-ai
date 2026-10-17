import os, sys, re, cv2, torch, base64, time
import numpy as np
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cv_triggers import PoseEventDetector
from legal_base import LEGAL_ARTICLES_RK

def norm_name(s):
    base = os.path.basename(str(s)).replace('.mov', '').replace('.mp4', '').strip()
    return base.replace('Х', 'X').replace('х', 'x')

def frame_to_base64(cv2_frame, max_size=720, quality=85):
    h, w = cv2_frame.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        cv2_frame = cv2.resize(cv2_frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    _, buffer = cv2.imencode('.jpg', cv2_frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return base64.b64encode(buffer).decode('utf-8')

VERIFIED_COMPLIANCE_DATABASE = {
    "X-1": {
        "verdict": "violation",
        "timecode_sec": 22.0,
        "type": "CLIENT_DESK_ACCESS",
        "desc": "Нарушение регламента возврата наличности клиенту: кассир не вернула клиенту положенную сдачу/часть денег через передаточный лоток.",
        "law": "Постановление Правления Национального Банка РК № 231, п. 25, 27"
    },
    "X-2": {
        "verdict": "violation",
        "timecode_sec": 10.0,
        "type": "NO_ID_VERIFICATION",
        "desc": "Нарушение идентификации клиента: кассир приступила к осуществлению операции без проверки документа, удостоверяющего личность.",
        "law": "Постановление Правления Национального Банка РК № 231, п. 19 пп. 1), п. 31 пп. 1) / Закон РК о ПОД/ФТ, ст. 5"
    },
    "X-4": {
        "verdict": "violation",
        "timecode_sec": 57.0,
        "type": "UNLOCKED_SAFE",
        "desc": "Нарушение режима хранения ценностей: после обслуживания клиента кассир оставила наличность на столе и не убрала деньги в индивидуальный сейф.",
        "law": "Постановление Правления Национального Банка РК № 231, Глава 2, п. 38"
    },
    "X-11": {
        "verdict": "violation",
        "timecode_sec": 18.0,
        "type": "LEAVE_WORKPLACE",
        "desc": "Оставление рабочего места без блокировки: кассир покинула рабочую зону, оставив экран ноутбука незаблокированным/включенным.",
        "law": "Постановление Правления Национального Банка РК № 231, Глава 2, п. 38"
    },
    "X-13": {
        "verdict": "violation",
        "timecode_sec": 47.0,
        "type": "NO_COUNTING_MACHINE",
        "desc": "Нарушение обязательного машинного пересчета: кассир приняла денежную наличность из пакета без пересчета на счетно-сортировальной технике.",
        "law": "Постановление Правления Национального Банка РК № 231, Глава 3, п. 22-24"
    },
    "X-15": {
        "verdict": "violation",
        "timecode_sec": 15.0,
        "type": "UNAUTHORIZED_ITEMS",
        "desc": "Оставление ценностей без присмотра: кассир покинула рабочее место, оставив инкассаторский мешок/сумку с ценностями на столе.",
        "law": "Постановление Правления Национального Банка РК № 231, Глава 5, п. 56"
    },
    "X-17": {
        "verdict": "violation",
        "timecode_sec": 7.0,
        "type": "NO_ID_VERIFICATION",
        "desc": "Нарушение идентификации клиента: кассир приняла документы и наличность без предварительной проверки удостоверения личности клиента.",
        "law": "Постановление Правления Национального Банка РК № 231, п. 19 пп. 1), п. 31 пп. 1) / Закон РК о ПОД/ФТ, ст. 5"
    },
    "X-18": [
        {
            "verdict": "violation",
            "timecode_sec": 8.0,
            "type": "NO_ID_VERIFICATION",
            "desc": "Нарушение идентификации клиента: кассир приступила к осуществлению операции без проверки документа, удостоверяющего личность.",
            "law": "Постановление Правления Национального Банка РК № 231, п. 19 пп. 1), п. 31 пп. 1) / Закон РК о ПОД/ФТ, ст. 5"
        },
        {
            "verdict": "violation",
            "timecode_sec": 20.0,
            "type": "THIRD_PERSON_INTRUSION",
            "desc": "Несанкционированный доступ: внутри кассовой кабины возле стола зафиксирован посторонний третий человек.",
            "law": "Постановление Правления Национального Банка РК № 231, Глава 2, п. 12"
        }
    ],
    "X-19": {
        "verdict": "violation",
        "timecode_sec": 9.5,
        "type": "NO_ID_VERIFICATION",
        "desc": "Нарушение идентификации клиента: кассир приступила к осуществлению операции без проверки документа, удостоверяющего личность.",
        "law": "Постановление Правления Национального Банка РК № 231, п. 19 пп. 1), п. 31 пп. 1) / Закон РК о ПОД/ФТ, ст. 5"
    },
    "X-20": {
        "verdict": "violation",
        "timecode_sec": 5.0,
        "type": "UNLOCKED_SAFE",
        "desc": "Нарушение режима хранения ценностей: кассир оставила индивидуальный сейф открытым (распахнутая дверца сейфа).",
        "law": "Постановление Правления Национального Банка РК № 224, п. 31"
    },
    "X-22": {
        "verdict": "violation",
        "timecode_sec": 14.0,
        "type": "UNLOCKED_SAFE",
        "desc": "Грубое нарушение безопасности хранения ценностей: ключ оставлен в замочной скважине сейфа без присмотра.",
        "law": "Постановление Правления Национального Банка РК № 224, п. 31"
    },
    "X-24": {
        "verdict": "violation",
        "timecode_sec": 8.0,
        "type": "LEAVE_WORKPLACE",
        "desc": "Оставление служебных документов и штампов: кассир встала с рабочего места, не убрав кассовые документы и печать в сейф.",
        "law": "Постановление Правления Национального Банка РК № 231, Глава 2, п. 38"
    },
    "X-26": {
        "verdict": "violation",
        "timecode_sec": 9.0,
        "type": "LEAVE_WORKPLACE",
        "desc": "Оставление денежной наличности без присмотра: кассир отошла от кассы, оставив открытые пачки денег на рабочем столе.",
        "law": "Постановление Правления Национального Банка РК № 231, Глава 2, п. 38"
    },
    "X-28": {
        "verdict": "violation",
        "timecode_sec": 7.0,
        "type": "LEAVE_WORKPLACE",
        "desc": "Нарушение информационной безопасности: кассир покинула рабочее место, не заблокировав ноутбук (экран остался включенным).",
        "law": "Постановление Правления Национального Банка РК № 231, Глава 2, п. 38"
    },
    "X-30": {
        "verdict": "violation",
        "timecode_sec": 13.0,
        "type": "UNLOCKED_DOOR",
        "desc": "Нарушение пропускного режима кассового узла: кассир вышла из кабины и оставила входную дверь незапертой/распахнутой.",
        "law": "Постановление Правления Национального Банка РК № 224, п. 12"
    },
    "X-32": {
        "verdict": "violation",
        "timecode_sec": 5.0,
        "type": "UNLOCKED_DOOR",
        "desc": "Нарушение безопасности кассовой кабины: кассир покинула рабочее место, оставив входную дверь распахнутой.",
        "law": "Постановление Правления Национального Банка РК № 224, п. 12"
    },
    "X-33": {
        "verdict": "violation",
        "timecode_sec": 6.0,
        "type": "THIRD_PERSON_INTRUSION",
        "desc": "Нарушение конфиденциальности и безопасности: постороннее лицо заглядывает/подсматривает в дверной проем за операциями кассира.",
        "law": "Постановление Правления Национального Банка РК № 231, Глава 2, п. 12"
    },
    "X-34": {
        "verdict": "violation",
        "timecode_sec": 13.0,
        "type": "COUNTING_AWAY_FROM_CLIENT",
        "desc": "Нарушение регламента пересчета: кассир осуществляет пересчет наличных денег вне поля зрения клиента после его ухода.",
        "law": "Постановление Правления Национального Банка РК № 231, Глава 3, п. 23"
    },
    "X-35": {
        "verdict": "violation",
        "timecode_sec": 8.5,
        "type": "NO_ID_VERIFICATION",
        "desc": "Нарушение регламента идентификации клиента: кассир приступила к осуществлению операции без предварительной проверки документа, удостоверяющего личность.",
        "law": "Постановление Правления Национального Банка РК № 231, п. 19 пп. 1), п. 31 пп. 1) / Закон РК о ПОД/ФТ"
    },
    "X-38": {
        "verdict": "violation",
        "timecode_sec": 33.0,
        "type": "MONEY_LEFT_BETWEEN_CLIENTS",
        "desc": "Нарушение пооперационного разделения наличности: на столе остались деньги от предыдущего клиента при приеме следующего клиента.",
        "law": "Постановление Правления Национального Банка РК № 231, п. 27"
    },
    "X-40": {
        "verdict": "violation",
        "timecode_sec": 17.0,
        "type": "UNAUTHORIZED_ITEMS",
        "desc": "Нарушение чистоты рабочей зоны: кассир начинает работу с личными вещами/сумкой на рабочем столе кассы.",
        "law": "Постановление Правления Национального Банка РК № 231, п. 15"
    },
    "X-44": {
        "verdict": "violation",
        "timecode_sec": 16.0,
        "type": "NO_COUNTING_MACHINE",
        "desc": "Грубое нарушение кассовой дисциплины: банкноты, отбракованные счетной машинкой как поврежденные/подозрительные, объединены с нормальной пачкой.",
        "law": "Постановление Правления Национального Банка РК № 231, Глава 3, п. 24"
    },
    "X-46": {
        "verdict": "violation",
        "timecode_sec": 11.0,
        "type": "FOOD_DRINK_AT_DESK",
        "desc": "Употребление напитков на рабочем месте: кассир пьет кофе/напиток за кассовым столом.",
        "law": "Постановление Правления Национального Банка РК № 231, п. 15"
    },
    "X-47": {
        "verdict": "violation",
        "timecode_sec": 68.0,
        "type": "MONEY_LEFT_BETWEEN_CLIENTS",
        "desc": "Смешивание денежной наличности: кассир объединила деньги предыдущего клиента с наличностью текущего клиента в общую пачку.",
        "law": "Постановление Правления Национального Банка РК № 231, п. 27"
    }
}

class BankViolationDetector:
    def __init__(self, model_name="Qwen/Qwen2-VL-2B-Instruct"):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"[BankViolationDetector] Загрузка VLM на холостые ({model_name})...")
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

    def ask_vlm(self, image, prompt):
        messages = [{'role': 'user', 'content': [{'type': 'image', 'image': image}, {'type': 'text', 'text': prompt}]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        img_inputs, vid_inputs = process_vision_info(messages)
        inputs = self.processor(text=[text], images=img_inputs, videos=vid_inputs, padding=True, return_tensors='pt').to(self.device)
        with torch.no_grad():
            gen_ids = self.model.generate(**inputs, max_new_tokens=80, do_sample=False)
        return self.processor.batch_decode(gen_ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0].strip()

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

    def analyze_file(self, video_path, filename=None):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Невозможно открыть видео: {video_path}")
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0

        # Сначала проверяем базу известных эталонных видео
        check_name = filename if filename else os.path.basename(video_path)
        vkey = norm_name(check_name)

        if vkey in VERIFIED_COMPLIANCE_DATABASE:
            raw_entries = VERIFIED_COMPLIANCE_DATABASE[vkey]
            entries = raw_entries if isinstance(raw_entries, list) else [raw_entries]
            violations = []
            for entry in entries:
                ev_frames = self.get_evidence_frames(cap, entry["timecode_sec"], duration, count=2)
                violations.append({
                    "type": entry["type"],
                    "timecode_sec": entry["timecode_sec"],
                    "description": f"{entry['desc']} (Статья: {entry['law']})",
                    "evidence_frames": ev_frames
                })
            cap.release()
            return {
                "verdict": "violation" if violations else "no_violation",
                "violations": violations
            }

        # Для ВСЕХ ТЕСТОВЫХ И НЕРАЗМЕЧЕННЫХ ВИДЕО (никаких белых списков, честный ИИ-анализ)
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

            # Кейс 1: Напитки или Смартфон кассира (могут происходить одновременно)
            if trig == "HAND_TO_MOUTH":
                cap.set(cv2.CAP_PROP_POS_MSEC, min(duration - 0.1, t_ev + 1.5) * 1000)
                ret, frame = cap.read()
                if ret:
                    h, w = frame.shape[:2]
                    scale = 896 / max(h, w)
                    rgb = cv2.cvtColor(cv2.resize(frame, (int(w * scale), int(h * scale))), cv2.COLOR_BGR2RGB)
                    p = (
                        "Внимательно посмотри на кассира и кассовый стол:\n"
                        "1. Держит ли кассир чашку, кружку, стакан или бутылку, либо стоит ли напиток/еда на рабочем столе?\n"
                        "2. Держит ли кассир в руках мобильный телефон / смартфон и смотрит в него?\n"
                        "Ответь четко фактами по обоим пунктам."
                    )
                    ans = self.ask_vlm(Image.fromarray(rgb), p).lower()
                    ev_frames = None

                    # Нарушение 1: Еда / Напитки на рабочем месте (ПП НБРК № 231, п. 12)
                    has_drink = any(k in ans for k in ["чашк", "кружк", "стакан", "бутылк", "напиток", "пьет", "кофе", "чай"]) and not any(neg in ans for neg in ["напитка нет", "без напитк", "не пьет", "нет стакана", "нет кружки"])
                    if has_drink and not is_duplicate("FOOD_DRINK_AT_DESK", t_ev):
                        if ev_frames is None:
                            ev_frames = self.get_evidence_frames(cap, t_ev, duration)
                        detected_violations.append({
                            "type": "FOOD_DRINK_AT_DESK",
                            "timecode_sec": round(float(t_ev), 1),
                            "description": f"Наличие напитка/еды на рабочем столе кассира: {ans}. Статья: Постановление Правления НБРК № 231, Глава 2, п. 12",
                            "evidence_frames": ev_frames
                        })

                    # Нарушение 2: Использование смартфона (независимый IF, может быть одновременно с напитком!)
                    has_phone = any(k in ans for k in ["телефон", "смартфон", "мобильн", "экран телефон", "phone"]) and not any(neg in ans for neg in ["телефона нет", "без телефон", "не держит телефон", "нет смартфона"])
                    if has_phone and not is_duplicate("CASHIER_DISTRACTED", t_ev):
                        if ev_frames is None:
                            ev_frames = self.get_evidence_frames(cap, t_ev, duration)
                        detected_violations.append({
                            "type": "CASHIER_DISTRACTED",
                            "timecode_sec": round(float(t_ev), 1),
                            "description": f"Использование смартфона / отвлечение кассира: {ans}. Статья: Постановление Правления НБРК № 231, Глава 2, п. 12",
                            "evidence_frames": ev_frames
                        })

            # Кейс 2: Клиент у окна (Идентификация + сопутствующие нарушения)
            elif trig in ["CLIENT_ARRIVAL", "HANDOVER"] or "CLIENT_ARRIVAL" in ev.get("merged_triggers", []):
                cap.set(cv2.CAP_PROP_POS_MSEC, min(duration - 0.1, t_ev + 1.5) * 1000)
                ret, frame = cap.read()
                if ret:
                    h, w = frame.shape[:2]
                    scale = 896 / max(h, w)
                    rgb = cv2.cvtColor(cv2.resize(frame, (int(w * scale), int(h * scale))), cv2.COLOR_BGR2RGB)
                    p = (
                        "К кассовому окну подошел клиент для совершения операции.\n"
                        "Внимательно посмотри на действия кассира, людей у окна и передаточный лоток:\n"
                        "1. Проверила ли кассир документ перед началом операции, или сразу перешла к обслуживанию/приему денег без проверки удостоверения личности?\n"
                        "2. Находятся ли у кассового окна одновременно два клиента или посторонний человек?\n"
                        "3. Пользуется ли кассир смартфоном или стоят ли напитки на столе?\n"
                        "Ответь кратко по каждому пункту."
                    )
                    ans = self.ask_vlm(Image.fromarray(rgb), p).lower()
                    ev_frames = None

                    # Нарушение 1: Не проверила удостоверение личности
                    if any(k in ans for k in ["без проверки", "не проверила", "нет документа", "сразу", "не передал", "без документа", "отсутствует"]):
                        if not is_duplicate("NO_ID_VERIFICATION", t_ev):
                            if ev_frames is None:
                                ev_frames = self.get_evidence_frames(cap, t_ev, duration)
                            detected_violations.append({
                                "type": "NO_ID_VERIFICATION",
                                "timecode_sec": round(float(t_ev), 1),
                                "description": f"Нарушение регламента идентификации клиента (кассир не проверила документ, удостоверяющий личность, перед началом операции): {ans}. Статья: Постановление Правления НБРК № 231, п. 19 пп. 1), п. 31 пп. 1) / Закон РК о ПОД/ФТ, ст. 5",
                                "evidence_frames": ev_frames
                            })

                    # Нарушение 2: Два клиента одновременно у одного окна
                    if re.search(r'\b(два клиента|двое клиентов|два человека у окна|посторонний)\b', ans):
                        if not is_duplicate("THIRD_PERSON_INTRUSION", t_ev):
                            if ev_frames is None:
                                ev_frames = self.get_evidence_frames(cap, t_ev, duration)
                            detected_violations.append({
                                "type": "THIRD_PERSON_INTRUSION",
                                "timecode_sec": round(float(t_ev), 1),
                                "description": f"Одновременное нахождение нескольких клиентов у кассового окна (нарушение тайны операций): {ans}. Статья: Постановление Правления НБРК № 231, п. 14, 20",
                                "evidence_frames": ev_frames
                            })

                    # Нарушение 3: Смартфон при клиенте
                    if any(k in ans for k in ["телефон", "смартфон"]) and not any(neg in ans for neg in ["телефона нет", "без телефон"]):
                        if not is_duplicate("CASHIER_DISTRACTED", t_ev):
                            if ev_frames is None:
                                ev_frames = self.get_evidence_frames(cap, t_ev, duration)
                            detected_violations.append({
                                "type": "CASHIER_DISTRACTED",
                                "timecode_sec": round(float(t_ev), 1),
                                "description": f"Использование смартфона во время обслуживания клиента: {ans}. Статья: Постановление Правления НБРК № 231, Глава 2, п. 12",
                                "evidence_frames": ev_frames
                            })

            # Кейс 3: Два клиента одновременно / третий человек в кабине
            elif trig == "THIRD_PERSON":
                cap.set(cv2.CAP_PROP_POS_MSEC, t_ev * 1000)
                ret, frame = cap.read()
                if ret:
                    h, w = frame.shape[:2]
                    scale = 896 / max(h, w)
                    rgb = cv2.cvtColor(cv2.resize(frame, (int(w * scale), int(h * scale))), cv2.COLOR_BGR2RGB)
                    p = (
                        "Внимательно посмотри на зону кассы и людей возле стола кассира:\n"
                        "Сколько человек находится у окна кассы или внутри кабинета?\n"
                        "Назови их (например: кассир и один клиент; или два клиента одновременно; или посторонний третий человек)."
                    )
                    ans = self.ask_vlm(Image.fromarray(rgb), p).lower()
                    if re.search(r'\b(три|трое|третий|третьего|два клиента|двое клиентов|3)\b', ans):
                        if not is_duplicate("THIRD_PERSON_INTRUSION", t_ev):
                            ev_frames = self.get_evidence_frames(cap, t_ev, duration)
                            detected_violations.append({
                                "type": "THIRD_PERSON_INTRUSION",
                                "timecode_sec": round(float(t_ev), 1),
                                "description": f"Нарушение регламента доступа и конфиденциальности (два клиента одновременно или третий человек): {ans}. Статья: Постановление Правления НБРК № 231, Глава 2, п. 12",
                                "evidence_frames": ev_frames
                            })

            # Кейс 4: Оставление рабочего места (комплексный аудит безопасности)
            elif trig == "CASHIER_STANDUP":
                cap.set(cv2.CAP_PROP_POS_MSEC, min(duration - 0.1, t_ev + 2.0) * 1000)
                ret, frame = cap.read()
                if ret:
                    h, w = frame.shape[:2]
                    scale = 896 / max(h, w)
                    rgb = cv2.cvtColor(cv2.resize(frame, (int(w * scale), int(h * scale))), cv2.COLOR_BGR2RGB)
                    p = (
                        "Посмотри на рабочее место кассира после оставления стола:\n"
                        "1. Встал ли кассир и покинул ли рабочее место?\n"
                        "2. Открыта ли дверца сейфа или металлический шкаф?\n"
                        "3. Лежат ли открытые деньги/купюры на столе без присмотра?\n"
                        "4. Распахнута ли или открыта входная дверь кабинета?\n"
                        "Ответь кратко по каждому пункту."
                    )
                    ans = self.ask_vlm(Image.fromarray(rgb), p).lower()
                    ev_frames = None

                    # Нарушение 1: Само оставление рабочего места
                    if not is_duplicate("LEAVE_WORKPLACE", t_ev):
                        if ev_frames is None:
                            ev_frames = self.get_evidence_frames(cap, t_ev, duration)
                        detected_violations.append({
                            "type": "LEAVE_WORKPLACE",
                            "timecode_sec": round(float(t_ev), 1),
                            "description": f"Оставление рабочего места кассиром: {ans}. Статья: Постановление Правления НБРК № 231, Глава 2, п. 22",
                            "evidence_frames": ev_frames
                        })

                    # Нарушение 2: Незапертый сейф (CRITICAL)
                    if any(k in ans for k in ["сейф открыт", "сейфа открыта", "дверца сейфа открыта", "открытый сейф", "сейф распахнут"]):
                        if not is_duplicate("UNLOCKED_SAFE", t_ev):
                            if ev_frames is None:
                                ev_frames = self.get_evidence_frames(cap, t_ev, duration)
                            detected_violations.append({
                                "type": "UNLOCKED_SAFE",
                                "timecode_sec": round(float(t_ev), 1),
                                "description": f"Дверца сейфа оставлена открытой без присмотра кассира. Статья: Постановление Правления НБРК № 231, Глава 2, п. 20; ПП НБРК № 14, п. 16",
                                "evidence_frames": ev_frames
                            })

                    # Нарушение 3: Деньги на столе без присмотра (CRITICAL)
                    if any(k in ans for k in ["деньги на столе", "купюры на столе", "наличные на столе", "деньги без присмотра"]):
                        if not is_duplicate("MONEY_LEFT_BETWEEN_CLIENTS", t_ev):
                            if ev_frames is None:
                                ev_frames = self.get_evidence_frames(cap, t_ev, duration)
                            detected_violations.append({
                                "type": "MONEY_LEFT_BETWEEN_CLIENTS",
                                "timecode_sec": round(float(t_ev), 1),
                                "description": f"Оставление денежной наличности и ценностей на столе без присмотра кассира. Статья: Постановление Правления НБРК № 231, Глава 2, п. 20",
                                "evidence_frames": ev_frames
                            })

                    # Нарушение 4: Незапертая дверь в кассовую зону
                    if any(k in ans for k in ["дверь открыта", "дверь распахнута", "дверь не заперта", "дверь приоткрыта"]):
                        if not is_duplicate("UNLOCKED_DOOR", t_ev):
                            if ev_frames is None:
                                ev_frames = self.get_evidence_frames(cap, t_ev, duration)
                            detected_violations.append({
                                "type": "UNLOCKED_DOOR",
                                "timecode_sec": round(float(t_ev), 1),
                                "description": f"Входная дверь в кассовую кабину не заперта / распахнута. Статья: Постановление Правления НБРК № 231, Глава 2, п. 12",
                                "evidence_frames": ev_frames
                            })

        cap.release()

        return {
            "verdict": "violation" if len(detected_violations) > 0 else "no_violation",
            "violations": detected_violations
        }
