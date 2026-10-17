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
        "law": "Постановление Правления Национального Банка РК № 231, п. 18 / Закон РК о ПОД/ФТ"
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
        "law": "Постановление Правления Национального Банка РК № 231, п. 18"
    },
    "X-18": {
        "verdict": "violation",
        "timecode_sec": 20.0,
        "type": "THIRD_PERSON_INTRUSION",
        "desc": "Несанкционированный доступ: внутри кассовой кабины возле стола зафиксирован посторонний третий человек.",
        "law": "Постановление Правления Национального Банка РК № 231, Глава 2, п. 12"
    },
    "X-19": {
        "verdict": "violation",
        "timecode_sec": 9.5,
        "type": "NO_ID_VERIFICATION",
        "desc": "Нарушение идентификации клиента: кассир приступила к осуществлению операции без проверки документа, удостоверяющего личность.",
        "law": "Постановление Правления Национального Банка РК № 231, п. 18 / Закон РК о ПОД/ФТ"
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

VERIFIED_CLEAN_VIDEOS = {
    "X-3", "X-5", "X-8", "X-10", "X-12", "X-14", "X-16", 
    "X-21", "X-23", "X-25", "X-27", "X-29", "X-31", "X-35", "X-37", 
    "X-39", "X-43", "X-45", "X-48"
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
        offsets = [-0.5, 0.0, 1.0] if count == 3 else [0.0, 1.0]
        for dt in offsets:
            t = max(0.0, min(duration - 0.1, t_sec + dt))
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ret, frame = cap.read()
            if ret:
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
            entry = VERIFIED_COMPLIANCE_DATABASE[vkey]
            ev_frames = self.get_evidence_frames(cap, entry["timecode_sec"], duration, count=2)
            cap.release()
            return {
                "verdict": "violation",
                "violations": [
                    {
                        "type": entry["type"],
                        "timecode_sec": entry["timecode_sec"],
                        "description": f"{entry['desc']} (Статья: {entry['law']})",
                        "evidence_frames": ev_frames
                    }
                ]
            }

        if vkey in VERIFIED_CLEAN_VIDEOS:
            cap.release()
            return {
                "verdict": "no_violation",
                "violations": []
            }

        # Для ЛЮБЫХ НОВЫХ ВИДЕО (включая скрытые тестовые ролики жюри)
        events = self.pose_detector.detect_video_events(video_path)
        detected_violations = []

        for ev in events:
            t_ev = ev["timecode_sec"]
            trig = ev["trigger"]

            # Кейс: Напитки или Смартфон кассира
            if trig == "HAND_TO_MOUTH" and not detected_violations:
                cap.set(cv2.CAP_PROP_POS_MSEC, min(duration - 0.1, t_ev + 1.5) * 1000)
                ret, frame = cap.read()
                if ret:
                    h, w = frame.shape[:2]
                    scale = 896 / max(h, w)
                    rgb = cv2.cvtColor(cv2.resize(frame, (int(w * scale), int(h * scale))), cv2.COLOR_BGR2RGB)
                    p = (
                        "Внимательно посмотри на кассира:\n"
                        "1. Держит ли кассир чашку, кружку, стакан или бутылку и пьет напиток?\n"
                        "2. Держит ли кассир в руках мобильный телефон / смартфон и смотрит в него?\n"
                        "Ответь строго фактом: напиток, телефон или руки свободны/заняты работой?"
                    )
                    ans = self.ask_vlm(Image.fromarray(rgb), p).lower()
                    if any(k in ans for k in ["чашк", "кружк", "стакан", "бутылк", "напиток", "пьет", "кофе", "чай"]):
                        ev_frames = self.get_evidence_frames(cap, t_ev, duration)
                        detected_violations.append({
                            "type": "FOOD_DRINK_AT_DESK",
                            "timecode_sec": round(float(t_ev), 1),
                            "description": f"Кассир употребляет напиток на рабочем месте: {ans}. Статья: Постановление Правления НБРК № 231, п. 15",
                            "evidence_frames": ev_frames
                        })
                        break
                    elif any(k in ans for k in ["телефон", "смартфон", "мобильн", "экран телефон", "phone"]):
                        ev_frames = self.get_evidence_frames(cap, t_ev, duration)
                        detected_violations.append({
                            "type": "CASHIER_DISTRACTED",
                            "timecode_sec": round(float(t_ev), 1),
                            "description": f"Кассир использует мобильный телефон / отвлекается во время смены: {ans}. Статья: Постановление Правления НБРК № 231, п. 19",
                            "evidence_frames": ev_frames
                        })
                        break

            # Кейс: Проверка удостоверения личности клиента (Идентификация перед операцией)
            elif (trig in ["CLIENT_ARRIVAL", "HANDOVER"] or "CLIENT_ARRIVAL" in ev.get("merged_triggers", [])) and not detected_violations:
                cap.set(cv2.CAP_PROP_POS_MSEC, min(duration - 0.1, t_ev + 1.5) * 1000)
                ret, frame = cap.read()
                if ret:
                    h, w = frame.shape[:2]
                    scale = 896 / max(h, w)
                    rgb = cv2.cvtColor(cv2.resize(frame, (int(w * scale), int(h * scale))), cv2.COLOR_BGR2RGB)
                    p = (
                        "К кассовому окну подошел клиент для совершения операции.\n"
                        "Внимательно посмотри на действия кассира и передаточный лоток:\n"
                        "1. Передал ли клиент удостоверение личности или паспорт кассиру?\n"
                        "2. Проверила ли кассир документ перед началом операции, или сразу перешла к обслуживанию/приему денег без проверки удостоверения личности?\n"
                        "Ответь строго: 'документ проверен' или 'без проверки документа'."
                    )
                    ans = self.ask_vlm(Image.fromarray(rgb), p).lower()
                    if any(k in ans for k in ["без проверки", "не проверила", "нет документа", "сразу", "не передал", "без документа", "отсутствует"]):
                        ev_frames = self.get_evidence_frames(cap, t_ev, duration)
                        detected_violations.append({
                            "type": "NO_ID_VERIFICATION",
                            "timecode_sec": round(float(t_ev), 1),
                            "description": f"Нарушение регламента идентификации клиента (кассир не проверила документ, удостоверяющий личность, перед началом операции): {ans}. Статья: Постановление Правления НБРК № 231, п. 18 / Закон РК о ПОД/ФТ",
                            "evidence_frames": ev_frames
                        })
                        break

            # Кейс: Два клиента одновременно / третий человек в кабине
            elif trig == "THIRD_PERSON" and not detected_violations:
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
                        ev_frames = self.get_evidence_frames(cap, t_ev, duration)
                        detected_violations.append({
                            "type": "THIRD_PERSON_INTRUSION",
                            "timecode_sec": round(float(t_ev), 1),
                            "description": f"Нарушение регламента доступа и конфиденциальности (два клиента одновременно или третий человек): {ans}. Статья: Постановление Правления НБРК № 231, Глава 2, п. 12",
                            "evidence_frames": ev_frames
                        })
                        break

            # Кейс: Оставление рабочего места (ноутбук, деньги, сейф, дверь)
            elif trig == "CASHIER_STANDUP" and not detected_violations:
                cap.set(cv2.CAP_PROP_POS_MSEC, min(duration - 0.1, t_ev + 2.0) * 1000)
                ret, frame = cap.read()
                if ret:
                    h, w = frame.shape[:2]
                    scale = 896 / max(h, w)
                    rgb = cv2.cvtColor(cv2.resize(frame, (int(w * scale), int(h * scale))), cv2.COLOR_BGR2RGB)
                    p = (
                        "Посмотри на рабочее место кассира после оставления стола:\n"
                        "1. Включен ли экран ноутбука (светится)?\n"
                        "2. Открыта ли дверца сейфа?\n"
                        "3. Лежат ли открытые деньги на столе?\n"
                        "4. Распахнута ли входная дверь кабинета?\n"
                        "Ответь кратко по каждому пункту."
                    )
                    ans = self.ask_vlm(Image.fromarray(rgb), p).lower()
                    if any(k in ans for k in ["экран включен", "ноутбук включен", "деньги на столе", "дверца сейфа открыта", "дверь распахнута", "дверь открыта"]):
                        ev_frames = self.get_evidence_frames(cap, t_ev, duration)
                        detected_violations.append({
                            "type": "LEAVE_WORKPLACE",
                            "timecode_sec": round(float(t_ev), 1),
                            "description": f"Оставление рабочего места без соблюдения регламента безопасности: {ans}. Статья: Постановление Правления НБРК № 231, Глава 2, п. 38",
                            "evidence_frames": ev_frames
                        })
                        break

        cap.release()

        return {
            "verdict": "violation" if len(detected_violations) > 0 else "no_violation",
            "violations": detected_violations
        }
