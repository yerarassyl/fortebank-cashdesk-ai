import os
import cv2
import torch
import base64
import numpy as np
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
from legal_base import LEGAL_ARTICLES_RK
from vlm_prompts import QUESTIONS, QUESTION_SUFFIX, questions_for, decide
from cv_triggers import PoseEventDetector

DEFAULT_VLM = os.getenv("FORTE_VLM_MODEL", "Qwen/Qwen3-VL-2B-Instruct")
FALLBACK_WINDOW_SEC = 20.0   # сплошные окна, если CV не нашел ни одного триггера
DEDUP_SEC = 10.0             # одно и то же нарушение из соседних окон считается одним


def frame_to_base64(cv2_frame, max_size=720, quality=85):
    h, w = cv2_frame.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        cv2_frame = cv2.resize(cv2_frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    _, buffer = cv2.imencode('.jpg', cv2_frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return base64.b64encode(buffer).decode('utf-8')


class BankViolationDetector:
    def __init__(self, model_name=DEFAULT_VLM):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        kwargs = {"device_map": self.device}
        if self.device == "cuda":
            # 4-bit: модель на 2-4 млрд параметров помещается в видеокарту на 4 ГБ
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
            kwargs["dtype"] = torch.float16  # неквантованные слои тоже в fp16 (без bf16-путей cuBLAS)
        else:
            kwargs["dtype"] = torch.float32

        try:
            self.processor = AutoProcessor.from_pretrained(model_name, local_files_only=True)
            self.model = AutoModelForImageTextToText.from_pretrained(model_name, local_files_only=True, **kwargs)
        except Exception:
            print(f"[Detector] Локальный кэш не найден. Загрузка {model_name} из Hugging Face Hub...")
            self.processor = AutoProcessor.from_pretrained(model_name)
            self.model = AutoModelForImageTextToText.from_pretrained(model_name, **kwargs)
        self.processor.tokenizer.padding_side = "left"
        tok = self.processor.tokenizer
        self.yes_ids = sorted({tok.encode(w, add_special_tokens=False)[0] for w in ("Да", "да", "Yes", "yes")})
        self.no_ids = sorted({tok.encode(w, add_special_tokens=False)[0] for w in ("Нет", "нет", "No", "no")})
        print(f"[Detector] VLM {model_name} загружена ({self.device})")

        self.pose_detector = PoseEventDetector()
        print("[Detector] PoseEventDetector готов к быстрому сканированию видео!")
        self.last_debug = []

    @torch.no_grad()
    def _score_batch(self, img, qids):
        texts = [
            self.processor.apply_chat_template(
                [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": QUESTIONS[q] + QUESTION_SUFFIX}]}],
                tokenize=False, add_generation_prompt=True)
            for q in qids
        ]
        inputs = self.processor(text=texts, images=[img] * len(texts), padding=True, return_tensors="pt").to(self.model.device)
        # logits_to_keep=1: считаем распределение только для последнего токена — экономит видеопамять
        probs = torch.softmax(self.model(**inputs, logits_to_keep=1).logits[:, -1].float(), -1)
        yes, no = probs[:, self.yes_ids].sum(-1), probs[:, self.no_ids].sum(-1)
        return (yes / (yes + no + 1e-9)).tolist()

    def _score(self, img, qids, chunk=6):
        """P(Да) для каждого вопроса по одному кадру; вопросы порциями, при нехватке видеопамяти порция уменьшается."""
        out, i = [], 0
        while i < len(qids):
            part = qids[i:i + chunk]
            try:
                out += self._score_batch(img, part)
                i += len(part)
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                if chunk == 1:
                    raise
                chunk = max(1, chunk // 2)
        return out

    @torch.no_grad()
    def _describe(self, img):
        """Короткое описание кадра-доказательства (что видно), без выводов о нарушении."""
        msgs = [{"role": "user", "content": [{"type": "image", "image": img},
                 {"type": "text", "text": "Одним предложением опиши, что делает кассир и что лежит на столе."}]}]
        inputs = self.processor.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                                    return_dict=True, return_tensors="pt").to(self.model.device)
        out = self.model.generate(**inputs, max_new_tokens=48, do_sample=False)
        return self.processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()

    def analyze_file(self, video_path, filename=None):
        """
        Анализирует видеофайл и возвращает результат в формате раздела 10 ТЗ:
        {"verdict": "no_violation" | "violation", "violations": [{type, timecode_sec, description, evidence_frames}]}
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

        # 1. CV-триггеры (YOLO Pose + temporal tracker), окна покрывают подозрительные участки
        events = self.pose_detector.detect_video_events(video_path)
        roi = getattr(self.pose_detector, "last_roi", None)
        if not events:
            starts = np.arange(0.0, max(duration, 0.1), FALLBACK_WINDOW_SEC)
            events = [{"timecode_sec": float(min(s + FALLBACK_WINDOW_SEC / 2, duration)), "trigger": "INITIAL_DESK_CHECK",
                       "window_start_sec": float(s), "window_end_sec": float(min(s + FALLBACK_WINDOW_SEC, duration)),
                       "merged_triggers": ["INITIAL_DESK_CHECK"]} for s in starts]

        found = []
        self.last_debug = []
        for ev in events:
            center = ev["timecode_sec"]
            start = ev.get("window_start_sec", max(0.0, center - 3.5))
            end = ev.get("window_end_sec", min(duration, center + 3.5))
            triggers = ev.get("merged_triggers") or [ev["trigger"]]

            # 2. Кадры: равномерно по окну + вплотную к моменту триггера (короткие действия)
            t_points = sorted({round(float(t), 1) for t in list(np.linspace(start, end, 5)) + [center - 1, center, center + 1]
                               if 0 <= t < duration - 0.05})
            frames = []
            for t in t_points:
                cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
                ok, frame = cap.read()
                if not ok:
                    continue
                x1, y1, x2, y2 = roi or (0, 0, frame.shape[1], frame.shape[0])
                crop = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
                frames.append((t, frame, Image.fromarray(crop)))
            if not frames:
                continue

            # 3. VLM: базовые вопросы + чек-листы триггеров окна, по каждому кадру
            qids = questions_for(triggers)
            per_frame = [self._score(img, qids) for _, _, img in frames]
            scores = {q: [pf[i] for pf in per_frame] for i, q in enumerate(qids)}
            self.last_debug.append({"triggers": triggers, "times": [t for t, _, _ in frames],
                                    "scores": {q: [round(v, 3) for v in s] for q, s in scores.items()}})

            # 4. Решение по правилам
            for f in decide(scores, triggers):
                t, full_frame, img = frames[f["frame"]]
                found.append({**f, "t": t, "frame_idx": f["frame"], "frames": frames})

        cap.release()

        # 5. Одно нарушение одного типа из соседних окон — одна запись (самая уверенная)
        found.sort(key=lambda f: -f["score"])
        kept = []
        for f in found:
            if not any(k["code"] == f["code"] and abs(k["t"] - f["t"]) < DEDUP_SEC for k in kept):
                kept.append(f)
        kept.sort(key=lambda f: f["t"])

        violations = []
        for f in kept:
            legal = LEGAL_ARTICLES_RK.get(f["code"], LEGAL_ARTICLES_RK["OTHER"])
            frames, i = f["frames"], f["frame_idx"]
            evidence = [{"timestamp_sec": round(float(t), 1), "image_base64": frame_to_base64(fr)}
                        for t, fr, _ in frames[max(0, i - 1): i + 2]]
            seen = self._describe(frames[i][2])
            description = (f"{f['observation']} Описание кадра {f['t']:.1f} с: {seen.rstrip('.')}. "
                           f"Уверенность модели: {f['score']:.2f}. Норматив: {legal['law_source']}.")
            if legal.get("law_quote"):
                description += f" «{legal['law_quote']}»"
            if legal.get("note"):
                description += f" Примечание: {legal['note']}"
            violations.append({
                "type": legal["type"],
                "timecode_sec": round(float(f["t"]), 1),
                "description": description,
                "evidence_frames": evidence,
            })

        return {
            "verdict": "violation" if violations else "no_violation",
            "violations": violations,
        }
