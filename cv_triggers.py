"""
cv_triggers.py
Модуль извлечения макро-событий кассовой зоны с помощью YOLO Pose.
Использует ЧИСТУЮ ВРЕМЕННУЮ СТАБИЛЬНОСТЬ (Temporal Stability Tracker):
- Фильтрация отражений в стеклянной двери без хардкода пикселей и координат.
  (Отражения и тени мерцают и исчезают < 1.3с; реальные люди стабильно отслеживаются >= 1.3с).
- Динамическое определение кассира по стартовой позиции (без статичных полигонов зон).
- Определение жестов (рука у лица, сближение рук при передаче денег) через нормализованную биометрию скелета.
"""

import cv2
import numpy as np
import os
import torch
from ultralytics import YOLO

class TemporalTrack:
    def __init__(self, track_id, box, keypoints, sec):
        self.track_id = track_id
        self.box = box
        self.keypoints = keypoints
        self.start_sec = sec
        self.last_sec = sec
        self.hits = 1
        self.misses = 0
        self.confirmed = False
        self.center = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
        self.initial_center = self.center
        self.is_cashier = False

def select_covering_events(events, duration, radius_sec=10.0):
    """
    Минимизация числа триггеров без потери покрытия таймлайна.
    Каждый триггер "отвечает" за окно [t - R, t + R]. Оставляем минимальное подмножество
    триггеров, окна которых покрывают ровно ту же область видео, что и окна всех триггеров
    (жадный алгоритм покрытия интервалами одинаковой длины — оптимален).
    Любой момент нарушения в пределах R от исходного триггера остается в пределах R от
    выбранного, поэтому recall при допуске R не падает.
    Подавленные триггеры не теряются: их тип и prompt_focus сливаются в ближайший выбранный.
    """
    if not events:
        return []
    evs = sorted(events, key=lambda e: e["timecode_sec"])
    kept = []
    i, n = 0, len(evs)
    while i < n:
        covered = max(0.0, evs[i]["timecode_sec"] - radius_sec)
        while i < n and evs[i]["timecode_sec"] - radius_sec <= covered:
            best = None
            while i < n and evs[i]["timecode_sec"] - radius_sec <= covered:
                best = evs[i]
                i += 1
            kept.append(dict(best))
            covered = best["timecode_sec"] + radius_sec
            if duration and covered >= duration:
                i = n

    for k in kept:
        k["merged_triggers"] = []
        k["window_start_sec"] = round(max(0.0, k["timecode_sec"] - radius_sec), 1)
        k["window_end_sec"] = round(min(duration, k["timecode_sec"] + radius_sec) if duration else k["timecode_sec"] + radius_sec, 1)
    for ev in evs:
        owner = min(kept, key=lambda k: abs(k["timecode_sec"] - ev["timecode_sec"]))
        owner["merged_triggers"].append(ev["trigger"])
        if ev.get("prompt_focus") and ev["prompt_focus"] not in owner["prompt_focus"]:
            owner["prompt_focus"] += f"\n[{ev['timecode_sec']}с, {ev['trigger']}] " + ev["prompt_focus"]
    return kept

class PoseEventDetector:
    def __init__(self, model_path=None):
        if model_path is None:
            candidates = ["yolo11m-pose.pt", r"e:\sentra\yolo11m-pose.pt", "yolov8n-pose.pt", r"e:\sentra\yolov8n-pose.pt"]
            for c in candidates:
                if os.path.exists(c):
                    model_path = c
                    break
            if model_path is None:
                model_path = "yolo11m-pose.pt"
        print(f"[PoseEventDetector] Загрузка YOLO Pose: {model_path}...")
        self.model = YOLO(model_path)
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    def detect_video_events(self, video_path, step_frames=5, batch_size=16, min_stable_sec=1.3, max_lost_sec=1.0, cover_radius_sec=10.0):
        """
        Быстрое батч-сканирование видео через YOLO Pose на GPU.
        Фильтрация шумов и отражений строго через Temporal Stability Tracker (без хардкода пикселей).
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Невозможно открыть видео: {video_path}")
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # 1-2. Потоковое чтение кадров и батч-инференс YOLO Pose: в памяти не больше batch_size кадров
        # (раньше все кадры видео держались в RAM — ~1 ГБ на 2.5-минутном ролике)
        sampled_secs = []
        all_boxes = []
        all_kpts = []
        batch = []

        def flush():
            for p in self.model.predict(batch, conf=0.35, device=self.device, verbose=False):
                all_boxes.append(p.boxes.xyxy.cpu().numpy() if p.boxes is not None else [])
                all_kpts.append(p.keypoints.data.cpu().numpy() if p.keypoints is not None else [])
            batch.clear()

        f_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if f_idx % step_frames == 0:
                batch.append(frame)
                sampled_secs.append(f_idx / fps)
                if len(batch) == batch_size:
                    flush()
            f_idx += 1
        cap.release()
        if batch:
            flush()

        if not sampled_secs:
            return []

        fps_sample = fps / step_frames if step_frames > 0 else 6.0
        min_stable_hits = max(3, int(min_stable_sec * fps_sample))
        max_miss_frames = max(2, int(max_lost_sec * fps_sample))

        # Рабочая зона для VLM: по вертикали объединение рамок людей по всему видео + запас 25%
        # (без хардкода координат — вычисляется по каждому видео, подходит для другого ракурса)
        ys = [(b[1], b[3]) for frame_boxes in all_boxes for b in frame_boxes]
        if ys:
            y1, y2 = min(y for y, _ in ys), max(y for _, y in ys)
            pad = 0.25 * (y2 - y1)
            self.last_roi = (0, int(max(0, y1 - pad)), width, int(min(height, y2 + pad)))
        else:
            self.last_roi = (0, 0, width, height)

        # 3. Временной трекер с подавлением отражений
        tracks = {}
        next_tid = 0
        raw_events = []
        prev_confirmed_count = None
        cashier_id = None

        # Состояния гистерезиса (State Transition Flags)
        in_handover = False
        handover_hold_count = 0
        in_hand_to_mouth = False
        mouth_hold_count = 0
        cashier_is_standing = False

        # Аудит начала смены / чистоты стола в первые секунды
        raw_events.append({
            "timecode_sec": 0.5,
            "trigger": "INITIAL_DESK_CHECK",
            "context": "alone",
            "prompt_focus": (
                "Начало смены / кассир на рабочем месте. Внимательно проверь:\n"
                "1. Закрыта ли дверца сейфа, заперт ли замок?\n"
                "2. Нет ли на столе посторонних предметов, открытых пачек денег или оставленных документов?\n"
                "3. Закрыта ли дверь кассового узла?"
            )
        })

        for frame_i, (sec, boxes, kpts) in enumerate(zip(sampled_secs, all_boxes, all_kpts)):
            matched_tids = set()

            for b_idx, b in enumerate(boxes):
                cx = (b[0] + b[2]) / 2
                cy = (b[1] + b[3]) / 2
                box_diag = np.hypot(b[2] - b[0], b[3] - b[1])
                match_threshold = max(80, box_diag * 0.7)

                best_tid = None
                best_dist = 9999
                for tid, tr in tracks.items():
                    if tid in matched_tids:
                        continue
                    dist = np.hypot(cx - tr.center[0], cy - tr.center[1])
                    if dist < match_threshold and dist < best_dist:
                        best_dist = dist
                        best_tid = tid

                kp = kpts[b_idx] if b_idx < len(kpts) else None

                if best_tid is not None:
                    matched_tids.add(best_tid)
                    tr = tracks[best_tid]
                    tr.box = b
                    tr.keypoints = kp
                    tr.center = (cx, cy)
                    tr.last_sec = sec
                    tr.hits += 1
                    tr.misses = 0
                    if tr.hits >= min_stable_hits and not tr.confirmed:
                        tr.confirmed = True
                else:
                    new_tr = TemporalTrack(next_tid, b, kp, sec)
                    tracks[next_tid] = new_tr
                    matched_tids.add(next_tid)
                    next_tid += 1

            # Потери треков
            lost_tids = []
            for tid, tr in tracks.items():
                if tid not in matched_tids:
                    tr.misses += 1
                    if tr.misses > max_miss_frames:
                        lost_tids.append(tid)

            for tid in lost_tids:
                if tracks[tid].confirmed and tracks[tid].is_cashier:
                    raw_events.append({
                        "timecode_sec": round(sec, 1),
                        "trigger": "CASHIER_STANDUP",
                        "context": "alone",
                        "prompt_focus": (
                            "Кассир поднялась и покинула зону кассы. Внимательно проверь:\n"
                            "1. Выключен/заблокирован ли экран ноутбука кассира?\n"
                            "2. Заперт ли сейф или дверца осталась открытой / торчит ли ключ в замке?\n"
                            "3. Убраны ли ценности, деньги и документы со стола?\n"
                            "4. Закрыта ли за собой дверь кабинета?"
                        )
                    })
                del tracks[tid]

            # Динамическая идентификация кассира (первый стабильный трек в начале видео)
            if cashier_id is None and len(tracks) > 0:
                confirmed_candidates = [t for t in tracks.values() if t.confirmed]
                if confirmed_candidates:
                    cashier_tr = max(confirmed_candidates, key=lambda t: t.hits)
                    cashier_tr.is_cashier = True
                    cashier_id = cashier_tr.track_id

            # 1. Смещение кассира со стула (Edge: ровно 1 раз)
            if cashier_id in tracks and tracks[cashier_id].confirmed and not cashier_is_standing:
                ctr = tracks[cashier_id]
                shift = np.hypot(ctr.center[0] - ctr.initial_center[0], ctr.center[1] - ctr.initial_center[1])
                body_h = ctr.box[3] - ctr.box[1]
                if shift > (body_h * 0.75):
                    raw_events.append({
                        "timecode_sec": round(sec, 1),
                        "trigger": "CASHIER_STANDUP",
                        "context": "alone",
                        "prompt_focus": (
                            "Кассир встала с рабочего места. Внимательно проверь:\n"
                            "1. Заблокирован ли ноутбук?\n"
                            "2. Заперт ли сейф?\n"
                            "3. Не оставлены ли деньги без присмотра на столе?"
                        )
                    })
                    cashier_is_standing = True

            # 2. Изменение количества стабильно подтвержденных людей
            confirmed_tracks = [t for t in tracks.values() if t.confirmed]
            conf_count = len(confirmed_tracks)

            if prev_confirmed_count is not None and conf_count != prev_confirmed_count:
                if prev_confirmed_count == 1 and conf_count >= 2:
                    raw_events.append({
                        "timecode_sec": round(sec, 1),
                        "trigger": "CLIENT_ARRIVAL",
                        "context": "client",
                        "prompt_focus": (
                            "К кассиру подошел клиент. Внимательно проверь:\n"
                            "1. Проверила ли кассир удостоверение личности клиента перед операцией?\n"
                            "2. Остались ли на столе неубранные деньги от прошлой операции?\n"
                            "3. Закрыта ли за клиентом входная дверь кассового узла?"
                        )
                    })
                elif prev_confirmed_count >= 2 and conf_count == 1:
                    raw_events.append({
                        "timecode_sec": round(sec, 1),
                        "trigger": "CLIENT_DEPARTURE",
                        "context": "client",
                        "prompt_focus": (
                            "Клиент завершил операцию и покинул кассу. Внимательно проверь:\n"
                            "1. Убрала ли кассир принятые/выданные деньги в сейф?\n"
                            "2. Пересчитывает ли кассир деньги вне поля зрения клиента после его ухода (Прав. 231, п. 23)?\n"
                            "3. Закрыта ли дверь кассового узла?"
                        )
                    })
                elif conf_count >= 3 and prev_confirmed_count < 3:
                    raw_events.append({
                        "timecode_sec": round(sec, 1),
                        "trigger": "THIRD_PERSON",
                        "context": "client",
                        "prompt_focus": (
                            "В зоне кассы зафиксирован третий человек. Внимательно проверь:\n"
                            "1. Зашел ли посторонний человек внутрь кассовой кабины?\n"
                            "2. Подсматривает ли посторонний из дверного проема за действиями кассира?"
                        )
                    })

            prev_confirmed_count = conf_count

            # 3. Рука у лица с гистерезисом (вход: ratio < 1.20, выход: ratio > 1.45)
            min_face_ratio = 999.0
            for tr in confirmed_tracks:
                if tr.keypoints is not None and len(tr.keypoints) > 10:
                    kp = tr.keypoints
                    if kp[0][2] > 0.3:
                        sh_w = np.linalg.norm(kp[5][:2] - kp[6][:2]) if (kp[5][2] > 0.3 and kp[6][2] > 0.3) else 0
                        w_dists = [np.linalg.norm(kp[0][:2] - kp[i][:2]) for i in [9, 10] if kp[i][2] > 0.25]
                        if w_dists and sh_w > 0:
                            min_face_ratio = min(min_face_ratio, min(w_dists) / sh_w)

            if not in_hand_to_mouth:
                if min_face_ratio < 1.20:
                    mouth_hold_count += 1
                    if mouth_hold_count >= 2:
                        raw_events.append({
                            "timecode_sec": round(sec, 1),
                            "trigger": "HAND_TO_MOUTH",
                            "context": "alone",
                            "prompt_focus": "Кассир поднесла руку к лицу/рту. Проверь: пьет ли кассир воду/напиток или принимает пищу на рабочем месте?"
                        })
                        in_hand_to_mouth = True
                        mouth_hold_count = 0
                else:
                    mouth_hold_count = 0
            else:
                if min_face_ratio > 1.45:
                    mouth_hold_count += 1
                    if mouth_hold_count >= 2:
                        in_hand_to_mouth = False
                        mouth_hold_count = 0
                else:
                    mouth_hold_count = 0

            # 4. Передача денег с гистерезисом (вход: dist < 0.12*w, выход: dist > 0.18*w)
            min_hand_dist = 9999.0
            if len(confirmed_tracks) >= 2:
                t1, t2 = confirmed_tracks[0], confirmed_tracks[1]
                if t1.keypoints is not None and t2.keypoints is not None:
                    kp1, kp2 = t1.keypoints, t2.keypoints
                    if len(kp1) > 10 and len(kp2) > 10:
                        w1 = [kp1[9][:2], kp1[10][:2]]
                        w2 = [kp2[9][:2], kp2[10][:2]]
                        min_hand_dist = min(np.linalg.norm(a - b) for a in w1 for b in w2)

            if not in_handover:
                if min_hand_dist < (width * 0.12):
                    handover_hold_count += 1
                    if handover_hold_count >= 2:
                        raw_events.append({
                            "timecode_sec": round(sec, 1),
                            "trigger": "HANDOVER",
                            "context": "client",
                            "prompt_focus": (
                                "Кассир и клиент осуществляют передачу ценностей через стол. Проверь:\n"
                                "1. Вернула ли кассир сдачу/деньги клиенту полностью через передаточный лоток?\n"
                                "2. Были ли проверены купюры на счетной машинке или переданы напрямую из рук в руки?\n"
                                "3. Не смешала ли кассир фальшивые/отбракованные купюры с нормальными?"
                            )
                        })
                        in_handover = True
                        handover_hold_count = 0
                else:
                    handover_hold_count = 0
            else:
                if min_hand_dist > (width * 0.18):
                    handover_hold_count += 1
                    if handover_hold_count >= 2:
                        in_handover = False
                        handover_hold_count = 0
                else:
                    handover_hold_count = 0

        # Дебаунс событий + отбор минимального покрывающего набора триггеров
        return select_covering_events(self._debounce_events(raw_events), duration, cover_radius_sec)

    def _debounce_events(self, events):
        if not events:
            return []
        merged = []
        for ev in events:
            if not merged:
                merged.append(ev)
            else:
                last = merged[-1]
                if ev["trigger"] == last["trigger"] and abs(ev["timecode_sec"] - last["timecode_sec"]) < 5.0:
                    continue
                if abs(ev["timecode_sec"] - last["timecode_sec"]) < 2.5:
                    continue
                merged.append(ev)
        return merged

if __name__ == "__main__":
    detector = PoseEventDetector()
    test_videos = ['videos/Х-20.mov', 'videos/Х-4.mov', 'videos/X-1.mov', 'videos/Х-46.mov', 'videos/Х-34.mov']
    for v in test_videos:
        if os.path.exists(v):
            print("=" * 60)
            print(f"Анализ: {v}")
            evs = detector.detect_video_events(v)
            print(f"Найдено событий: {len(evs)}")
            for e in evs:
                print(f"   ⏱️ {e['timecode_sec']}с [{e['trigger']}]: {e['context']} (покрывает: {', '.join(e['merged_triggers'])})")
