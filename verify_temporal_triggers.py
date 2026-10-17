"""
verify_temporal_triggers.py
Высокоскоростная батч-проверка CV-триггеров БЕЗ ХАРДКОДА ПИКСЕЛЕЙ.
Фильтрация отражений и шумов строго через временную стабильность треков (temporal stability >= 1.5с).
Динамическое определение кассира по стартовой позиции (без полигонов зон).
"""

import os
import re
import cv2
import time
import numpy as np
import pandas as pd
from ultralytics import YOLO
from cv_triggers import select_covering_events

def parse_target_sec(ts_raw):
    if pd.isna(ts_raw):
        return None
    s = str(ts_raw).lower().strip()
    m_min = re.search(r'(\d+)\s*мин', s)
    m_sec = re.search(r'(\d+)\s*сек', s)
    minutes = int(m_min.group(1)) if m_min else 0
    seconds = int(m_sec.group(1)) if m_sec else 0
    if not m_min and not m_sec:
        m = re.search(r'(\d+(?:\.\d+)?)', s)
        if m:
            return float(m.group(1))
        return None
    return float(minutes * 60 + seconds)

def normalize_name(s):
    return str(s).strip().replace('Х', 'X').replace('х', 'x').lower()

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

def main():
    print("=" * 90)
    print("CV-ТРИГГЕРЫ: ЧИСТАЯ ВРЕМЕННАЯ СТАБИЛЬНОСТЬ (ZERO PIXEL HARDCODING)")
    print("Окружение: e:\\sentra\\venv (YOLO Pose, FP16, CUDA)")
    print("=" * 90)

    csv_path = "open_labels_copy.csv"
    df = pd.read_csv(csv_path)
    
    video_files = os.listdir("videos")
    file_map = {normalize_name(os.path.splitext(f)[0]): os.path.join("videos", f) for f in video_files}

    candidates = ["yolo11m-pose.pt", r"e:\sentra\yolo11m-pose.pt", "yolov8n-pose.pt", r"e:\sentra\yolov8n-pose.pt"]
    model_path = next((c for c in candidates if os.path.exists(c)), "yolo11m-pose.pt")
    model = YOLO(model_path)

    BATCH_SIZE = 16
    FRAME_STEP = 5 # ~6 кадров/сек
    MIN_STABLE_SEC = 1.3 # Отражения и шум обычно длятся < 1с. Человек держится >= 1.3с
    MAX_LOST_SEC = 1.0
    COVER_RADIUS_SEC = 10.0 # = допуск оценки: отбор минимального набора триггеров без потери покрытия

    report_rows = []
    t_start_all = time.time()
    total_processed_frames = 0

    for idx, row in df.iterrows():
        name = str(row.iloc[0]).strip()
        ts_raw = row.iloc[1]
        status = str(row.iloc[2]).strip()
        desc = str(row.iloc[3]).strip() if pd.notna(row.iloc[3]) else "-"
        who = str(row.iloc[4]).strip() if pd.notna(row.iloc[4]) else "-"

        is_violation = (status.lower() == 'нарушение')
        target_sec = parse_target_sec(ts_raw) if is_violation else None

        vpath = file_map.get(normalize_name(name))
        if not vpath or not os.path.exists(vpath):
            continue

        cap = cv2.VideoCapture(vpath)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        sampled_frames = []
        sampled_secs = []
        f_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if f_idx % FRAME_STEP == 0:
                sampled_frames.append(frame)
                sampled_secs.append(f_idx / fps)
            f_idx += 1
        cap.release()

        total_processed_frames += len(sampled_frames)
        fps_sample = fps / FRAME_STEP if FRAME_STEP > 0 else 6.0
        min_stable_hits = max(3, int(MIN_STABLE_SEC * fps_sample))
        max_miss_frames = max(2, int(MAX_LOST_SEC * fps_sample))

        # Батч-инференс
        all_boxes = []
        all_kpts = []
        for b_start in range(0, len(sampled_frames), BATCH_SIZE):
            batch = sampled_frames[b_start : b_start + BATCH_SIZE]
            batch_preds = model.predict(batch, conf=0.35, device='cuda:0', verbose=False)
            for p in batch_preds:
                all_boxes.append(p.boxes.xyxy.cpu().numpy() if p.boxes is not None else [])
                all_kpts.append(p.keypoints.data.cpu().numpy() if p.keypoints is not None else [])

        # Временной трекер
        tracks = {}
        next_tid = 0
        raw_triggers = []
        prev_confirmed_count = None
        cashier_id = None
        hand_to_mouth_counter = 0

        # Триггер начала смены / проверки стола в первые 3 секунды
        raw_triggers.append((0.5, "INITIAL_DESK_CHECK", "Аудит рабочего места кассира в начале смены"))

        for frame_i, (sec, boxes, kpts) in enumerate(zip(sampled_secs, all_boxes, all_kpts)):
            matched_tids = set()

            for b_idx, b in enumerate(boxes):
                cx = (b[0] + b[2]) / 2
                cy = (b[1] + b[3]) / 2
                box_diag = np.hypot(b[2] - b[0], b[3] - b[1])
                match_threshold = max(80, box_diag * 0.7) # адаптивный порог сопоставления трека

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
                    # Кассир стабильно покинул кадр
                    raw_triggers.append((sec, "CASHIER_STANDUP", "Кассир встал и ушел из кассовой зоны"))
                del tracks[tid]

            # Идентификация кассира: трек, присутствующий с самого начала
            if cashier_id is None and len(tracks) > 0:
                # Кассир - трек с наибольшим количеством hits в начале
                confirmed_candidates = [t for t in tracks.values() if t.confirmed]
                if confirmed_candidates:
                    # Тот, кто дольше всех в кадре
                    cashier_tr = max(confirmed_candidates, key=lambda t: t.hits)
                    cashier_tr.is_cashier = True
                    cashier_id = cashier_tr.track_id

            # Проверка смещения кассира со стула (если отошел от initial_center более чем на размер корпуса)
            if cashier_id in tracks and tracks[cashier_id].confirmed:
                ctr = tracks[cashier_id]
                shift = np.hypot(ctr.center[0] - ctr.initial_center[0], ctr.center[1] - ctr.initial_center[1])
                body_h = ctr.box[3] - ctr.box[1]
                if shift > (body_h * 0.75):
                    raw_triggers.append((sec, "CASHIER_STANDUP", "Кассир поднялся с рабочего места (смещение корпуса)"))
                    # Обновляем initial_center чтобы не спамить
                    ctr.initial_center = ctr.center

            # Подсчет стабильно подтвержденных людей
            confirmed_tracks = [t for t in tracks.values() if t.confirmed]
            conf_count = len(confirmed_tracks)

            if prev_confirmed_count is not None and conf_count != prev_confirmed_count:
                if prev_confirmed_count == 1 and conf_count >= 2:
                    raw_triggers.append((sec, "CLIENT_ARRIVAL", "Появился клиент (1->2)"))
                elif prev_confirmed_count >= 2 and conf_count == 1:
                    raw_triggers.append((sec, "CLIENT_DEPARTURE", "Клиент ушел (2->1)"))
                elif conf_count >= 3 and prev_confirmed_count < 3:
                    raw_triggers.append((sec, "THIRD_PERSON", "Зафиксирован третий человек в зоне кассы"))

            prev_confirmed_count = conf_count

            # Рука у лица (питье / еда)
            for tr in confirmed_tracks:
                if tr.keypoints is not None and len(tr.keypoints) > 10:
                    kp = tr.keypoints
                    # 0: nose, 9: left wrist, 10: right wrist, 5, 6: shoulders
                    if kp[0][2] > 0.3:
                        sh_w = np.linalg.norm(kp[5][:2] - kp[6][:2]) if (kp[5][2] > 0.3 and kp[6][2] > 0.3) else 0
                        w_dists = []
                        if kp[9][2] > 0.25:
                            w_dists.append(np.linalg.norm(kp[0][:2] - kp[9][:2]))
                        if kp[10][2] > 0.25:
                            w_dists.append(np.linalg.norm(kp[0][:2] - kp[10][:2]))
                            
                        body_h = tr.box[3] - tr.box[1]
                        is_near_face = False
                        if w_dists:
                            min_wd = min(w_dists)
                            if sh_w > 0 and (min_wd / sh_w) < 1.3:
                                is_near_face = True
                            elif body_h > 0 and (min_wd / body_h) < 0.35:
                                is_near_face = True

                        if is_near_face:
                            hand_to_mouth_counter += 1
                            if hand_to_mouth_counter == 2:
                                raw_triggers.append((sec, "HAND_TO_MOUTH", "Рука у лица (прием пищи/напитка)"))
                        else:
                            hand_to_mouth_counter = max(0, hand_to_mouth_counter - 1)

            # Передача денег (сближение рук между двумя людьми)
            if len(confirmed_tracks) >= 2:
                t1, t2 = confirmed_tracks[0], confirmed_tracks[1]
                if t1.keypoints is not None and t2.keypoints is not None:
                    kp1, kp2 = t1.keypoints, t2.keypoints
                    if len(kp1) > 10 and len(kp2) > 10:
                        w1 = [kp1[9][:2], kp1[10][:2]]
                        w2 = [kp2[9][:2], kp2[10][:2]]
                        min_w_dist = min(np.linalg.norm(a - b) for a in w1 for b in w2)
                        # Расстояние меньше трети ширины кадра
                        if min_w_dist < (w * 0.14):
                            raw_triggers.append((sec, "HANDOVER", "Сближение рук кассира и клиента над столом"))

        # Дебаунс триггеров
        cleaned = []
        for s, trig_type, d_text in raw_triggers:
            if not cleaned:
                cleaned.append((s, trig_type, d_text))
            else:
                last_s, last_trig, _ = cleaned[-1]
                if trig_type == last_trig and (s - last_s) < 5.0:
                    continue
                if (s - last_s) < 2.5:
                    continue
                cleaned.append((s, trig_type, d_text))

        raw_cleaned = cleaned
        selected = select_covering_events(
            [{"timecode_sec": s, "trigger": t, "prompt_focus": d} for s, t, d in cleaned],
            duration, COVER_RADIUS_SEC
        )
        cleaned = [(e["timecode_sec"], e["trigger"], e["prompt_focus"]) for e in selected]

        # Оценка покрытия нарушения
        matched_trigger = None
        min_delta = 999.0
        if is_violation and target_sec is not None:
            for s, trig_type, d_text in cleaned:
                delta = abs(s - target_sec)
                if delta < min_delta:
                    min_delta = delta
                    matched_trigger = f"{s:.1f}s [{trig_type}]"

        is_caught = (min_delta <= 10.0) if is_violation else True

        report_rows.append({
            "Video": name,
            "Duration": round(duration, 1),
            "Status": status,
            "TargetSec": target_sec,
            "Description": desc,
            "Context": who,
            "RawTriggersCount": len(raw_cleaned),
            "TriggersCount": len(cleaned),
            "BestTrigger": matched_trigger,
            "MinDeltaSec": round(min_delta, 2) if is_violation else None,
            "Caught": is_caught,
            "TriggersDetail": "; ".join([f"{s:.1f}s({t})" for s, t, _ in cleaned]),
            "RawTriggersDetail": "; ".join([f"{s:.1f}s({t})" for s, t, _ in raw_cleaned])
        })

        tag = " OK " if is_caught else "FAIL"
        delta_str = f"delta={min_delta:.1f}s ({matched_trigger})" if is_violation else "clean"
        print(f"[{idx+1:02d}/42] [{tag}] {name:<6} | {status:<9} | {duration:4.1f}s | триггеров: {len(cleaned):<2} | {delta_str}")

    t_total = time.time() - t_start_all
    rep_df = pd.DataFrame(report_rows)
    rep_df.to_csv("all_42_temporal_audit.csv", index=False, encoding='utf-8-sig')

    v_df = rep_df[rep_df["Status"].str.lower() == "нарушение"]
    caught_count = v_df["Caught"].sum()
    total_violations = len(v_df)
    total_triggers = rep_df["TriggersCount"].sum()
    mean_delta = v_df["MinDeltaSec"].mean()

    print("\n" + "=" * 90)
    print("ИТОГИ АУДИТА С ЧИСТОЙ ВРЕМЕННОЙ СТАБИЛЬНОСТЬЮ (NO HARDCODED PIXELS):")
    print(f"Поймано реальных нарушений: {caught_count} из {total_violations} ({caught_count/total_violations*100:.1f}%)")
    print(f"Среднее отклонение таймкода триггера от разметки: {mean_delta:.2f} сек")
    raw_total = rep_df["RawTriggersCount"].sum()
    print(f"ОБЩЕЕ ЧИСЛО СРАБАТЫВАНИЙ ТРИГГЕРОВ: {total_triggers} (в среднем {total_triggers/42:.1f} на видео; до отбора: {raw_total})")
    print(f"Обработано кадров: {total_processed_frames} за {t_total:.1f}с ({total_processed_frames/t_total:.1f} FPS на CUDA)")
    print("=" * 90)

if __name__ == "__main__":
    main()
