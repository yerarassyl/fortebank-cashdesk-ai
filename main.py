import os
import sys
import uuid
import json
import time
import cv2
import base64
import numpy as np
from typing import Dict, Any, Optional
from fastapi import FastAPI, UploadFile, File, Header, BackgroundTasks, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# Настройка путей
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CURRENT_DIR)

from detector import BankViolationDetector, norm_name
from legal_base import LEGAL_ARTICLES_RK

app = FastAPI(
    title="Forte AI — Cashier Discipline Analytics",
    description="Локальный AI-сервис видеоаудита кассовых операций ForteBank (Datathon 26)",
    version="2.0.0"
)

# Разрешаем CORS для любых клиентов
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = os.path.join(CURRENT_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

# Загружаем нормативную базу законов laws.json
LAWS_FILE = os.path.join(CURRENT_DIR, "laws.json")
if os.path.exists(LAWS_FILE):
    with open(LAWS_FILE, "r", encoding="utf-8") as f:
        LAWS_DB = json.load(f)
else:
    LAWS_DB = {}

# Сопоставление внутренних типов детектора с каноническими типами из laws_database_json.md
LAW_TYPE_MAPPING = {
    "CLIENT_DESK_ACCESS": "Bypassing Security Pass Tray",
    "NO_ID_VERIFICATION": "Premature Cash Dispensing",
    "NO_ID_CHECK": "Premature Cash Dispensing",
    "UNLOCKED_SAFE": "Open Safe or Storage Cabinet",
    "LEAVE_WORKPLACE": "Leaving Cash Desk Unattended",
    "NO_COUNTING_MACHINE": "Incorrect Cash Handling",
    "UNAUTHORIZED_ITEMS": "Personal Belongings on Work Desk",
    "PHONE_USAGE": "Smartphone Usage in Cash Area",
    "CASHIER_DISTRACTED": "Distracted Cashier or Loss of Control",
    "FOOD_DRINK_AT_DESK": "Personal Belongings on Work Desk",
    "THIRD_PERSON_INTRUSION": "Unauthorized Person in Cash Cabin",
    "MULTIPLE_CLIENTS": "Customer Information Exposure",
    "COUNTING_AWAY_FROM_CLIENT": "Hidden Cash Counting",
    "CASH_OUT_OF_VIEW": "Hidden Cash Counting",
    "UNLOCKED_DOOR": "Unlocked Cash Cabin Door",
    "MONEY_LEFT_BETWEEN_CLIENTS": "Unattended Cash and Valuables",
    "MIXED_CASH": "Unattended Cash and Valuables",
    "INCASSATION_VIOLATION": "Incashment Bag Transfer Without Verification"
}

# Хранилище задач в памяти
JOBS_DB: Dict[str, Dict[str, Any]] = {}

# Глобальный экземпляр детектора
detector: Optional[BankViolationDetector] = None

@app.on_event("startup")
async def startup_event():
    global detector
    print("=" * 60)
    print("[Forte AI] Инициализация локального AI-детектора на GPU...")
    detector = BankViolationDetector()
    print("[Forte AI] Сервер готов к работе! Аудит выполняется локально на этом ПК.")
    print("=" * 60)

def extract_incident_sequence(video_path: str, t_event: float, duration_sec: float):
    """
    Извлекает серию из 11 кадров:
    - 5 кадров ДО момента нарушения (с шагом 0.5 с)
    - 1 центральный кадр МОМЕНТА нарушения (0.0 с)
    - 5 кадров ПОСЛЕ момента нарушения (с шагом 0.5 с)
    А также генерирует склеенный коллаж (composite collage) для мгновенного обзора всей раскадровки.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0 or np.isnan(fps):
        fps = 25.0
    offsets = [-2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
    sequence_frames = []
    collage_thumbs = []

    cell_w, cell_h = 280, 160

    for i, dt in enumerate(offsets):
        t = max(0.0, min(max(0.0, duration_sec - 0.05), t_event + dt))
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ret, frame = cap.read()
        if not ret or frame is None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
            ret, frame = cap.read()
        if not ret or frame is None:
            continue

        # HD кадр для детального зума и рассмотрения вблизи
        h, w = frame.shape[:2]
        scale = 960 / max(h, w)
        hd_frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        _, hd_buf = cv2.imencode('.jpg', hd_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        hd_b64 = base64.b64encode(hd_buf).decode('utf-8')

        is_moment = (dt == 0.0)
        if is_moment:
            label = f"★ МОМЕНТ ({t:.1f}с)"
            tag = "Момент"
        elif dt < 0:
            label = f"{dt:+.1f}с (До)"
            tag = "До"
        else:
            label = f"{dt:+.1f}с (После)"
            tag = "После"

        sequence_frames.append({
            "index": i,
            "offset_sec": dt,
            "timestamp_sec": round(t, 1),
            "label": label,
            "tag": tag,
            "is_moment": is_moment,
            "image_data": f"data:image/jpeg;base64,{hd_b64}"
        })

        # Миниатюра для склеенного коллажа
        thumb = cv2.resize(frame, (cell_w, cell_h))
        bg_color = (20, 20, 200) if is_moment else (45, 45, 45)
        cv2.rectangle(thumb, (0, 0), (cell_w, 24), bg_color, -1)
        txt = f"НАРУШЕНИЕ [{t:.1f}s]" if is_moment else f"{label} [{t:.1f}s]"
        cv2.putText(thumb, txt, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        if is_moment:
            cv2.rectangle(thumb, (0, 0), (cell_w - 1, cell_h - 1), (0, 0, 255), 3)
        collage_thumbs.append(thumb)

    cap.release()

    collage_b64 = ""
    if collage_thumbs:
        collage_w = cell_w * 6
        collage_h = cell_h * 2
        canvas = np.zeros((collage_h, collage_w, 3), dtype=np.uint8)
        for idx, thumb in enumerate(collage_thumbs):
            r = idx // 6
            c = idx % 6
            canvas[r * cell_h:(r + 1) * cell_h, c * cell_w:(c + 1) * cell_w] = thumb
        _, col_buf = cv2.imencode('.jpg', canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        collage_b64 = f"data:image/jpeg;base64,{base64.b64encode(col_buf).decode('utf-8')}"

    return sequence_frames, collage_b64

def process_video_locally(job_id: str, file_path: str, original_filename: str):
    """Локальная обработка видеозаписи на видеокарте этого ПК."""
    try:
        JOBS_DB[job_id]["status"] = "processing"
        JOBS_DB[job_id]["progress"] = 15
        JOBS_DB[job_id]["stage"] = "Валидация видео и параметров потока"
        JOBS_DB[job_id]["stage_step"] = 1
        t_start = time.time()

        # 1. Получаем характеристики видео через OpenCV
        cap = cv2.VideoCapture(file_path)
        fps_video = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_sec = round(total_frames / fps_video, 1) if fps_video > 0 else 0.0
        cap.release()

        # 2. Запускаем локальный аудит (YOLO Pose + Qwen2-VL-2B FP16)
        JOBS_DB[job_id]["progress"] = 35
        JOBS_DB[job_id]["stage"] = "Покадровое сканирование поз кассира (YOLO11m-Pose)"
        JOBS_DB[job_id]["stage_step"] = 2
        print(f"[{job_id}] Анализ видео '{original_filename}' (длительность {duration_sec}с, кадров {total_frames})...")
        raw_result = detector.analyze_file(file_path, filename=original_filename)

        JOBS_DB[job_id]["progress"] = 70
        JOBS_DB[job_id]["stage"] = "Анализ зон кассы и проверка VLM (Qwen2-VL)"
        JOBS_DB[job_id]["stage_step"] = 3

        t_end = time.time()
        actual_time = max(0.01, t_end - t_start)

        # Реальный замер времени и FPS работы нейросетевого пайплайна
        proc_time = round(actual_time, 2)
        proc_fps = round(total_frames / proc_time, 1) if proc_time > 0 else 0.0
        speed_mult = f"{round(duration_sec / proc_time, 1)}x" if (proc_time > 0 and duration_sec > 0) else "1.0x"

        # 3. Обогащаем инциденты для фронтенда (analyzed_2.html)
        enriched_incidents = []
        violations_list = raw_result.get("violations", [])

        for idx, v in enumerate(violations_list):
            v_type_raw = v.get("type", "LEAVE_WORKPLACE")
            canonical_type = LAW_TYPE_MAPPING.get(v_type_raw, v_type_raw)

            # Получаем запись из новой нормативной базы laws.json
            law_entry = LAWS_DB.get(v_type_raw) or LAWS_DB.get(canonical_type) or {}
            
            # Фоллбэк на legal_base если есть специальное примечание
            legal_entry = LEGAL_ARTICLES_RK.get(v_type_raw) or LEGAL_ARTICLES_RK.get(canonical_type) or {}

            law_article = law_entry.get("law") or law_entry.get("law_title") or legal_entry.get("law_source") or "ПП НБРК № 231"
            law_name_ru = law_entry.get("title") or law_entry.get("name_ru") or v.get("description", "Нарушение кассовой дисциплины")
            law_quote = law_entry.get("quote") or legal_entry.get("law_quote") or "Кассовый работник обязан строго соблюдать правила кассовой дисциплины и регламент обслуживания."
            law_visual = law_entry.get("visual_sign", "")
            law_group = law_entry.get("group") or "Нарушения кассовой дисциплины"
            criticality = law_entry.get("criticality") or "HIGH"
            law_url = law_entry.get("url") or "https://adilet.zan.kz/rus/docs/P1900000231"
            law_note = legal_entry.get("note", "")

            full_quote = law_quote
            if law_note and law_note not in full_quote:
                full_quote = f"{law_quote} (Примечание: {law_note})"

            law_info = {
                "title": law_article,
                "law": law_article,
                "name_ru": law_name_ru,
                "quote": full_quote,
                "visual_sign": law_visual,
                "group": law_group,
                "criticality": criticality,
                "url": law_url,
                "note": law_note
            }

            obs_text = v.get("description", law_name_ru)
            if law_note and "Примечание" not in obs_text:
                obs_text = f"{obs_text.rstrip('.')} Примечание: {law_note}"

            t_viol = float(v.get("timecode_sec", 0.0))

            # Извлекаем серию из 11 кадров (5 до, момент, 5 после) и коллаж
            seq_frames, collage_img = extract_incident_sequence(file_path, t_viol, duration_sec)

            # Главный кадр инцидента — момент нарушения
            moment_frame = ""
            for sf in seq_frames:
                if sf.get("is_moment"):
                    moment_frame = sf["image_data"]
                    break
            if not moment_frame and seq_frames:
                moment_frame = seq_frames[len(seq_frames) // 2]["image_data"]

            # Если кадры не извлеклись через sequence, берем из evidence_frames детектора
            if not moment_frame and v.get("evidence_frames"):
                b64 = v["evidence_frames"][0].get("image_base64", "")
                if b64:
                    moment_frame = f"data:image/jpeg;base64,{b64}" if not b64.startswith("data:") else b64

            # Если всё еще нет кадра, напрямую считываем кадр из видеофайла
            if not moment_frame:
                c_fb = cv2.VideoCapture(file_path)
                c_fb.set(cv2.CAP_PROP_POS_MSEC, t_viol * 1000)
                r_fb, f_fb = c_fb.read()
                if not r_fb or f_fb is None:
                    c_fb.set(cv2.CAP_PROP_POS_FRAMES, int(t_viol * 25.0))
                    r_fb, f_fb = c_fb.read()
                if not r_fb or f_fb is None:
                    c_fb.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    r_fb, f_fb = c_fb.read()
                if r_fb and f_fb is not None:
                    _, buf_fb = cv2.imencode('.jpg', f_fb, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                    moment_frame = "data:image/jpeg;base64," + base64.b64encode(buf_fb).decode('utf-8')
                c_fb.release()

            if not seq_frames and moment_frame:
                seq_frames = [{
                    "index": 5,
                    "offset_sec": 0.0,
                    "timestamp_sec": t_viol,
                    "label": f"★ МОМЕНТ ({t_viol:.1f}с)",
                    "tag": "Момент",
                    "is_moment": True,
                    "image_data": moment_frame
                }]
            if not collage_img and moment_frame:
                collage_img = moment_frame

            enriched_incidents.append({
                "id": idx + 1,
                "timeSec": t_viol,
                "type": canonical_type,
                "type_raw": v_type_raw,
                "type_ru": law_name_ru,
                "group": law_group,
                "criticality": criticality,
                "observation": obs_text,
                "frame": moment_frame,
                "collage": collage_img,
                "sequence_frames": seq_frames,
                "citation": law_info,
                "status": "Violation"
            })

        JOBS_DB[job_id]["progress"] = 90
        JOBS_DB[job_id]["stage"] = "Формирование юридического отчета и раскадровки"
        JOBS_DB[job_id]["stage_step"] = 4

        # Извлечение чистого кадра кассовой зоны (для видео без нарушений)
        clean_frame_b64 = ""
        c = cv2.VideoCapture(file_path)
        t_clean = min(duration_sec - 0.1, max(0.5, duration_sec / 2.0))
        c.set(cv2.CAP_PROP_POS_MSEC, t_clean * 1000)
        ret, cf = c.read()
        if ret:
            h, w = cf.shape[:2]
            scale = 960 / max(h, w)
            hd_cf = cv2.resize(cf, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
            _, buf = cv2.imencode('.jpg', hd_cf, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            clean_frame_b64 = "data:image/jpeg;base64," + base64.b64encode(buf).decode('utf-8')
        c.release()

        clean_citation = {
            "title": "ПП НБРК № 231 (Правила ведения кассовых операций)",
            "law": "ПП НБРК № 231",
            "name_ru": "Кассовый регламент и порядок ведения кассовых операций соблюдены в полном объеме",
            "quote": "Кассир осуществляет прием и выдачу наличных денег и ценностей клиентам исключительно через передаточный кассовый лоток в пределах защищенной кассовой кабины, обеспечивая непрерывный визуальный контроль и соблюдение тайны операций.",
            "visual_sign": "Кассир находится на рабочем месте, сейф закрыт, посторонних лиц и предметов нет.",
            "url": "https://adilet.zan.kz/rus/docs/P1900000231",
            "group": "Соответствие нормативным требованиям НБ РК",
            "criticality": "NORMAL",
            "note": "Нарушений кассовой дисциплины, оставления рабочего места, несанкционированного доступа или использования смартфона не выявлено."
        }

        file_size_mb = round(os.path.getsize(file_path) / (1024 * 1024), 1) if os.path.exists(file_path) else 0.0
        mins = int(duration_sec // 60)
        secs = int(duration_sec % 60)
        duration_str = f"{mins:02d}:{secs:02d}"

        violations_count = len(enriched_incidents)
        compliance_rate = 100 if violations_count == 0 else max(10, 100 - violations_count * 25)

        # Сохраняем финальный результат (совместим И с фронтендом 111, И со спецификацией ТЗ жюри)
        JOBS_DB[job_id].update({
            "status": "done",
            "progress": 100,
            "stage": "Аудит завершен",
            "stage_step": 4,
            "verdict": raw_result.get("verdict", "violation" if violations_count > 0 else "no_violation"),
            "violations": violations_list,
            "file": {
                "name": original_filename,
                "size": f"{file_size_mb} MB",
                "url": f"/uploads/{os.path.basename(file_path)}",
                "clean_frame": clean_frame_b64
            },
            "clean_frame": clean_frame_b64,
            "clean_citation": clean_citation,
            "branch": {"name": "Almaty Branch #12", "id": "0123"},
            "analyzedAt": time.strftime("%d %b %Y, %H:%M"),
            "durationSec": duration_sec,
            "durationStr": duration_str,
            "processingTimeSec": proc_time,
            "fps": proc_fps,
            "speedMultiplier": speed_mult,
            "complianceRate": compliance_rate,
            "violationsCount": violations_count,
            "incidents": enriched_incidents
        })
        print(f"[{job_id}] Успешно завершено за {proc_time}с ({proc_fps} FPS, {violations_count} нарушений).")

    except Exception as e:
        print(f"[{job_id}] Ошибка анализа: {e}")
        JOBS_DB[job_id]["status"] = "failed"
        JOBS_DB[job_id]["error"] = str(e)


# ==============================================================================
# API ЭНДПОИНТЫ (БЕЗ ОБЯЗАТЕЛЬНОГО X-API-KEY)
# ==============================================================================

@app.post("/api/v1/analyze", summary="Загрузка видео на аудит")
@app.post("/analyze", summary="Анализ видео (алиас)")
async def analyze_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key")
):
    """
    Принимает видеофайл и запускает его обработку на видеокарте этого ПК.
    X-API-Key опционален (авторизация не блокирует запрос).
    """
    job_id = f"job_{uuid.uuid4().hex[:8]}"
    
    # Сохраняем файл в uploads для возможности воспроизведения в браузере
    safe_filename = f"{job_id}_{file.filename}"
    save_path = os.path.join(UPLOAD_DIR, safe_filename)

    with open(save_path, "wb") as buffer:
        while chunk := await file.read(1024 * 1024):  # 1MB чанки
            buffer.write(chunk)

    JOBS_DB[job_id] = {
        "job_id": job_id,
        "status": "processing",
        "verdict": None,
        "violations": [],
        "incidents": []
    }

    # Запускаем локальный анализ в фоне
    background_tasks.add_task(process_video_locally, job_id, save_path, file.filename)

    return {"job_id": job_id, "status": "processing"}


@app.get("/api/v1/jobs/{job_id}", summary="Получение статуса и результатов аудита")
@app.get("/jobs/{job_id}", summary="Статус задачи (алиас)")
async def get_job_status(
    job_id: str,
    x_api_key: Optional[str] = Header(None, alias="X-API-Key")
):
    """Возвращает статус задачи и результаты аудита."""
    if job_id not in JOBS_DB:
        raise HTTPException(status_code=404, detail="Задача не найдена")
    return JOBS_DB[job_id]


@app.get("/api/v1/health")
async def health_check():
    import torch
    return {
        "status": "ok",
        "gpu_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "model": "YOLO11m-Pose + Qwen2-VL-2B-Instruct (FP16)",
        "offline_engine": True
    }


@app.get("/api/v1/laws", summary="Нормативно-правовая база законов НБ РК")
async def get_laws_database():
    """Возвращает базу канонических нарушений кассовой дисциплины НБ РК (laws_database_json.md)."""
    return {
        "status": "ok",
        "count": len(LAWS_DB),
        "laws": LAWS_DB
    }


# ==============================================================================
# РАЗДАЧА ВЕБ-ИНТЕРФЕЙСА (empty_2.html и analyzed_2.html)
# ==============================================================================

@app.get("/", response_class=HTMLResponse)
@app.get("/empty_2.html", response_class=HTMLResponse)
@app.get("/empty.html", response_class=HTMLResponse)
async def serve_empty_ui():
    html_path = os.path.join(CURRENT_DIR, "empty_2.html")
    if not os.path.exists(html_path):
        html_path = os.path.join(CURRENT_DIR, "empty.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return f.read()


@app.get("/analyzed_2.html", response_class=HTMLResponse)
@app.get("/analyzed.html", response_class=HTMLResponse)
async def serve_analyzed_ui():
    html_path = os.path.join(CURRENT_DIR, "analyzed_2.html")
    if not os.path.exists(html_path):
        html_path = os.path.join(CURRENT_DIR, "analyzed.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return f.read()


@app.get("/forte_logo.png")
@app.get("/static/forte_logo.png")
async def serve_forte_png_logo():
    logo_path = os.path.join(CURRENT_DIR, "forte_logo.png")
    if os.path.exists(logo_path):
        return FileResponse(logo_path, media_type="image/png")
    raise HTTPException(status_code=404, detail="Logo not found")


@app.get("/fortebank_logo.svg")
@app.get("/static/fortebank_logo.svg")
async def serve_fortebank_logo():
    logo_path = os.path.join(CURRENT_DIR, "fortebank_logo.svg")
    if os.path.exists(logo_path):
        return FileResponse(logo_path, media_type="image/svg+xml")
    raise HTTPException(status_code=404, detail="Logo not found")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)