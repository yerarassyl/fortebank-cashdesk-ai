import os
import sys
import uuid
import json
import time
import cv2
from typing import Dict, Any, Optional
from fastapi import FastAPI, UploadFile, File, Header, BackgroundTasks, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# Настройка путей
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CURRENT_DIR)

from detector import BankViolationDetector, norm_name

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

# Сопоставление внутренних типов детектора с разделами laws.json
LAW_TYPE_MAPPING = {
    "CLIENT_DESK_ACCESS": "Bypassing Cash Tray",
    "NO_ID_VERIFICATION": "Customer Information Exposure",
    "UNLOCKED_SAFE": "Open Safe Door Unattended",
    "LEAVE_WORKPLACE": "Leaving Cash Desk Unattended",
    "NO_COUNTING_MACHINE": "Incorrect Cash Handling",
    "UNAUTHORIZED_ITEMS": "Personal Belongings on Cash Desk",
    "PHONE_USAGE": "Distraction or Smartphone Use",
    "CASHIER_DISTRACTED": "Distraction or Smartphone Use",
    "FOOD_DRINK_AT_DESK": "Food or Beverages at Workstation",
    "THIRD_PERSON_INTRUSION": "Unauthorized Persons in Cash Area",
    "MULTIPLE_CLIENTS": "Unauthorized Persons in Cash Area",
    "INCASSATION_VIOLATION": "Violation of Incassation Protocol",
    "CASH_OUT_OF_VIEW": "Cash Counting Out of Customer View"
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

def process_video_locally(job_id: str, file_path: str, original_filename: str):
    """Локальная обработка видеозаписи на видеокарте этого ПК."""
    try:
        JOBS_DB[job_id]["status"] = "processing"
        t_start = time.time()

        # 1. Получаем характеристики видео через OpenCV
        cap = cv2.VideoCapture(file_path)
        fps_video = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_sec = round(total_frames / fps_video, 1) if fps_video > 0 else 0.0
        cap.release()

        # 2. Запускаем локальный аудит (YOLO Pose + Qwen2-VL-2B FP16)
        print(f"[{job_id}] Анализ видео '{original_filename}' (длительность {duration_sec}с, кадров {total_frames})...")
        raw_result = detector.analyze_file(file_path, filename=original_filename)

        t_end = time.time()
        proc_time = max(0.1, round(t_end - t_start, 2))
        proc_fps = round(total_frames / proc_time, 1) if proc_time > 0 else 350.0
        speed_mult = f"{round(duration_sec / proc_time, 1)}x" if duration_sec > 0 else "10x"

        # 3. Обогащаем инциденты для фронтенда (analyzed_2.html)
        enriched_incidents = []
        violations_list = raw_result.get("violations", [])

        for idx, v in enumerate(violations_list):
            v_type_raw = v.get("type", "LEAVE_WORKPLACE")
            law_key = LAW_TYPE_MAPPING.get(v_type_raw, v_type_raw)

            law_info = LAWS_DB.get(law_key, {
                "title": "Постановление Правления НБРК № 231",
                "quote": "Кассовый работник обязан строго соблюдать правила кассовой дисциплины и регламент обслуживания.",
                "url": "https://adilet.zan.kz/rus/docs/P1900000231"
            })

            # Получаем кадр-доказательство
            frame_url = ""
            if v.get("evidence_frames") and len(v["evidence_frames"]) > 0:
                b64 = v["evidence_frames"][0].get("image_base64", "")
                if b64:
                    frame_url = f"data:image/jpeg;base64,{b64}" if not b64.startswith("data:") else b64

            # Если кадр отсутствует, захватываем напрямую из видео
            if not frame_url:
                c = cv2.VideoCapture(file_path)
                c.set(cv2.CAP_PROP_POS_MSEC, float(v.get("timecode_sec", 0)) * 1000)
                ret, frame = c.read()
                if ret:
                    import base64
                    _, buf = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                    frame_url = "data:image/jpeg;base64," + base64.b64encode(buf).decode('utf-8')
                c.release()

            enriched_incidents.append({
                "id": idx + 1,
                "timeSec": v.get("timecode_sec", 0.0),
                "type": law_key,
                "observation": v.get("description", "Зафиксировано отклонение от кассового регламента."),
                "frame": frame_url,
                "citation": law_info,
                "status": "Violation"
            })

        file_size_mb = round(os.path.getsize(file_path) / (1024 * 1024), 1) if os.path.exists(file_path) else 0.0
        mins = int(duration_sec // 60)
        secs = int(duration_sec % 60)
        duration_str = f"{mins:02d}:{secs:02d}"

        violations_count = len(enriched_incidents)
        compliance_rate = 100 if violations_count == 0 else max(10, 100 - violations_count * 25)

        # Сохраняем финальный результат (совместим И с фронтендом 111, И со спецификацией ТЗ жюри)
        JOBS_DB[job_id].update({
            "status": "done",
            "verdict": raw_result.get("verdict", "violation" if violations_count > 0 else "no_violation"),
            "violations": violations_list,
            "file": {
                "name": original_filename,
                "size": f"{file_size_mb} MB",
                "url": f"/uploads/{os.path.basename(file_path)}"
            },
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)