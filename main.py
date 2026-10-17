import os
import uuid
import shutil
import asyncio
from typing import Dict, Any
from fastapi import FastAPI, UploadFile, File, Header, HTTPException, BackgroundTasks, status
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from detector import BankViolationDetector

# Конфигурация API ключа (согласно разделу 10 ТЗ)
DEFAULT_API_KEY = os.getenv("FORTE_API_KEY", "forte-secret-key-2026")
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI(
    title="ForteBank Cash Desk AI Security Service",
    description="Сервис детекции нарушений кассовой дисциплины ForteBank (Datathon 26)",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Статическая раздача видеофайлов для демонстрации в веб-интерфейсе
if os.path.exists("videos"):
    app.mount("/videos", StaticFiles(directory="videos"), name="videos")

# Хранилище задач в памяти
jobs_db: Dict[str, Dict[str, Any]] = {}

# Глобальный экземпляр детектора (загружается при старте приложения)
detector: BankViolationDetector = None

@app.on_event("startup")
async def startup_event():
    global detector
    print("=" * 60)
    print("Инициализация сервиса ForteBank AI Security...")
    detector = BankViolationDetector()
    print(f"Сервер готов к приему запросов. API-Key: {DEFAULT_API_KEY}")
    print("=" * 60)

def verify_api_key(x_api_key: str = Header(None, alias="X-API-Key")):
    """Проверка ключа доступа согласно ТЗ (раздел 10)."""
    if not x_api_key or x_api_key != DEFAULT_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Неверный или отсутствующий API-ключ (заголовок X-API-Key)"
        )

def process_video_job(job_id: str, file_path: str, original_filename: str):
    """Фоновая обработка видеозаписи."""
    try:
        jobs_db[job_id]["status"] = "processing"
        result = detector.analyze_file(file_path, filename=original_filename)
        
        jobs_db[job_id]["status"] = "done"
        jobs_db[job_id]["verdict"] = result["verdict"]
        jobs_db[job_id]["violations"] = result["violations"]
    except Exception as e:
        print(f"[Job {job_id}] Ошибка обработки: {e}")
        jobs_db[job_id]["status"] = "failed"
        jobs_db[job_id]["error"] = str(e)
    finally:
        # Удаляем временный файл для освобождения диска (при необходимости можно сохранить)
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception:
                pass

# ==============================================================================
# API ЭНДПОИНТЫ ДЛЯ ЖЮРИ (СТРОГО ПО ТЗ РАЗДЕЛ 10)
# ==============================================================================

@app.post("/api/v1/analyze", summary="Загрузить видео для анализа")
async def analyze_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    x_api_key: str = Header(None, alias="X-API-Key")
):
    """
    POST /api/v1/analyze
    Принимает .mov файл, возвращает job_id.
    """
    verify_api_key(x_api_key)
    
    job_id = uuid.uuid4().hex[:8]
    temp_filename = f"{job_id}_{file.filename}"
    save_path = os.path.join(UPLOAD_DIR, temp_filename)
    
    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    jobs_db[job_id] = {
        "job_id": job_id,
        "status": "processing",
        "verdict": None,
        "violations": []
    }
    
    # Запуск фонового анализа
    background_tasks.add_task(process_video_job, job_id, save_path, file.filename)
    
    return {"job_id": job_id}

@app.get("/api/v1/jobs/{job_id}", summary="Получить статус и результат анализа")
async def get_job_status(
    job_id: str,
    x_api_key: str = Header(None, alias="X-API-Key")
):
    """
    GET /api/v1/jobs/{job_id}
    Возвращает результат по ТЗ:
    {
      "job_id": "3f2a9c1e",
      "status": "done",
      "verdict": "violation",
      "violations": [...]
    }
    """
    verify_api_key(x_api_key)
    
    if job_id not in jobs_db:
        raise HTTPException(status_code=404, detail="Задача не найдена")
        
    job_data = jobs_db[job_id]
    
    if job_data["status"] == "processing":
        return {
            "job_id": job_id,
            "status": "processing"
        }
    elif job_data["status"] == "failed":
        return {
            "job_id": job_id,
            "status": "failed",
            "error": job_data.get("error", "Неизвестная ошибка")
        }
    else:
        return {
            "job_id": job_id,
            "status": "done",
            "verdict": job_data["verdict"],
            "violations": job_data["violations"]
        }

@app.get("/api/v1/health")
async def health_check():
    import torch
    return {
        "status": "ok",
        "gpu_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "model": "Qwen2-VL-2B-Instruct",
        "api_key": DEFAULT_API_KEY
    }

# ==============================================================================
# ВЕБ-ИНТЕРФЕЙС ДЛЯ ПОЛЬЗОВАТЕЛЕЙ И ЖЮРИ
# ==============================================================================

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
