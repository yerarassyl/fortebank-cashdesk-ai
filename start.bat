@echo off
echo ========================================================
echo Starting ForteBank Cash Desk AI Security Server...
echo ========================================================
python -m uvicorn main:app --host 0.0.0.0 --port 8000
pause
