#!/bin/bash
echo "========================================================"
echo "Starting ForteBank Cash Desk AI Security Server..."
echo "========================================================"
python3 -m uvicorn main:app --host 0.0.0.0 --port 8000
