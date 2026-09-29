@echo off
echo ==================================================
echo    AI PR ^& KOL Specialist DSS - Startup Script
echo ==================================================
echo.

echo [1/2] Memulai FastAPI Backend di Port 9998...
echo Jendela CMD baru akan terbuka untuk Backend.
start "FastAPI Backend (Port 9998)" cmd /k "python -m uvicorn api.main:app --host 0.0.0.0 --port 9998"

echo.
echo Menunggu beberapa detik agar Backend siap...
timeout /t 5 /nobreak > nul

echo.
echo [2/2] Memulai Streamlit Frontend di Port 8501...
echo Jendela CMD baru akan terbuka untuk Frontend, dan Browser Anda akan otomatis terbuka.
start "Streamlit Frontend (Port 8501)" cmd /k "python -m streamlit run frontend/app.py --server.port 8501"

echo.
echo ==================================================
echo SEMUA SISTEM TELAH BERJALAN!
echo - Backend API : http://localhost:9998
echo - Frontend UI : http://localhost:8501
echo.
echo Catatan: Untuk mematikan sistem, cukup tutup (X) kedua jendela CMD warna hitam yang baru terbuka.
echo ==================================================
pause
