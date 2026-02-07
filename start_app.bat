@echo off
echo Starting AI Stroke Detection System...

start "Backend Server" cmd /k "cd backend && python main.py"
start "Frontend App" cmd /k "cd frontend && npm run dev"

echo Servers started in new windows.
echo Backend: http://localhost:8000
echo Frontend: http://localhost:5173
pause
