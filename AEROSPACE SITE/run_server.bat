@echo off

REM Kill any old server processes on port 5000
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":5000" ^| findstr "LISTENING"') do (
    taskkill /F /PID %%a >nul 2>&1
)

REM Load API keys from .env file
for /f "usebackq delims=" %%x in ("%~dp0.env") do set "%%x"

echo.
echo === Loaded API keys ===
echo OPENROUTER: %OPENROUTER_API_KEY:~0,25%...
echo TELEGRAM:   %TELEGRAM_BOT_TOKEN:~0,15%...
echo ========================
echo.

python -u server.py
pause
