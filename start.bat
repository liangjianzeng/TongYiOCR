@echo off
chcp 936 >nul
cd /d %~dp0
if not exist .venv\Scripts\activate.bat (
  echo [ERROR] .venv not found. Run: python -m venv --system-site-packages .venv && pip install -r requirements.txt
  pause
  exit /b 1
)
call .venv\Scripts\activate.bat
set PADDLEX_HOME=%CD%\.cache\paddlex
if not exist "%PADDLEX_HOME%" mkdir "%PADDLEX_HOME%"
python run.py
