@echo off
setlocal EnableExtensions DisableDelayedExpansion

rem Start the Streamlit UI and the CrewAI Study Buddy A2A service.
rem A2A client calls are made by the application through a2a_services\a2a_client.py.

set "ROOT=%~dp0"
set "PYTHON=%ROOT%.venv\Scripts\python.exe"

if not exist "%PYTHON%" (
    echo [ERROR] Python virtual environment not found:
    echo         %PYTHON%
    echo Create it and install requirements.txt before starting the system.
    exit /b 1
)

if exist "%ROOT%.env" (
    rem Load simple KEY=VALUE entries without printing credentials.
    for /f "usebackq eol=# tokens=1,* delims==" %%A in ("%ROOT%.env") do (
        if not "%%A"=="" set "%%A=%%B"
    )
)

echo Starting CrewAI Study Buddy A2A service on http://localhost:9002 ...
start "Learning Accelerator - Study Buddy A2A" /D "%ROOT%" cmd /k ""%PYTHON%" src\crewai_agent\study_buddy.py"

timeout /t 2 /nobreak >nul

echo Starting Streamlit UI ...
start "Learning Accelerator - Streamlit" /D "%ROOT%" cmd /k ""%PYTHON%" -m streamlit run streamlit_app.py"

echo.
echo Learning Accelerator started.
echo Streamlit:    http://localhost:8501
echo Study Buddy:  http://localhost:9002
echo.
echo Keep the three service windows open while using the application.
endlocal
