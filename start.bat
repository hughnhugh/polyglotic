@echo off
REM Starts vc-translator without a console window - what it prints goes to
REM logs\app.log. Run setup.bat first.
if not exist "%~dp0venv\Scripts\pythonw.exe" (
    echo vc-translator isn't set up yet - double-click setup.bat first.
    pause
    exit /b 1
)
start "" "%~dp0venv\Scripts\pythonw.exe" "%~dp0translate_vc.py" %*
