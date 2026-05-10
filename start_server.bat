@echo off
title XSIVE Server

echo.
echo  ====================================================
echo   XSIVE - Spectral Curation Engine
echo   Motor de Curacion de Audio con CNN
echo  ====================================================
echo.

cd /d "%~dp0"

:: --- Buscar Python ---
set PYTHON_CMD=

if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" (
    set PYTHON_CMD="%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
    goto :run
)
if exist "%LOCALAPPDATA%\Programs\Python\Python311\python.exe" (
    set PYTHON_CMD="%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
    goto :run
)
if exist "%LOCALAPPDATA%\Programs\Python\Python310\python.exe" (
    set PYTHON_CMD="%LOCALAPPDATA%\Programs\Python\Python310\python.exe"
    goto :run
)
if exist "%LOCALAPPDATA%\Programs\Python\Python39\python.exe" (
    set PYTHON_CMD="%LOCALAPPDATA%\Programs\Python\Python39\python.exe"
    goto :run
)
if exist "C:\Python312\python.exe" (
    set PYTHON_CMD="C:\Python312\python.exe"
    goto :run
)
if exist "C:\Python311\python.exe" (
    set PYTHON_CMD="C:\Python311\python.exe"
    goto :run
)
if exist "C:\Python310\python.exe" (
    set PYTHON_CMD="C:\Python310\python.exe"
    goto :run
)

:: Intentar py launcher
where py >nul 2>&1
if %ERRORLEVEL%==0 (
    set PYTHON_CMD=py
    goto :run
)

echo  [ERROR] Python no encontrado en rutas comunes.
echo.
echo  Soluciones:
echo   1. Instala Python desde: https://www.python.org/downloads/
echo      (marca "Add Python to PATH" durante la instalacion)
echo   2. O ejecuta manualmente:
echo      py server.py
echo.
pause
exit /b 1

:run
echo  [OK] Python encontrado: %PYTHON_CMD%
echo.

:: --- Instalar Flask ---
echo  [1/2] Instalando Flask y Flask-CORS...
%PYTHON_CMD% -m pip install flask flask-cors --quiet --disable-pip-version-check
if %ERRORLEVEL% NEQ 0 (
    echo  [ERROR] No se pudo instalar Flask.
    echo  Intenta manualmente: %PYTHON_CMD% -m pip install flask flask-cors
    pause
    exit /b 1
)
echo  [OK] Flask listo.
echo.

:: --- Instalar dependencias de audio (no-criticas) ---
echo  [2/2] Verificando dependencias de audio...
%PYTHON_CMD% -m pip install librosa numpy scikit-learn matplotlib seaborn soundfile --quiet --disable-pip-version-check 2>nul
echo  [OK] Dependencias verificadas.
echo.

:: --- Iniciar servidor ---
echo  ====================================================
echo   Abre tu navegador en:  http://localhost:5000
echo  ====================================================
echo.
echo  Presiona Ctrl+C para detener el servidor.
echo.

%PYTHON_CMD% server.py

echo.
pause
