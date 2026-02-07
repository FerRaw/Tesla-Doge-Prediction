@echo off
REM Script para ejecutar la API FastAPI en Windows

echo ========================================
echo TFM - Market Prediction API
echo ========================================
echo.

REM Verificar si existe el entorno virtual
if not exist "venv\Scripts\activate.bat" (
    echo Creando entorno virtual...
    python -m venv venv
    call venv\Scripts\activate.bat
    echo Instalando dependencias...
    pip install -r requirements.txt
) else (
    echo Activando entorno virtual...
    call venv\Scripts\activate.bat
)

echo.
echo Iniciando API en http://localhost:8000
echo Documentacion en http://localhost:8000/docs
echo.

REM Ejecutar la API
cd src\api
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000

pause