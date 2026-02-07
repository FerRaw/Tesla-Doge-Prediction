#!/bin/bash

# Script para ejecutar la API FastAPI en Linux/Mac

echo "========================================"
echo "TFM - Market Prediction API"
echo "========================================"
echo ""

# Verificar si existe el entorno virtual
if [ ! -d "venv" ]; then
    echo "Creando entorno virtual..."
    python3 -m venv venv
    source venv/bin/activate
    echo "Instalando dependencias..."
    pip install -r requirements.txt
else
    echo "Activando entorno virtual..."
    source venv/bin/activate
fi

echo ""
echo "Iniciando API en http://localhost:8000"
echo "Documentación en http://localhost:8000/docs"
echo ""

# Ejecutar la API
cd src/api
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000