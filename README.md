# 🚀 TFM - Market Prediction API

API RESTful para predicción de mercados (DOGE y TSLA) mediante análisis de sentimiento de Twitter.

## 📋 Características

- ✅ Predicciones en tiempo real de DOGE y TSLA
- ✅ 4 modelos de ML: XGBoost, LightGBM, CatBoost, Stacking
- ✅ Backtesting con 3 configuraciones (Conservadora, Moderada, Agresiva)
- ✅ Generación de gráficos interactivos
- ✅ Clasificación de impacto de tweets
- ✅ Documentación interactiva con Swagger

## 🛠️ Instalación

### Prerrequisitos

- Python 3.8+
- Modelos entrenados en `models/`
- Dataset procesado en `data/processed/`

### Paso 1: Clonar y navegar

```bash
cd tu_proyecto
```

### Paso 2: Instalar dependencias

```bash
pip install -r requirements.txt
```

### Paso 3: Verificar estructura

Asegúrate de tener esta estructura:

```
Proyecto/
├── src/
│   ├── api/
│   │   ├── main.py
│   │   └── schemas.py
│   ├── models/
│   │   ├── improved_predictors.py
│   │   ├── evaluator.py
│   │   └── base_predictor.py
│   └── visualization/
│       └── charts.py
├── models/
│   ├── doge_predictor_v2_improved.pkl
│   ├── tsla_predictor_v2_improved.pkl
│   ├── impact_classifier_v2_improved.pkl
│   └── backtesting_results.json
├── data/
│   └── processed/
│       └── master_dataset.parquet
├── config/
│   └── settings.py
├── requirements.txt
├── run_api.bat (Windows)
└── run_api.sh (Linux/Mac)
```

## 🚀 Ejecución

### Windows

```bash
run_api.bat
```

### Linux/Mac

```bash
chmod +x run_api.sh
./run_api.sh
```

### Manual

```bash
cd src/api
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## 📖 Uso de la API

### Acceder a la documentación

Una vez iniciada, abre tu navegador:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **Ayuda completa**: http://localhost:8000/help

### Endpoints Principales

#### 1. **Información**

```bash
# Información general
GET http://localhost:8000/

# Estado de salud
GET http://localhost:8000/health

# Ayuda completa
GET http://localhost:8000/help
```

#### 2. **Modelos**

```bash
# Información de modelos
GET http://localhost:8000/models/info

# Performance de modelos para DOGE
GET http://localhost:8000/models/performance/DOGE

# Performance de modelos para TSLA
GET http://localhost:8000/models/performance/TSLA
```

#### 3. **Predicciones**

```bash
# Última predicción de DOGE
GET http://localhost:8000/predictions/DOGE/latest?model_name=stacking

# Últimas 100 predicciones de TSLA
GET http://localhost:8000/predictions/TSLA/batch?n=100&model_name=stacking

# Predicción con modelo específico
GET http://localhost:8000/predictions/DOGE/latest?model_name=xgboost
```

#### 4. **Backtesting**

```bash
# Resultados pre-computados de DOGE
GET http://localhost:8000/backtesting/DOGE/results

# Backtesting personalizado (POST)
POST http://localhost:8000/backtesting/DOGE/custom
Content-Type: application/json

{
  "asset": "DOGE",
  "threshold": 0.0025,
  "max_position_size": 0.75,
  "transaction_cost": 0.001,
  "initial_capital": 10000
}
```

#### 5. **Gráficos**

```bash
# Predicciones vs Reales
GET http://localhost:8000/charts/predictions/DOGE

# Equity Curve (configuración moderada)
GET http://localhost:8000/charts/equity/DOGE?strategy=moderate

# Feature Importance (top 20)
GET http://localhost:8000/charts/importance/DOGE?top_n=20

# Comparación de modelos
GET http://localhost:8000/charts/comparison/TSLA
```

#### 6. **Impact Classifier**

```bash
# Clasificación de últimos 10 tweets
GET http://localhost:8000/impact/predict?n=10
```

## 📊 Ejemplos de Uso

### Python

```python
import requests

# Obtener última predicción
response = requests.get("http://localhost:8000/predictions/DOGE/latest")
data = response.json()
print(f"Predicción DOGE: {data['prediction']}")

# Obtener gráfico
response = requests.get("http://localhost:8000/charts/predictions/DOGE")
chart_data = response.json()
# chart_data['image_base64'] contiene el gráfico en base64
```

### cURL

```bash
# Última predicción
curl http://localhost:8000/predictions/DOGE/latest

# Backtesting personalizado
curl -X POST http://localhost:8000/backtesting/DOGE/custom \
  -H "Content-Type: application/json" \
  -d '{
    "asset": "DOGE",
    "threshold": 0.003,
    "max_position_size": 0.8,
    "transaction_cost": 0.001,
    "initial_capital": 10000
  }'
```

### JavaScript (Fetch)

```javascript
// Obtener predicciones
fetch('http://localhost:8000/predictions/TSLA/batch?n=50')
  .then(response => response.json())
  .then(data => {
    console.log('Predicciones:', data.predictions);
    console.log('Valores reales:', data.actuals);
  });

// Mostrar gráfico
fetch('http://localhost:8000/charts/predictions/DOGE')
  .then(response => response.json())
  .then(data => {
    const img = document.createElement('img');
    img.src = data.image_base64;
    document.body.appendChild(img);
  });
```

## 🔧 Configuración

### Variables de Entorno (Opcional)

Crea un archivo `.env` en la raíz:

```env
API_HOST=0.0.0.0
API_PORT=8000
RELOAD=True
LOG_LEVEL=info
```

### Personalizar Puerto

```bash
# En el script de ejecución o manualmente:
python -m uvicorn main:app --host 0.0.0.0 --port 8080
```

## 📈 Performance

- **Startup time**: ~5-10 segundos (carga de modelos)
- **Latencia promedio**: 
  - Predicciones: 50-100ms
  - Backtesting: 200-500ms
  - Gráficos: 500-1000ms

## 🐛 Troubleshooting

### Error: Modelos no encontrados

```bash
FileNotFoundError: models/doge_predictor_v2_improved.pkl
```

**Solución**: Entrena los modelos primero:
```bash
python scripts/02_improved_train_models.py --evaluate --backtesting
```

### Error: Dataset no encontrado

```bash
FileNotFoundError: data/processed/master_dataset.parquet
```

**Solución**: Procesa los datos primero:
```bash
python scripts/01_preprocess_data.py
```

### Puerto 8000 en uso

```bash
ERROR: [Errno 48] Address already in use
```

**Solución**: Cambia el puerto:
```bash
python -m uvicorn main:app --port 8080
```

## 📚 Documentación de Modelos

### DOGE Predictor

- **Mejor modelo**: Stacking (R² = 0.224, Dir.Acc = 66%)
- **Features**: 50+ (sentiment, market data, wavelets)
- **Lag óptimo**: 3 horas

### TSLA Predictor

- **Mejor modelo**: XGBoost (R² = 0.297, Dir.Acc = 58%)
- **Features**: 40+ (sentiment, market data, trading hours)
- **Lag óptimo**: 1 hora

### Impact Classifier

- **Accuracy**: 88%
- **Clases**: No Impact, DOGE Only, TSLA Only, Both

## 🎯 Roadmap

- [ ] Autenticación JWT
- [ ] Rate limiting
- [ ] WebSocket para predicciones en tiempo real
- [ ] Cache con Redis
- [ ] Logging estructurado
- [ ] Métricas con Prometheus
- [ ] Contenedor Docker

## 📝 Licencia

TFM - Uso académico

## 👤 Autor

Fernando - Master en Data Science

## 🤝 Contribuciones

Este es un proyecto de TFM. Para sugerencias, abre un issue.