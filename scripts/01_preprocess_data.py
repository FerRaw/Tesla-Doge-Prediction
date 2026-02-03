"""
Script Principal de Preprocesamiento
Ejecuta todo el pipeline de transformación de datos raw → dataset final

Uso:
    python scripts/01_preprocess_data.py [--force-download]
"""

import argparse
import sys
from pathlib import Path

# Añadir src al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import settings
from src.data.loaders import load_all_raw_data
from src.data.preprocessor import preprocess_all_data
from src.sentiment.keywords import KeywordExtractor
from src.sentiment.analyzer import SentimentAnalyzer
from src.data.features import FeatureEngineer
from statsmodels.tsa.stattools import grangercausalitytests
import pandas as pd


def parse_args():
    """Parsea argumentos de línea de comandos"""
    parser = argparse.ArgumentParser(
        description="Preprocesa datos para TFM"
    )
    
    parser.add_argument(
        '--force-download',
        action='store_true',
        help='Fuerza descarga de datos desde APIs'
    )
    
    parser.add_argument(
        '--skip-granger',
        action='store_true',
        help='Omite test de causalidad de Granger'
    )
    
    return parser.parse_args()


def run_granger_causality(df_master: pd.DataFrame, maxlag: int = 12):
    """
    Ejecuta test de causalidad de Granger
    """
    print("\n" + "="*70)
    print("🔬 TEST DE CAUSALIDAD DE GRANGER")
    print("="*70)
    print("H0: Los tweets NO causan movimiento de precio")
    print("H1: Los tweets SÍ ayudan a predecir movimiento de precio")
    print("(Rechazamos H0 si p-value < 0.05)\n")
    print("-" * 70)
    
    tests = [
        ('TARGET_DOGE', 'sentiment_ensemble', 'Sentimiento → Retorno Doge'),
        ('TARGET_DOGE', 'relevance_score', 'Relevancia → Retorno Doge'),
        ('TARGET_TSLA', 'sentiment_ensemble', 'Sentimiento → Retorno Tesla'),
        ('TARGET_TSLA', 'relevance_score', 'Relevancia → Retorno Tesla'),
        ('doge_vol_zscore', 'sentiment_ensemble', 'Sentimiento → Volatilidad Doge')
    ]
    
    results = []
    
    for target, exogenous, label in tests:
        print(f"\n📊 TEST: {label}")
        
        data = df_master[[target, exogenous]].dropna()
        
        if len(data) < maxlag * 2:
            print(f"   ⚠️ Insuficientes datos")
            continue
        
        try:
            gc_res = grangercausalitytests(data, maxlag=maxlag, verbose=False)
            
            # Encontrar mejor lag
            lags_data = []
            for lag in range(1, maxlag + 1):
                f_stat = gc_res[lag][0]['ssr_ftest'][0]
                p_value = gc_res[lag][0]['ssr_ftest'][1]
                lags_data.append({
                    'Lag': lag,
                    'p-value': p_value
                })
            
            best = min(lags_data, key=lambda x: x['p-value'])
            
            results.append({
                'Test': label,
                'Best Lag (h)': best['Lag'],
                'p-value': best['p-value'],
                'Significant': '✅ SÍ' if best['p-value'] < 0.05 else '❌ NO'
            })
            
            print(f"   Mejor lag encontrado: {best['Lag']}h")
            print(f"   p-value: {best['p-value']:.4f}")
            print(f"   {'✅ EVIDENCIA SIGNIFICATIVA' if best['p-value'] < 0.05 else '❌ No significativo'}")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    # Resumen
    if results:
        print("\n" + "="*70)
        print("📋 RESUMEN DE CAUSALIDAD")
        print("="*70)
        df_results = pd.DataFrame(results)
        df_results['p-value'] = df_results['p-value'].apply(lambda x: f"{x:.4f}")
        print(df_results.to_string(index=False))
        print("="*70 + "\n")
        
        # Guardar resultados
        results_path = settings.DATA_PROCESSED_DIR / "granger_results.csv"
        df_results.to_csv(results_path, index=False)
        print(f"💾 Resultados guardados en: {results_path}")
    
    return results


def main():
    """Pipeline completo de preprocesamiento"""
    args = parse_args()
    
    print("="*70)
    print("🎓 TFM - PIPELINE DE PREPROCESAMIENTO")
    print("="*70)
    print(f"Fecha inicio: {settings.START_DATE}")
    print(f"Fecha fin: {settings.END_DATE}")
    print("="*70)
    
    try:
        # ================================================================
        # PASO 1: Cargar datos raw
        # ================================================================
        df_posts, df_quotes, df_doge, df_tesla = load_all_raw_data(
            force_download_crypto=args.force_download,
            force_download_stocks=args.force_download,
            databento_api_key=settings.DATABENTO_API_KEY
        )
        
        # ================================================================
        # PASO 2: Preprocesar
        # ================================================================
        df_tweets, df_doge_prep, df_tesla_prep = preprocess_all_data(
            df_posts, df_quotes, df_doge, df_tesla
        )
        
        # Guardar tweets procesados
        tweets_path = settings.DATA_PROCESSED_DIR / settings.PROCESSED_TWEETS_FILE
        df_tweets.to_parquet(tweets_path)
        print(f"\n💾 Tweets guardados: {tweets_path}")
        
        # ================================================================
        # PASO 3: Análisis de sentimiento
        # ================================================================
        print("\n" + "="*70)
        print("🤖 ANÁLISIS DE SENTIMIENTO")
        print("="*70)
        
        # Identificar keywords
        kw_extractor = KeywordExtractor()
        keywords = kw_extractor.identify_keywords(
            df_tweets,
            tesla_threshold=settings.TESLA_THRESHOLD,
            doge_threshold=settings.DOGE_THRESHOLD,
            sentiment_threshold=settings.SENTIMENT_THRESHOLD
        )
        
        # Analizar sentimiento
        sentiment_analyzer = SentimentAnalyzer()
        df_sentiment = sentiment_analyzer.analyze(df_tweets, keywords)
        
        print(f"\n✅ Análisis completado: {len(df_sentiment):,} tweets")
        
        # ================================================================
        # PASO 4: Feature engineering
        # ================================================================
        print("\n" + "="*70)
        print("🛠️ FEATURE ENGINEERING")
        print("="*70)
        
        # Features de mercado
        df_market = FeatureEngineer.create_market_features(df_doge_prep, df_tesla_prep)
        
        # Guardar
        market_path = settings.DATA_PROCESSED_DIR / settings.PROCESSED_MARKET_FILE
        df_market.to_parquet(market_path)
        print(f"💾 Features de mercado guardadas: {market_path}")
        
        # Dataset maestro
        df_master = FeatureEngineer.create_master_dataset(df_market, df_sentiment)
        
        # Guardar dataset final
        master_path = settings.DATA_PROCESSED_DIR / settings.FINAL_DATASET_FILE
        df_master.to_parquet(master_path)
        print(f"💾 Dataset maestro guardado: {master_path}")
        
        # ================================================================
        # PASO 5: Test de Granger (opcional)
        # ================================================================
        if not args.skip_granger:
            granger_results = run_granger_causality(df_master, maxlag=settings.MAX_GRANGER_LAG)
        
        # ================================================================
        # RESUMEN FINAL
        # ================================================================
        print("\n" + "="*70)
        print("✅ PREPROCESAMIENTO COMPLETADO EXITOSAMENTE")
        print("="*70)
        print(f"\n📊 RESUMEN:")
        print(f"   Tweets procesados: {len(df_sentiment):,}")
        print(f"   Horas de mercado: {len(df_market):,}")
        print(f"   Dataset final: {df_master.shape}")
        print(f"   Columnas: {len(df_master.columns)}")
        
        print(f"\n📁 ARCHIVOS GENERADOS:")
        print(f"   {tweets_path}")
        print(f"   {market_path}")
        print(f"   {master_path}")
        
        print(f"\n🚀 PRÓXIMO PASO:")
        print(f"   python scripts/02_train_models.py")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ ERROR EN PREPROCESAMIENTO: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())