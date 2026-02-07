"""
Módulo de Evaluación Completa de Modelos

Incluye evaluación detallada para:
- Regresión (DOGE, TSLA)
- Clasificación (Impact Classifier)
- Backtesting y trading simulation
"""

import pandas as pd
import numpy as np
from typing import Dict, List
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns


class ModelEvaluator:
    """Evaluador completo de modelos"""
    
    def __init__(self):
        self.results = {}
    
    def evaluate_regression(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str
    ) -> Dict:
        """
        Evalúa un modelo de regresión
        
        Returns métricas: RMSE, MAE, R², Directional Accuracy
        """
        # Métricas básicas
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # Directional accuracy
        direction_true = np.sign(y_true)
        direction_pred = np.sign(y_pred)
        directional_accuracy = (direction_true == direction_pred).mean()
        
        # Correlation
        correlation = np.corrcoef(y_true, y_pred)[0, 1]
        
        results = {
            'model_name': model_name,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'directional_accuracy': directional_accuracy,
            'correlation': correlation,
            'n_samples': len(y_true)
        }
        
        self.results[model_name] = results
        return results
    
    def print_regression_results(self, results: Dict):
        """Imprime resultados de regresión de forma legible"""
        print(f"\n📊 EVALUACIÓN: {results['model_name']}")
        print("="*70)
        print(f"   RMSE: {results['rmse']:.6f}")
        print(f"   MAE: {results['mae']:.6f}")
        print(f"   R²: {results['r2']:.4f}")
        print(f"   Directional Accuracy: {results['directional_accuracy']*100:.2f}%")
        print(f"   Correlation: {results['correlation']:.4f}")
        print("="*70)
    
    def evaluate_classification(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray,
        model_name: str,
        class_names: List[str] = None
    ) -> Dict:
        """
        Evalúa un clasificador
        
        Returns métricas completas de clasificación
        """
        if class_names is None:
            class_names = ['No Impact', 'DOGE Only', 'TSLA Only', 'Both']
        
        # Métricas globales
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0
        )
        
        # Métricas por clase
        precision_per_class, recall_per_class, f1_per_class, support_per_class = \
            precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Classification report
        report_dict = classification_report(
            y_true, y_pred,
            target_names=class_names,
            zero_division=0,
            output_dict=True
        )
        
        results = {
            'model_name': model_name,
            'accuracy': accuracy,
            'precision_weighted': precision,
            'recall_weighted': recall,
            'f1_weighted': f1,
            'precision_per_class': precision_per_class,
            'recall_per_class': recall_per_class,
            'f1_per_class': f1_per_class,
            'support_per_class': support_per_class,
            'confusion_matrix': cm,
            'classification_report': report_dict,
            'class_names': class_names
        }
        
        return results
    
    def print_classification_results(self, results: Dict):
        """Imprime resultados de clasificación"""
        print(f"\n🎯 EVALUACIÓN: {results['model_name']}")
        print("="*70)
        print(f"   Accuracy: {results['accuracy']*100:.2f}%")
        print(f"   Precision (weighted): {results['precision_weighted']:.4f}")
        print(f"   Recall (weighted): {results['recall_weighted']:.4f}")
        print(f"   F1-Score (weighted): {results['f1_weighted']:.4f}")
        
        print(f"\n   Métricas por Clase:")
        for i, name in enumerate(results['class_names']):
            if i < len(results['precision_per_class']):
                print(f"   {name:15s}: "
                      f"Prec={results['precision_per_class'][i]:.3f} "
                      f"Rec={results['recall_per_class'][i]:.3f} "
                      f"F1={results['f1_per_class'][i]:.3f} "
                      f"(n={results['support_per_class'][i]})")
        
        print(f"\n   Matriz de Confusión:")
        cm = results['confusion_matrix']
        print("   ", end="")
        for i, name in enumerate(results['class_names']):
            print(f"{name[:8]:>8s} ", end="")
        print()
        for i, row in enumerate(cm):
            print(f"   {results['class_names'][i][:8]:>8s} ", end="")
            for val in row:
                print(f"{val:>8d} ", end="")
            print()
        
        print(f"\n   Distribución de Clases (Ground Truth):")
        for i, count in enumerate(results['support_per_class']):
            total = sum(results['support_per_class'])
            pct = (count / total * 100) if total > 0 else 0
            print(f"   {results['class_names'][i]:15s}: {count:>5d} ({pct:>5.1f}%)")
        
        print("="*70)
    
    def plot_confusion_matrix(self, results: Dict, save_path: str = None):
        """Plotea matriz de confusión"""
        cm = results['confusion_matrix']
        class_names = results['class_names']
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names
        )
        plt.title(f'Confusion Matrix - {results["model_name"]}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.close()
    
    def compare_models(self) -> pd.DataFrame:
        """Compara todos los modelos evaluados"""
        if not self.results:
            return pd.DataFrame()
        
        comparison = []
        for name, metrics in self.results.items():
            comparison.append({
                'model': name,
                'rmse': metrics.get('rmse', np.nan),
                'mae': metrics.get('mae', np.nan),
                'r2': metrics.get('r2', np.nan),
                'directional_accuracy': metrics.get('directional_accuracy', np.nan)
            })
        
        df = pd.DataFrame(comparison)
        df = df.sort_values('rmse')
        return df
    
    def get_best_model(self, metric: str = 'rmse', minimize: bool = True) -> str:
        """Obtiene el mejor modelo según una métrica"""
        if not self.results:
            return None
        
        valid_results = {
            name: metrics 
            for name, metrics in self.results.items() 
            if metric in metrics and not np.isnan(metrics[metric])
        }
        
        if not valid_results:
            return None
        
        if minimize:
            best = min(valid_results.items(), key=lambda x: x[1][metric])
        else:
            best = max(valid_results.items(), key=lambda x: x[1][metric])
        
        return best[0]


class BacktestEvaluator:
    """Evaluador de backtesting y trading"""
    
    def __init__(self, initial_capital: float = 10000):
        self.initial_capital = initial_capital
    
    def run_backtest(
        self,
        df: pd.DataFrame,
        predictions: np.ndarray,
        actual_returns: np.ndarray,
        threshold: float = 0.0,
        max_position_size: float = 1.0,
        transaction_cost: float = 0.0005
    ) -> Dict:
        """
        Ejecuta backtesting CORREGIDO con gestión de riesgo
        
        Args:
            df: DataFrame con timestamps
            predictions: Retornos predichos
            actual_returns: Retornos reales
            threshold: Umbral mínimo de predicción para operar
            max_position_size: Fracción máxima del capital por trade (default 100%)
            transaction_cost: Costo por transacción como % (default 0.1%)
        
        Returns:
            Métricas de trading + datos para gráficos
        """
        capital = self.initial_capital
        positions = []
        pnl_history = [capital]
        equity_curve = [capital]
        
        n_trades = 0
        wins = 0
        losses = 0
        total_profit = 0
        total_loss = 0
        
        for i, (pred, actual) in enumerate(zip(predictions, actual_returns)):
            # Señal de trading
            if abs(pred) > threshold:
                # Determinar dirección: +1 (long) o -1 (short)
                direction = np.sign(pred)
                
                # Calcular retorno de la posición
                # Si pred > 0 (long) y actual > 0 (sube) → ganas
                # Si pred > 0 (long) y actual < 0 (baja) → pierdes
                # Si pred < 0 (short) y actual < 0 (baja) → ganas
                # Si pred < 0 (short) y actual > 0 (sube) → pierdes
                position_return = direction * actual
                
                # CRÍTICO: Limitar retorno máximo por trade
                # Evita que un solo trade multiplique el capital exponencialmente
                position_return = np.clip(position_return, -0.5, 0.5)  # Max ±50% por trade
                
                # Aplicar tamaño de posición
                position_return *= max_position_size
                
                # Restar costos de transacción
                position_return -= transaction_cost
                
                # Actualizar capital (capitalización simple por trade)
                trade_pnl = capital * position_return
                capital += trade_pnl
                
                # Evitar capital negativo
                if capital < 0:
                    capital = 0
                
                positions.append({
                    'index': i,
                    'prediction': pred,
                    'actual': actual,
                    'direction': direction,
                    'position_return': position_return,
                    'pnl': trade_pnl,
                    'capital': capital
                })
                
                n_trades += 1
                if trade_pnl > 0:
                    wins += 1
                    total_profit += trade_pnl
                else:
                    losses += 1
                    total_loss += abs(trade_pnl)
            
            equity_curve.append(capital)
        
        # Calcular métricas finales
        total_return = (capital - self.initial_capital) / self.initial_capital
        win_rate = wins / n_trades if n_trades > 0 else 0
        avg_win = total_profit / wins if wins > 0 else 0
        avg_loss = total_loss / losses if losses > 0 else 0
        profit_factor = total_profit / total_loss if total_loss > 0 else 0
        
        # Sharpe ratio (corregido)
        if len(equity_curve) > 1:
            returns_series = np.diff(equity_curve) / equity_curve[:-1]
            returns_series = returns_series[np.isfinite(returns_series)]  # Eliminar NaN/Inf
            
            if len(returns_series) > 0 and np.std(returns_series) > 0:
                sharpe = np.mean(returns_series) / np.std(returns_series)
                sharpe_annualized = sharpe * np.sqrt(252 * 24)  # Datos horarios
            else:
                sharpe_annualized = 0
            
            # Sortino ratio
            negative_returns = returns_series[returns_series < 0]
            if len(negative_returns) > 0:
                downside_std = np.std(negative_returns)
                sortino = np.mean(returns_series) / downside_std if downside_std > 0 else 0
                sortino_annualized = sortino * np.sqrt(252 * 24)
            else:
                sortino_annualized = 0
        else:
            sharpe_annualized = 0
            sortino_annualized = 0
        
        # Max drawdown
        cumulative = np.array(equity_curve)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        drawdown = drawdown[np.isfinite(drawdown)]  # Eliminar NaN
        max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0
        
        return {
            'initial_capital': self.initial_capital,
            'final_capital': capital,
            'total_return_pct': total_return * 100,
            'n_trades': n_trades,
            'n_wins': wins,
            'n_losses': losses,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe_annualized,
            'sortino_ratio': sortino_annualized,
            'max_drawdown_pct': max_drawdown * 100,
            'equity_curve': equity_curve,
            'positions': positions,
            'returns_series': returns_series.tolist() if 'returns_series' in locals() else []
        }
    
    def print_backtest_results(self, results: Dict):
        """Imprime resultados de backtesting mejorado"""
        print(f"\n💰 RESULTADOS DE TRADING")
        print("="*70)
        print(f"   Initial Capital: ${results['initial_capital']:,.2f}")
        print(f"   Final Capital: ${results['final_capital']:,.2f}")
        print(f"   Total Return: {results['total_return_pct']:.2f}%")
        print(f"   ")
        print(f"   Number of Trades: {results['n_trades']}")
        print(f"   Winning Trades: {results['n_wins']}")
        print(f"   Losing Trades: {results['n_losses']}")
        print(f"   Win Rate: {results['win_rate']*100:.2f}%")
        print(f"   ")
        print(f"   Average Win: ${results['avg_win']:,.2f}")
        print(f"   Average Loss: ${results['avg_loss']:,.2f}")
        print(f"   Profit Factor: {results['profit_factor']:.2f}")
        print(f"   ")
        print(f"   Sharpe Ratio: {results['sharpe_ratio']:.4f}")
        print(f"   Sortino Ratio: {results['sortino_ratio']:.4f}")
        print(f"   Max Drawdown: {results['max_drawdown_pct']:.2f}%")
        print("="*70)


def evaluate_model_complete(
    model,
    test_df: pd.DataFrame,
    asset_name: str,
    evaluator: ModelEvaluator
):
    """
    Evaluación completa de un modelo de regresión
    
    Args:
        model: Modelo entrenado (FinalDOGEPredictor o FinalTSLAPredictor)
        test_df: DataFrame de test
        asset_name: 'DOGE' o 'TSLA'
        evaluator: Instancia de ModelEvaluator
    """
    print(f"\n{'='*70}")
    print(f"📊 EVALUANDO {asset_name}")
    print(f"{'='*70}")
    
    target_col = f'TARGET_{asset_name}'
    
    # Evaluar cada modelo
    for model_name in ['xgboost', 'lightgbm', 'catboost', 'stacking']:
        if model_name in model.models:
            try:
                # Predicciones
                predictions = model.predict(test_df, model_name=model_name)
                y_true = test_df[target_col].values
                
                # Ajustar longitudes si es necesario
                min_len = min(len(predictions), len(y_true))
                predictions = predictions[-min_len:]
                y_true = y_true[-min_len:]
                
                # Evaluar
                full_model_name = f"{asset_name}_{model_name}"
                results = evaluator.evaluate_regression(
                    y_true, predictions, full_model_name
                )
                evaluator.print_regression_results(results)
                
            except Exception as e:
                print(f"   ⚠️  Error evaluando {model_name}: {e}")


def evaluate_impact_classifier_complete(
    model,
    test_df: pd.DataFrame,
    evaluator: ModelEvaluator
):
    """
    Evaluación completa del Impact Classifier
    
    Args:
        model: FinalImpactClassifier entrenado
        test_df: DataFrame de test
        evaluator: Instancia de ModelEvaluator
    """
    if model is None:
        print(f"\n⚠️ No hay modelo Impact Classifier para evaluar")
        return
    
    print(f"\n{'='*70}")
    print(f"🎯 EVALUANDO IMPACT CLASSIFIER")
    print(f"{'='*70}")
    
    try:
        # Evaluar cada modelo
        for model_name in ['random_forest', 'xgboost', 'lightgbm']:
            if model_name in model.models:
                metrics = model.get_classification_metrics(test_df, model_name=model_name)
                
                # Extraer métricas por clase correctamente
                precision_per_class = []
                recall_per_class = []
                f1_per_class = []
                support_per_class = []
                
                for i in range(4):  # 4 clases
                    class_key = str(i)
                    if class_key in metrics['classification_report']:
                        precision_per_class.append(metrics['classification_report'][class_key]['precision'])
                        recall_per_class.append(metrics['classification_report'][class_key]['recall'])
                        f1_per_class.append(metrics['classification_report'][class_key]['f1-score'])
                        support_per_class.append(metrics['classification_report'][class_key]['support'])
                    else:
                        precision_per_class.append(0.0)
                        recall_per_class.append(0.0)
                        f1_per_class.append(0.0)
                        support_per_class.append(0)
                
                # Crear results para el evaluator
                evaluator_results = {
                    'model_name': f"Impact_{model_name}",
                    'accuracy': metrics['accuracy'],
                    'precision_weighted': metrics['precision'],
                    'recall_weighted': metrics['recall'],
                    'f1_weighted': metrics['f1_score'],
                    'precision_per_class': precision_per_class,
                    'recall_per_class': recall_per_class,
                    'f1_per_class': f1_per_class,
                    'support_per_class': support_per_class,
                    'confusion_matrix': metrics['confusion_matrix'],
                    'class_names': metrics['class_names']
                }
                
                evaluator.print_classification_results(evaluator_results)
                
    except Exception as e:
        print(f"❌ Error evaluando Impact Classifier: {e}")
        import traceback
        traceback.print_exc()