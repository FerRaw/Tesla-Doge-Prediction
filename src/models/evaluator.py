"""
Evaluación de modelos con métricas financieras y backtesting
"""
import numpy as np
import pandas as pd
from typing import Dict, List
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class FinancialMetrics:
    """Métricas financieras para trading"""
    
    @staticmethod
    def sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
        """Sharpe Ratio"""
        excess_returns = returns - risk_free_rate
        if len(returns) < 2:
            return 0.0
        return np.mean(excess_returns) / (np.std(excess_returns) + 1e-10)
    
    @staticmethod
    def sortino_ratio(returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
        """Sortino Ratio (solo downside risk)"""
        excess_returns = returns - risk_free_rate
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) < 2:
            return 0.0
        
        downside_std = np.std(downside_returns)
        return np.mean(excess_returns) / (downside_std + 1e-10)
    
    @staticmethod
    def max_drawdown(returns: np.ndarray) -> float:
        """Maximum Drawdown"""
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        return np.min(drawdown)
    
    @staticmethod
    def win_rate(returns: np.ndarray) -> float:
        """Porcentaje de operaciones ganadoras"""
        if len(returns) == 0:
            return 0.0
        return np.sum(returns > 0) / len(returns)
    
    @staticmethod
    def directional_accuracy(predictions: np.ndarray, actuals: np.ndarray) -> float:
        """
        Directional Accuracy: % de veces que predice correctamente la dirección
        MUY IMPORTANTE en trading
        """
        pred_direction = np.sign(predictions)
        actual_direction = np.sign(actuals)
        return np.mean(pred_direction == actual_direction)


class ModelEvaluator:
    """Evaluador de modelos con métricas estadísticas y financieras"""
    
    def __init__(self):
        self.results = {}
    
    def evaluate_regression(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str = "Model"
    ) -> Dict:
        """Evalúa modelo de regresión"""
        
        # Métricas estadísticas
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # Métricas financieras
        directional_acc = FinancialMetrics.directional_accuracy(y_pred, y_true)
        
        results = {
            'model_name': model_name,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'directional_accuracy': directional_acc
        }
        
        self.results[model_name] = results
        return results
    
    def print_evaluation(self, model_name: str):
        """Imprime resultados de evaluación"""
        if model_name not in self.results:
            print(f"⚠️ No hay resultados para {model_name}")
            return
        
        res = self.results[model_name]
        
        print(f"\n📊 EVALUACIÓN: {model_name}")
        print("="*60)
        print(f"   RMSE: {res['rmse']:.6f}")
        print(f"   MAE: {res['mae']:.6f}")
        print(f"   R²: {res['r2']:.4f}")
        print(f"   Directional Accuracy: {res['directional_accuracy']:.2%}")
        print("="*60)
    
    def compare_models(self) -> pd.DataFrame:
        """Compara todos los modelos evaluados"""
        if not self.results:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.results).T
        df = df.sort_values('rmse', ascending=True)
        return df


class TradingSimulator:
    """Simula estrategia de trading basada en predicciones"""
    
    def __init__(
        self,
        initial_capital: float = 10000,
        transaction_cost: float = 0.001,
        position_size: float = 1.0
    ):
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.position_size = position_size
    
    def simulate_strategy(
        self,
        predictions: np.ndarray,
        actual_returns: np.ndarray,
        threshold: float = 0.01
    ) -> Dict:
        """
        Simula estrategia de trading simple:
        - BUY si predicción > threshold
        - SELL si predicción < -threshold
        - HOLD si abs(predicción) < threshold
        """
        capital = self.initial_capital
        position = 0  # 1=long, -1=short, 0=neutral
        
        portfolio_values = [capital]
        returns = []
        trades = []
        
        for i, (pred, actual) in enumerate(zip(predictions, actual_returns)):
            # Decisión
            if pred > threshold and position != 1:
                # Comprar
                if position == -1:
                    capital *= (1 - actual) * (1 - self.transaction_cost)
                capital *= (1 - self.transaction_cost)
                position = 1
                trades.append({'timestamp': i, 'action': 'BUY'})
                
            elif pred < -threshold and position != -1:
                # Vender
                if position == 1:
                    capital *= (1 + actual) * (1 - self.transaction_cost)
                capital *= (1 - self.transaction_cost)
                position = -1
                trades.append({'timestamp': i, 'action': 'SELL'})
            
            # Actualizar portfolio
            if position == 1:
                capital *= (1 + actual)
                returns.append(actual)
            elif position == -1:
                capital *= (1 - actual)
                returns.append(-actual)
            else:
                returns.append(0)
            
            portfolio_values.append(capital)
        
        # Métricas
        total_return = (capital - self.initial_capital) / self.initial_capital
        returns_array = np.array(returns)
        
        metrics = {
            'total_return': total_return,
            'final_capital': capital,
            'num_trades': len(trades),
            'sharpe_ratio': FinancialMetrics.sharpe_ratio(returns_array),
            'sortino_ratio': FinancialMetrics.sortino_ratio(returns_array),
            'max_drawdown': FinancialMetrics.max_drawdown(returns_array),
            'win_rate': FinancialMetrics.win_rate(returns_array),
            'portfolio_values': portfolio_values,
            'trades': trades
        }
        
        return metrics
    
    def print_trading_results(self, metrics: Dict):
        """Imprime resultados de trading"""
        print("\n💰 RESULTADOS DE TRADING")
        print("="*60)
        print(f"   Total Return: {metrics['total_return']:.2%}")
        print(f"   Final Capital: ${metrics['final_capital']:.2f}")
        print(f"   Number of Trades: {metrics['num_trades']}")
        print(f"   Win Rate: {metrics['win_rate']:.2%}")
        print(f"   Sharpe Ratio: {metrics['sharpe_ratio']:.4f}")
        print(f"   Sortino Ratio: {metrics['sortino_ratio']:.4f}")
        print(f"   Max Drawdown: {metrics['max_drawdown']:.2%}")
        print("="*60)