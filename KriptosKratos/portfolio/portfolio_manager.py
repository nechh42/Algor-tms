import numpy as np
import pandas as pd
from scipy.optimize import minimize

class PortfolioManager:
    def __init__(self, risk_free_rate=0.01):
        self.risk_free_rate = risk_free_rate
        
    def calculate_metrics(self, returns):
        """Portföy metriklerini hesapla"""
        metrics = {
            'return': returns.mean() * 252,  # Yıllık getiri
            'volatility': returns.std() * np.sqrt(252),  # Yıllık volatilite
            'sharpe': (returns.mean() - self.risk_free_rate/252) / (returns.std()) * np.sqrt(252),
            'sortino': (returns.mean() - self.risk_free_rate/252) / (returns[returns < 0].std()) * np.sqrt(252),
            'max_drawdown': (returns.cumsum() - returns.cumsum().cummax()).min(),
            'var_95': returns.quantile(0.05),
            'var_99': returns.quantile(0.01)
        }
        return metrics
        
    def optimize_portfolio(self, returns_df, target_return=None):
        """Modern Portföy Teorisi ile portföy optimizasyonu"""
        n_assets = len(returns_df.columns)
        
        # Ortalama getiriler ve kovaryans matrisi
        mu = returns_df.mean() * 252
        Sigma = returns_df.cov() * 252
        
        def portfolio_volatility(weights):
            return np.sqrt(np.dot(weights.T, np.dot(Sigma, weights)))
            
        def portfolio_return(weights):
            return np.sum(mu * weights)
            
        # Kısıtlamalar
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # Ağırlıklar toplamı 1
        ]
        if target_return is not None:
            constraints.append(
                {'type': 'eq', 'fun': lambda x: portfolio_return(x) - target_return}
            )
            
        bounds = tuple((0, 1) for _ in range(n_assets))
        
        # Minimum volatilite için optimize et
        result = minimize(
            portfolio_volatility,
            n_assets * [1./n_assets],
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        return {
            'weights': result.x,
            'expected_return': portfolio_return(result.x),
            'volatility': portfolio_volatility(result.x),
            'sharpe': (portfolio_return(result.x) - self.risk_free_rate) / portfolio_volatility(result.x)
        }
        
    def position_sizing(self, capital, risk_per_trade, stop_loss_percent):
        """Pozisyon büyüklüğü hesaplama"""
        risk_amount = capital * (risk_per_trade / 100)
        position_size = risk_amount / (stop_loss_percent / 100)
        return position_size
        
    def dynamic_stop_loss(self, df, atr_multiple=2):
        """ATR bazlı dinamik stop-loss hesaplama"""
        df['atr'] = df.ta.atr(length=14)
        stop_loss = df['close'].iloc[-1] - (df['atr'].iloc[-1] * atr_multiple)
        return stop_loss
        
    def trailing_stop(self, df, atr_multiple=2):
        """Trailing stop hesaplama"""
        df['atr'] = df.ta.atr(length=14)
        df['trailing_stop'] = df['close'] - (df['atr'] * atr_multiple)
        df['trailing_stop'] = df['trailing_stop'].shift(1)
        df['trailing_stop'] = df['trailing_stop'].fillna(method='ffill')
        
        # Trailing stop'u yükselt
        mask = df['trailing_stop'] < df['trailing_stop'].shift(1)
        df.loc[mask, 'trailing_stop'] = df['trailing_stop'].shift(1)
        
        return df['trailing_stop'].iloc[-1]
