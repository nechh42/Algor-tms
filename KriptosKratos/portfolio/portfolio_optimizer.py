"""
Portföy optimizasyonu ve çeşitlendirme modülü
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from config.trading_config import RISK_METRICS

class PortfolioOptimizer:
    def __init__(self):
        self.portfolio_weights = {}
        self.asset_correlations = {}
        self.volatilities = {}
        
    def optimize_portfolio(self, assets_data):
        """Modern Portföy Teorisi ile optimal ağırlıkları hesapla"""
        returns = pd.DataFrame(assets_data).pct_change()
        mean_returns = returns.mean()
        cov_matrix = returns.cov()
        
        # Portföy optimizasyonu
        num_assets = len(assets_data.columns)
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # Ağırlıklar toplamı 1
        ]
        bounds = tuple((0, 0.3) for asset in range(num_assets))  # Maksimum %30 tek varlık
        
        def portfolio_volatility(weights):
            return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            
        def sharpe_ratio(weights):
            ret = np.sum(mean_returns * weights)
            vol = portfolio_volatility(weights)
            return -(ret - 0.02) / vol  # Minimize negatif Sharpe
            
        result = minimize(sharpe_ratio,
                         num_assets * [1./num_assets],
                         method='SLSQP',
                         bounds=bounds,
                         constraints=constraints)
                         
        self.portfolio_weights = dict(zip(assets_data.columns, result.x))
        return self.portfolio_weights
        
    def calculate_diversification_metrics(self, assets_data):
        """Çeşitlendirme metriklerini hesapla"""
        # Korelasyon matrisi
        returns = pd.DataFrame(assets_data).pct_change()
        correlation_matrix = returns.corr()
        
        # Ortalama korelasyon
        correlations = correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)]
        avg_correlation = correlations.mean()
        
        # Efektif N (1 / HHI)
        weights = np.array(list(self.portfolio_weights.values()))
        hhi = np.sum(weights ** 2)
        effective_n = 1 / hhi
        
        return {
            'average_correlation': avg_correlation,
            'effective_n': effective_n,
            'max_correlation': correlations.max(),
            'diversification_ratio': 1 - avg_correlation
        }
        
    def check_sector_exposure(self, assets_sectors):
        """Sektör bazında risk kontrolü"""
        sector_exposure = {}
        for asset, weight in self.portfolio_weights.items():
            sector = assets_sectors[asset]
            sector_exposure[sector] = sector_exposure.get(sector, 0) + weight
            
        # Maksimum sektör riski %40
        return all(exposure <= 0.4 for exposure in sector_exposure.values())
        
    def calculate_risk_contribution(self, assets_data):
        """Her varlığın risk katkısını hesapla"""
        returns = pd.DataFrame(assets_data).pct_change()
        cov_matrix = returns.cov()
        weights = np.array(list(self.portfolio_weights.values()))
        
        portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        marginal_contrib = np.dot(cov_matrix, weights)
        risk_contrib = np.multiply(marginal_contrib, weights) / portfolio_vol
        
        return dict(zip(assets_data.columns, risk_contrib))
        
    def rebalance_signals(self, current_weights):
        """Rebalancing sinyalleri üret"""
        signals = {}
        for asset, target_weight in self.portfolio_weights.items():
            current = current_weights.get(asset, 0)
            diff = target_weight - current
            
            # %5'ten fazla sapma varsa rebalancing gerekli
            if abs(diff) > 0.05:
                signals[asset] = {
                    'action': 'buy' if diff > 0 else 'sell',
                    'target_weight': target_weight,
                    'current_weight': current,
                    'weight_difference': diff
                }
                
        return signals
        
    def get_portfolio_stats(self, assets_data):
        """Portföy istatistiklerini hesapla"""
        returns = pd.DataFrame(assets_data).pct_change()
        weights = np.array(list(self.portfolio_weights.values()))
        
        portfolio_returns = returns.dot(weights)
        stats = {
            'expected_return': portfolio_returns.mean() * 252,
            'volatility': portfolio_returns.std() * np.sqrt(252),
            'sharpe_ratio': (portfolio_returns.mean() - 0.02/252) / portfolio_returns.std() * np.sqrt(252),
            'max_drawdown': (portfolio_returns.cumsum() - portfolio_returns.cumsum().cummax()).min(),
            'var_95': portfolio_returns.quantile(0.05),
            'skewness': portfolio_returns.skew(),
            'kurtosis': portfolio_returns.kurtosis()
        }
        
        return stats
