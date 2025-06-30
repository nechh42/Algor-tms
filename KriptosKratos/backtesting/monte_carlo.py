import numpy as np
import pandas as pd
from scipy.stats import norm

class MonteCarloSimulator:
    def __init__(self, n_simulations=1000, confidence_level=0.95):
        self.n_simulations = n_simulations
        self.confidence_level = confidence_level
        
    def simulate_returns(self, returns, forecast_period=30):
        """Monte Carlo simülasyonu ile getiri tahminleri"""
        # Getiri istatistikleri
        mu = returns.mean()
        sigma = returns.std()
        
        # Simülasyonları gerçekleştir
        simulations = np.zeros((self.n_simulations, forecast_period))
        for i in range(self.n_simulations):
            # Rastgele getiriler üret
            rand_returns = np.random.normal(mu, sigma, forecast_period)
            # Kümülatif getiriyi hesapla
            simulations[i] = (1 + rand_returns).cumprod()
            
        # Güven aralıklarını hesapla
        percentiles = np.percentile(simulations, 
                                  [(1-self.confidence_level)*100/2, 
                                   50, 
                                   (1+self.confidence_level)*100/2], 
                                  axis=0)
                                   
        return {
            'lower_bound': percentiles[0],
            'median': percentiles[1],
            'upper_bound': percentiles[2],
            'simulations': simulations
        }
        
    def calculate_var(self, returns, confidence_level=0.95):
        """Value at Risk hesaplama"""
        # Parametrik VaR
        mu = returns.mean()
        sigma = returns.std()
        var = norm.ppf(1-confidence_level, mu, sigma)
        
        return {
            'VaR': -var,
            'CVaR': -returns[returns <= var].mean()
        }
        
    def stress_test(self, strategy, historical_data, scenarios):
        """Strateji stres testi"""
        results = []
        
        for scenario in scenarios:
            # Senaryo verilerini hazırla
            scenario_data = historical_data.copy()
            
            # Senaryo şoklarını uygula
            if 'price_shock' in scenario:
                scenario_data['close'] *= (1 + scenario['price_shock'])
            if 'volatility_shock' in scenario:
                scenario_data['high'] = scenario_data['close'] * (1 + scenario['volatility_shock'])
                scenario_data['low'] = scenario_data['close'] * (1 - scenario['volatility_shock'])
                
            # Stratejiyi test et
            strategy_result = strategy.backtest(scenario_data)
            
            results.append({
                'scenario_name': scenario.get('name', 'Unknown'),
                'return': strategy_result['return'],
                'max_drawdown': strategy_result['max_drawdown'],
                'sharpe': strategy_result['sharpe']
            })
            
        return pd.DataFrame(results)
        
    def optimize_parameters(self, strategy, historical_data, param_grid):
        """Strateji parametrelerini optimize et"""
        results = []
        
        # Grid search
        for params in self._generate_param_combinations(param_grid):
            # Parametreleri ayarla
            strategy.set_parameters(params)
            
            # Backtest yap
            backtest_result = strategy.backtest(historical_data)
            
            results.append({
                'parameters': params,
                'return': backtest_result['return'],
                'sharpe': backtest_result['sharpe'],
                'max_drawdown': backtest_result['max_drawdown']
            })
            
        return pd.DataFrame(results)
        
    def _generate_param_combinations(self, param_grid):
        """Parameter grid'inden tüm kombinasyonları üret"""
        keys = param_grid.keys()
        values = param_grid.values()
        
        for instance in itertools.product(*values):
            yield dict(zip(keys, instance))
