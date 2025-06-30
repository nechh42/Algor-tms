from typing import List, Dict
import pandas as pd
from .base_strategy import BaseStrategy
from .trend_following import TrendFollowingStrategy
from .grid_strategy import GridStrategy

class AdaptiveStrategy(BaseStrategy):
    def __init__(self):
        super().__init__("AdaptiveStrategy")
        self.strategies = {
            'trend': TrendFollowingStrategy(),
            'grid': GridStrategy()
        }
        
    def select_best_strategy(self, market_data: pd.DataFrame) -> BaseStrategy:
        """Select the best strategy based on market conditions"""
        volatility = market_data['close'].pct_change().std()
        trend_strength = self._calculate_trend_strength(market_data)
        
        if trend_strength > 0.7:  # Strong trend
            return self.strategies['trend']
        elif volatility < 0.02:  # Low volatility
            return self.strategies['grid']
        else:
            return self.strategies['trend']  # Default to trend following
            
    def generate_signal(self, market_data: pd.DataFrame) -> Dict:
        best_strategy = self.select_best_strategy(market_data)
        return best_strategy.generate_signal(market_data)
        
    def _calculate_trend_strength(self, market_data: pd.DataFrame) -> float:
        """Calculate trend strength using ADX-like method"""
        return abs(market_data['close'].pct_change().mean()) / market_data['close'].pct_change().std() 