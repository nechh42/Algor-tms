from .base_strategy import BaseStrategy
import pandas as pd
import numpy as np

class TrendFollowingStrategy(BaseStrategy):
    def __init__(self):
        super().__init__("TrendFollowing")
        
    def generate_signal(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Generate signals based on trend following indicators"""
        last_row = market_data.iloc[-1]
        
        # Check MACD crossover
        macd_signal = last_row['macd'] > last_row['macd_signal']
        
        # Check RSI conditions
        rsi_signal = last_row['rsi'] > 50
        
        # Combined signal
        if macd_signal and rsi_signal:
            return {
                'type': 'BUY',
                'confidence': 0.8,
                'price': last_row['close'],
                'timestamp': last_row.name
            }
        
        return {
            'type': 'HOLD',
            'confidence': 0.5,
            'price': last_row['close'],
            'timestamp': last_row.name
        } 