from .base_strategy import BaseStrategy
import pandas as pd
import numpy as np
from typing import Dict

class MeanReversionStrategy(BaseStrategy):
    def __init__(self):
        super().__init__("MeanReversion")
        
    def generate_signal(self, market_data: pd.DataFrame) -> Dict:
        """Generate signals based on mean reversion strategy"""
        last_row = market_data.iloc[-1]
        
        # Check if price is significantly below moving average
        if last_row['close'] < last_row['bb_lower']:
            return {
                'type': 'BUY',
                'price': last_row['close'],
                'confidence': 0.8,
                'reason': 'price_below_lower_band'
            }
            
        # Check if price is significantly above moving average
        elif last_row['close'] > last_row['bb_upper']:
            return {
                'type': 'SELL',
                'price': last_row['close'],
                'confidence': 0.8,
                'reason': 'price_above_upper_band'
            }
            
        return {
            'type': 'HOLD',
            'price': last_row['close'],
            'confidence': 0.5
        }

    def is_valid_for_market(self, market_data: pd.DataFrame) -> bool:
        """Check if mean reversion strategy is valid"""
        volatility = market_data['close'].pct_change().std()
        return volatility > 0.02  # Works better in volatile markets 