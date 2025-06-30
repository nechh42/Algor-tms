from .base_strategy import BaseStrategy
import pandas as pd
import numpy as np
from typing import Dict, List

class GridStrategy(BaseStrategy):
    def __init__(self, grid_levels: int = 10, grid_size: float = 0.01):
        super().__init__("GridTrading")
        self.grid_levels = grid_levels
        self.grid_size = grid_size
        self.grids: List[Dict] = []
        
    def setup_grids(self, current_price: float):
        """Setup grid levels around current price"""
        self.grids = []
        for i in range(-self.grid_levels, self.grid_levels + 1):
            level_price = current_price * (1 + i * self.grid_size)
            self.grids.append({
                'price': level_price,
                'type': 'BUY' if i < 0 else 'SELL',
                'status': 'ACTIVE'
            })
            
    def generate_signal(self, market_data: pd.DataFrame) -> Dict:
        """Generate trading signals based on price crossing grid levels"""
        current_price = market_data['close'].iloc[-1]
        
        # Initialize grids if not set
        if not self.grids:
            self.setup_grids(current_price)
            
        # Check for triggered grid levels
        for grid in self.grids:
            if grid['status'] == 'ACTIVE':
                if grid['type'] == 'BUY' and current_price <= grid['price']:
                    return {
                        'type': 'BUY',
                        'price': grid['price'],
                        'confidence': 0.8,
                        'grid_level': grid['price']
                    }
                elif grid['type'] == 'SELL' and current_price >= grid['price']:
                    return {
                        'type': 'SELL',
                        'price': grid['price'],
                        'confidence': 0.8,
                        'grid_level': grid['price']
                    }
                    
        return {
            'type': 'HOLD',
            'price': current_price,
            'confidence': 0.5
        }

    def is_valid_for_market(self, market_data: pd.DataFrame) -> bool:
        """Check if grid strategy is valid for current market conditions"""
        # Grid trading works best in ranging markets
        volatility = market_data['close'].pct_change().std()
        return volatility < 0.02  # Less than 2% volatility 