from abc import ABC, abstractmethod
from typing import Dict, Any
import pandas as pd

class BaseStrategy(ABC):
    def __init__(self, name: str):
        self.name = name
        self.performance_score = 0.0
        
    @abstractmethod
    def generate_signal(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Generate trading signal based on market data"""
        pass
        
    @abstractmethod
    def is_valid_for_market(self, market_data: pd.DataFrame) -> bool:
        """Check if strategy is valid for current market conditions"""
        pass
        
    def update_performance(self, score: float):
        """Update strategy performance score"""
        self.performance_score = 0.9 * self.performance_score + 0.1 * score 