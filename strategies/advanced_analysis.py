import pandas as pd
import numpy as np
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class AdvancedAnalysis:
    def __init__(self):
        logger.info("Advanced analysis initialized")
    
    def fibonacci_levels(self, high, low):
        """Fibonacci seviyeleri hesapla"""
        diff = high - low
        levels = {
            '0.0': low,
            '0.236': low + 0.236 * diff,
            '0.382': low + 0.382 * diff,
            '0.5': low + 0.5 * diff,
            '0.618': low + 0.618 * diff,
            '0.786': low + 0.786 * diff,
            '1.0': high
        }
        return levels
    
    def volume_profile(self, df, n_bins=10):
        """Hacim profili analizi"""
        price_bins = pd.qcut(df['close'], n_bins)
        volume_profile = df.groupby(price_bins)['volume'].sum()
        return volume_profile
    
    def pivot_points(self, high, low, close):
        """Pivot noktaları hesapla"""
        pivot = (high + low + close) / 3
        r1 = 2 * pivot - low
        r2 = pivot + (high - low)
        s1 = 2 * pivot - high
        s2 = pivot - (high - low)
        return {'pivot': pivot, 'r1': r1, 'r2': r2, 's1': s1, 's2': s2} 