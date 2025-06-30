import numpy as np
import pandas as pd
import pandas_ta as ta

class AdvancedIndicators:
    @staticmethod
    def fibonacci_levels(df):
        """Fibonacci seviyeleri hesapla"""
        high = df['high'].max()
        low = df['low'].min()
        diff = high - low
        
        levels = {
            'Level 0.0': low,
            'Level 0.236': low + 0.236 * diff,
            'Level 0.382': low + 0.382 * diff,
            'Level 0.5': low + 0.5 * diff,
            'Level 0.618': low + 0.618 * diff,
            'Level 0.786': low + 0.786 * diff,
            'Level 1.0': high
        }
        return levels
        
    @staticmethod
    def volume_profile(df, levels=30):
        """Hacim profili analizi"""
        price_range = np.linspace(df['low'].min(), df['high'].max(), levels)
        volume_profile = []
        
        for i in range(len(price_range)-1):
            mask = (df['close'] >= price_range[i]) & (df['close'] < price_range[i+1])
            volume_profile.append({
                'price_level': (price_range[i] + price_range[i+1]) / 2,
                'volume': df.loc[mask, 'volume'].sum()
            })
            
        return pd.DataFrame(volume_profile)
        
    @staticmethod
    def pivot_points(df):
        """Pivot noktaları hesapla"""
        pivot = (df['high'].iloc[-1] + df['low'].iloc[-1] + df['close'].iloc[-1]) / 3
        
        r1 = 2 * pivot - df['low'].iloc[-1]
        r2 = pivot + (df['high'].iloc[-1] - df['low'].iloc[-1])
        r3 = r1 + (df['high'].iloc[-1] - df['low'].iloc[-1])
        
        s1 = 2 * pivot - df['high'].iloc[-1]
        s2 = pivot - (df['high'].iloc[-1] - df['low'].iloc[-1])
        s3 = s1 - (df['high'].iloc[-1] - df['low'].iloc[-1])
        
        return {
            'Pivot': pivot,
            'R1': r1, 'R2': r2, 'R3': r3,
            'S1': s1, 'S2': s2, 'S3': s3
        }
        
    @staticmethod
    def elliott_wave_analysis(df):
        """Basit Elliott Wave analizi"""
        # Trend yönünü belirle
        df['trend'] = np.where(df['close'] > df['close'].shift(1), 1, -1)
        
        # Dalgaları tespit et
        waves = []
        current_wave = {'direction': df['trend'].iloc[0], 'start_idx': 0}
        
        for i in range(1, len(df)):
            if df['trend'].iloc[i] != current_wave['direction']:
                waves.append({
                    'direction': current_wave['direction'],
                    'start_idx': current_wave['start_idx'],
                    'end_idx': i-1,
                    'length': i - current_wave['start_idx']
                })
                current_wave = {'direction': df['trend'].iloc[i], 'start_idx': i}
                
        return waves
        
    @staticmethod
    def vwap_analysis(df):
        """VWAP analizi"""
        df['vwap'] = ta.vwap(df['high'], df['low'], df['close'], df['volume'])
        df['vwap_distance'] = ((df['close'] - df['vwap']) / df['vwap']) * 100
        
        return {
            'vwap': df['vwap'].iloc[-1],
            'distance_percent': df['vwap_distance'].iloc[-1],
            'above_vwap': df['close'].iloc[-1] > df['vwap'].iloc[-1]
        }
