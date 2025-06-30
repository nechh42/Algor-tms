import pandas as pd
import numpy as np
import pandas_ta as ta
from src.utils.logger import setup_logger
from src.strategies.advanced_analysis import AdvancedAnalysis

logger = setup_logger(__name__)

class TechnicalAnalyzer:
    def __init__(self):
        self.advanced = AdvancedAnalysis()
        logger.info("Technical analyzer initialized")
    
    def add_indicators(self, df):
        """Teknik indikatörleri hesapla ve DataFrame'e ekle"""
        try:
            # RSI
            df['rsi'] = ta.rsi(df['close'], length=14)
            
            # MACD
            macd = ta.macd(df['close'])
            df['macd'] = macd['MACD_12_26_9']
            df['macd_signal'] = macd['MACDs_12_26_9']
            
            # Bollinger Bands - Düzeltilmiş versiyon
            bb = ta.bbands(df['close'], length=20)
            df['bb_upper'] = bb['BBU_20_2.0']  # Üst bant
            df['bb_middle'] = bb['BBM_20_2.0']  # Orta bant
            df['bb_lower'] = bb['BBL_20_2.0']   # Alt bant
            
            # Moving Averages
            df['sma_20'] = ta.sma(df['close'], length=20)
            df['ema_20'] = ta.ema(df['close'], length=20)
            
            # Volume indicators
            df['vwap'] = ta.vwap(df['high'], df['low'], df['close'], df['volume'])
            
            # İleri düzey analizler
            high, low = df['high'].max(), df['low'].min()
            close = df['close'].iloc[-1]
            
            # Fibonacci seviyeleri
            fib_levels = self.advanced.fibonacci_levels(high, low)
            for level, value in fib_levels.items():
                df[f'fib_{level}'] = value
            
            # Pivot noktaları
            pivots = self.advanced.pivot_points(high, low, close)
            for point, value in pivots.items():
                df[f'pivot_{point}'] = value
            
            # Hacim profili son 20 mum için
            volume_profile = self.advanced.volume_profile(df.tail(20))
            df['volume_profile'] = df.index.map(lambda x: volume_profile.get(x, 0))
            
            # Remove NaN values
            df.dropna(inplace=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            raise
    
    def generate_signals(self, df):
        """Trading sinyalleri üret"""
        try:
            signals = pd.DataFrame(index=df.index)
            
            # RSI sinyalleri
            signals['rsi_oversold'] = df['rsi'] < 30
            signals['rsi_overbought'] = df['rsi'] > 70
            
            # MACD sinyalleri
            signals['macd_crossover'] = (df['macd'] > df['macd_signal']) & (df['macd'].shift(1) <= df['macd_signal'].shift(1))
            signals['macd_crossunder'] = (df['macd'] < df['macd_signal']) & (df['macd'].shift(1) >= df['macd_signal'].shift(1))
            
            # Bollinger Band sinyalleri
            signals['bb_lower_break'] = df['close'] < df['bb_lower']
            signals['bb_upper_break'] = df['close'] > df['bb_upper']
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating signals: {e}")
            raise 