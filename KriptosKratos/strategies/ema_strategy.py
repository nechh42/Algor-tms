import pandas as pd
import numpy as np
import talib

class EMAStrategy:
    def __init__(self, config):
        self.config = config
        self.ema_short = config.EMA_SHORT
        self.ema_long = config.EMA_LONG

    def calculate_signals(self, df: pd.DataFrame) -> tuple:
        """EMA sinyallerini hesapla"""
        try:
            # EMA hesapla
            df['ema_short'] = talib.EMA(df['close'], timeperiod=self.ema_short)
            df['ema_long'] = talib.EMA(df['close'], timeperiod=self.ema_long)
            
            # Son iki mum için EMA değerleri
            current_short = df['ema_short'].iloc[-1]
            current_long = df['ema_long'].iloc[-1]
            prev_short = df['ema_short'].iloc[-2]
            prev_long = df['ema_long'].iloc[-2]
            
            # Çapraz kontrol
            signal = None
            if current_short > current_long and prev_short <= prev_long:
                signal = 'LONG'
            elif current_short < current_long and prev_short >= prev_long:
                signal = 'SHORT'
            
            return signal, {
                'ema_short': current_short,
                'ema_long': current_long,
                'price': df['close'].iloc[-1]
            }
            
        except Exception as e:
            print(f"EMA hesaplama hatası: {e}")
            return None, None

    def calculate_stop_loss(self, entry_price: float, signal_type: str) -> float:
        """Stop loss hesapla"""
        if signal_type == 'LONG':
            return entry_price * (1 - self.config.MAX_POSITION_LOSS / 100)
        else:
            return entry_price * (1 + self.config.MAX_POSITION_LOSS / 100)

    def calculate_take_profit(self, entry_price: float, signal_type: str) -> float:
        """Take profit hesapla"""
        risk = entry_price * (self.config.MAX_POSITION_LOSS / 100)
        if signal_type == 'LONG':
            return entry_price + (risk * 2)  # Risk:Reward = 1:2
        else:
            return entry_price - (risk * 2)
