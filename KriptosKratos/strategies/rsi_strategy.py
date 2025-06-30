import pandas as pd
import numpy as np
import talib

class RSIStrategy:
    def __init__(self, config):
        self.config = config
        self.rsi_period = config.RSI_PERIOD
        self.rsi_overbought = config.RSI_OVERBOUGHT
        self.rsi_oversold = config.RSI_OVERSOLD

    def calculate_signals(self, df: pd.DataFrame) -> tuple:
        """RSI sinyallerini hesapla"""
        try:
            # RSI hesapla
            df['rsi'] = talib.RSI(df['close'], timeperiod=self.rsi_period)
            
            # Son RSI değeri
            current_rsi = df['rsi'].iloc[-1]
            
            # Sinyal oluştur
            signal = None
            if current_rsi < self.rsi_oversold:
                signal = 'LONG'
            elif current_rsi > self.rsi_overbought:
                signal = 'SHORT'
            
            return signal, {
                'rsi': current_rsi,
                'price': df['close'].iloc[-1]
            }
            
        except Exception as e:
            print(f"RSI hesaplama hatası: {e}")
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
