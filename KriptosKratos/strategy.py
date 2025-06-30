import numpy as np
import pandas as pd
import talib
from typing import Dict, List, Tuple

class TradingStrategy:
    def __init__(self, config):
        self.config = config
        
    def calculate_indicators(self, df: pd.DataFrame) -> Dict:
        """Teknik göstergeleri hesapla"""
        indicators = {}
        
        # RSI
        indicators['rsi'] = talib.RSI(df['close'], timeperiod=self.config.RSI_PERIOD)
        
        # EMA
        indicators['ema_short'] = talib.EMA(df['close'], timeperiod=self.config.EMA_SHORT)
        indicators['ema_long'] = talib.EMA(df['close'], timeperiod=self.config.EMA_LONG)
        
        # Bollinger Bands
        upper, middle, lower = talib.BBANDS(df['close'], timeperiod=20)
        indicators['bb_upper'] = upper
        indicators['bb_middle'] = middle
        indicators['bb_lower'] = lower
        
        # Pivot Points
        indicators['pivot'] = self.calculate_pivot_points(df)
        
        return indicators
    
    def calculate_pivot_points(self, df: pd.DataFrame) -> Dict:
        """Pivot noktalarını hesapla"""
        pivot = (df['high'].iloc[-1] + df['low'].iloc[-1] + df['close'].iloc[-1]) / 3
        r1 = 2 * pivot - df['low'].iloc[-1]
        s1 = 2 * pivot - df['high'].iloc[-1]
        r2 = pivot + (df['high'].iloc[-1] - df['low'].iloc[-1])
        s2 = pivot - (df['high'].iloc[-1] - df['low'].iloc[-1])
        
        return {
            'pivot': pivot,
            'r1': r1,
            's1': s1,
            'r2': r2,
            's2': s2
        }
    
    def calculate_position_size(self, balance: float, risk_per_trade: float) -> float:
        """Kelly Criterion bazlı pozisyon boyutu hesaplama"""
        kelly_size = balance * self.config.KELLY_FRACTION * risk_per_trade
        return min(kelly_size, balance * self.config.POSITION_SIZE)
    
    def check_entry_conditions(self, indicators: Dict, current_price: float) -> Tuple[bool, str]:
        """Giriş koşullarını kontrol et"""
        # RSI koşulları
        rsi = indicators['rsi'].iloc[-1]
        if rsi < self.config.RSI_OVERSOLD:
            return True, "LONG"
        elif rsi > self.config.RSI_OVERBOUGHT:
            return True, "SHORT"
            
        # EMA çapraz kontrolü
        if (indicators['ema_short'].iloc[-1] > indicators['ema_long'].iloc[-1] and 
            indicators['ema_short'].iloc[-2] <= indicators['ema_long'].iloc[-2]):
            return True, "LONG"
        elif (indicators['ema_short'].iloc[-1] < indicators['ema_long'].iloc[-1] and 
              indicators['ema_short'].iloc[-2] >= indicators['ema_long'].iloc[-2]):
            return True, "SHORT"
            
        return False, ""
    
    def calculate_stop_loss(self, entry_price: float, position_type: str) -> float:
        """Stop loss seviyesi hesapla"""
        if position_type == "LONG":
            return entry_price * (1 - self.config.MAX_POSITION_LOSS / 100)
        else:
            return entry_price * (1 + self.config.MAX_POSITION_LOSS / 100)
    
    def calculate_take_profit(self, entry_price: float, position_type: str, risk_reward_ratio: float = 2.0) -> float:
        """Take profit seviyesi hesapla"""
        stop_loss = self.calculate_stop_loss(entry_price, position_type)
        risk = abs(entry_price - stop_loss)
        
        if position_type == "LONG":
            return entry_price + (risk * risk_reward_ratio)
        else:
            return entry_price - (risk * risk_reward_ratio)
    
    def check_risk_levels(self, daily_pnl: float, max_drawdown: float) -> bool:
        """Risk seviyelerini kontrol et"""
        if (abs(daily_pnl) > self.config.MAX_DAILY_LOSS or 
            abs(max_drawdown) > self.config.MAX_DRAWDOWN):
            return False
        return True
