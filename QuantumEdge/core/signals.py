# Üst kısma ekleyin
import time
import logging
import pandas as pd
import numpy as np
import pandas_ta as ta
from core.market import get_market_data
import config.settings as settings  # DÜZELTİLMİŞ IMPORT

def generate_signal():
    try:
        # 300 mumluk veri (yaklaşık 1 günlük 5m veri)
        data = get_market_data(settings.SYMBOL, settings.TIMEFRAME, 300)
        if not data or len(data) < 100:
            return None
        
        # ... kodun geri kalanı aynı ...
        # DataFrame oluştur
        df = pd.DataFrame(data, columns=['timestamp','open','high','low','close','volume'])
        
        # Göstergeleri hesapla
        df['ema_fast'] = ta.ema(df['close'], settings.EMA_FAST)
        df['ema_slow'] = ta.ema(df['close'], settings.EMA_SLOW)
        df['rsi'] = ta.rsi(df['close'], settings.RSI_PERIOD)
        df['volume_ma'] = ta.sma(df['volume'], 20)
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        # Bollinger Bantları
        bb = ta.bbands(df['close'], length=settings.BB_PERIOD, std=settings.BB_STDDEV)
        df = pd.concat([df, bb], axis=1)
        
        # Son mum
        last = df.iloc[-1]
        prev = df.iloc[-2]
        
        # 1. Trend Filtresi (EMA)
        uptrend = last['ema_fast'] > last['ema_slow']
        downtrend = last['ema_fast'] < last['ema_slow']
        
        # 2. Momentum Filtresi (RSI)
        rsi_rising = last['rsi'] > prev['rsi']
        rsi_falling = last['rsi'] < prev['rsi']
        
        # 3. Hacim Filtresi
        high_volume = last['volume_ratio'] >= settings.VOLUME_MULTIPLIER
        
        # 4. Volatilite Filtresi (Bollinger)
        in_squeeze = (last['close'] > last[f'BBL_{settings.BB_PERIOD}_{settings.BB_STDDEV}']) and \
                     (last['close'] < last[f'BBU_{settings.BB_PERIOD}_{settings.BB_STDDEV}'])
        
        # Sinyal Koşulları
        long_condition = uptrend and rsi_rising and high_volume and in_squeeze
        short_condition = downtrend and rsi_falling and high_volume and in_squeeze
        
        if long_condition:
            return 'long'
        elif short_condition:
            return 'short'
            
        return None
        
    except Exception as e:
        print(f"Sinyal hatası: {str(e)}")
        return None