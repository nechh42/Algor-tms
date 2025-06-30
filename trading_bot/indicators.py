# indicators.py - Teknik göstergeler
import pandas as pd
import ta

def calculate_all_indicators(df):
    # RSI
    df['rsi'] = ta.momentum.RSIIndicator(df['close']).rsi()
    
    # MACD
    macd = ta.trend.MACD(df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    
    # Bollinger Bands
    bb = ta.volatility.BollingerBands(df['close'])
    df['bb_upper'] = bb.bollinger_hband()
    df['bb_middle'] = bb.bollinger_mavg()
    df['bb_lower'] = bb.bollinger_lband()
    
    # Fibonacci
    high = df['high'].max()
    low = df['low'].min()
    diff = high - low
    df['fib_0'] = low
    df['fib_236'] = low + (diff * 0.236)
    df['fib_382'] = low + (diff * 0.382)
    df['fib_618'] = low + (diff * 0.618)
    df['fib_100'] = high
    
    return df