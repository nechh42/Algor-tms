# config/settings.py
RISK_PER_TRADE = 0.01  # İşlem başına risk (%1)
DAILY_PROFIT_TARGET = 3
DAILY_LOSS_LIMIT = -5
MAX_TRADES = 10
SYMBOL = "BTC/USDT"
TIMEFRAME = "5m"

# Strateji Parametreleri
EMA_FAST = 12
EMA_SLOW = 26
RSI_PERIOD = 14
VOLUME_MULTIPLIER = 2.0  # Ortalama hacmin katı
MIN_SPREAD = 0.05        # %0.05
MAX_SPREAD = 0.10        # %0.10
BB_PERIOD = 20
BB_STDDEV = 1.5