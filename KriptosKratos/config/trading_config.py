"""
Trading bot konfigürasyon dosyası
"""

# Trading parametreleri
TRADE_EXECUTION = {
    'min_trade_amount': 10,  # Minimum işlem miktarı (USDT)
    'max_trade_amount': 100,  # Maksimum işlem miktarı (USDT)
    'min_trades_interval': 5,  # İşlemler arası minimum süre (dakika)
    'max_open_positions': 3,  # Maksimum açık pozisyon sayısı
}

# Risk yönetimi parametreleri
RISK_MANAGEMENT = {
    'max_risk_per_trade': 0.02,  # İşlem başına maksimum risk (%)
    'stop_loss': 0.02,  # Stop loss (%)
    'take_profit': 0.04,  # Take profit (%)
    'trailing_stop': 0.01,  # Trailing stop (%)
}

# Teknik gösterge parametreleri
TECHNICAL_INDICATORS = {
    'ema_periods': [9, 21, 50, 200],  # EMA periyotları
    'rsi_period': 14,  # RSI periyodu
    'rsi_overbought': 70,  # RSI aşırı alım seviyesi
    'rsi_oversold': 30,  # RSI aşırı satım seviyesi
    'macd_fast': 12,  # MACD hızlı periyot
    'macd_slow': 26,  # MACD yavaş periyot
    'macd_signal': 9,  # MACD sinyal periyodu
    'bb_period': 20,  # Bollinger Bands periyodu
    'bb_std': 2,  # Bollinger Bands standart sapma
    'adx_period': 14,  # ADX periyodu
    'adx_threshold': 25,  # ADX eşik değeri
    'cci_period': 20,  # CCI periyodu
    'cci_overbought': 100,  # CCI aşırı alım seviyesi
    'cci_oversold': -100,  # CCI aşırı satım seviyesi
    'atr_period': 14,  # ATR periyodu
    'volume_ma_period': 20,  # Hacim MA periyodu
}

# Giriş kriterleri
ENTRY_CRITERIA = {
    'min_signal_strength': 60,  # Minimum sinyal gücü (0-100)
    'min_volume_increase': 1.5,  # Minimum hacim artışı (kat)
    'volatility_range': {
        'min': 0.5,  # Minimum volatilite (%)
        'max': 5.0  # Maksimum volatilite (%)
    },
    'trend_strength': 25,  # Minimum trend gücü (ADX)
    'min_profit_potential': 0.02,  # Minimum kar potansiyeli (%)
}

# AI model parametreleri
AI_MODEL = {
    'confidence_threshold': 70,  # Minimum güven skoru (%)
    'prediction_window': 12,  # Tahmin penceresi (mum sayısı)
    'training_window': 1000,  # Eğitim penceresi (mum sayısı)
    'features': [
        'close',
        'volume',
        'rsi',
        'macd',
        'bb_upper',
        'bb_lower',
        'adx',
        'cci',
        'atr'
    ]
}
