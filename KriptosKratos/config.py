import os
from dotenv import load_dotenv

# .env dosyasını yükle
load_dotenv()

# API Ayarları
BINANCE_API_KEY = os.getenv('BINANCE_API_KEY')
BINANCE_API_SECRET = os.getenv('BINANCE_API_SECRET')

# Trading Bot Konfigürasyonu
class Config:
    def __init__(self):
        # Temel ayarlar
        self.SYMBOLS = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'XRP/USDT', 'ADA/USDT']
        self.TIMEFRAME = '1h'  # 1 saatlik mum
        self.CHECK_INTERVAL = 10  # 10 saniye
        
        # Teknik gösterge parametreleri
        self.RSI_PERIOD = 14
        self.EMA_SHORT = 9
        self.EMA_LONG = 21
        self.MACD_FAST = 12
        self.MACD_SLOW = 26
        self.MACD_SIGNAL = 9
        
        # Sinyal eşikleri
        self.RSI_OVERSOLD = 30
        self.RSI_OVERBOUGHT = 70
        self.SIGNAL_THRESHOLD = 0.5

# Dosya Yolları
ML_MODEL_PATH = 'ml_model.h5'
SCALER_PATH = 'scaler.save'
LOG_FILE = 'bot.log'
