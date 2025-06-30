from dotenv import load_dotenv
import os

# .env dosyasını yükle
load_dotenv()

# Binance API bilgileri
API_KEY = os.getenv('BINANCE_API_KEY')
API_SECRET = os.getenv('BINANCE_API_SECRET')

# Bot Ayarları
TRADE_MODE = "FUTURES"  # FUTURES veya MARGIN
USE_ISOLATED_MARGIN = True  # Izole margin kullan

# Risk Yönetimi
ACCOUNT_RISK_PERCENTAGE = 50  # Bakiyenin %50'sini kullan
MIN_VOLUME_RATIO = 1.5
MIN_SIGNAL_STRENGTH = 70  # Daha güçlü sinyaller
MIN_OPPORTUNITY_SCORE = 75  # Daha iyi fırsatlar

# İşlem Parametreleri
MIN_TRADE_AMOUNT = 20.0  # Minimum işlem miktarı 20 USDT
MIN_PROFIT_USDT = 2.0  # Minimum kâr hedefi 2 USDT
MAX_OPEN_POSITIONS = 2
SCAN_INTERVAL = 60

# Çoklu Take Profit Seviyeleri
TAKE_PROFIT_LEVELS = {
    50: 1.0,   # İlk %50'yi %1.0'de kapat
    100: 1.5   # Kalan %50'yi %1.5'te kapat
}

# Stop Loss Ayarları
STOP_LOSS_PERCENTAGE = 1.5  # Stop loss mesafesi %1.5
TRAILING_STOP = True
TRAILING_STOP_ACTIVATION = 1.2  # %1.2 kârda trailing stop başlat
TRAILING_STOP_DISTANCE = 0.5  # %0.5 trailing stop mesafesi

# Kaldıraç Ayarları
MAX_LEVERAGE = 5  # Maksimum 5x
DEFAULT_LEVERAGE = 3  # Varsayılan 3x

# Volatilite Bazlı Kaldıraç
VOLATILITY_LEVERAGE_RANGES = {
    1.0: 5,   # Volatilite ≤ %1: 5x
    2.0: 4,    # Volatilite ≤ %2: 4x
    3.0: 3,    # Volatilite ≤ %3: 3x
    999.0: 2   # Volatilite > %3: 2x
}

# Risk/Ödül Oranı
MIN_RISK_REWARD_RATIO = 1.5  # Minimum 1:1.5 risk/ödül oranı

# Taranacak Coinler
SCAN_SYMBOLS = [
    # Majör Coinler (En güvenilir)
    'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT', 'ADAUSDT',
    
    # Layer 1 Zincirler
    'SOLUSDT', 'AVAXUSDT', 'MATICUSDT', 'DOTUSDT', 'ATOMUSDT',
    'NEARUSDT', 'FTMUSDT', 'ONEUSDT', 'ALGOUSDT', 'ICPUSDT',
    
    # DeFi Projeleri
    'LINKUSDT', 'AAVEUSDT', 'UNIUSDT', 'SNXUSDT', 'CRVUSDT',
    'MKRUSDT', 'COMPUSDT', 'SUSHIUSDT', 'YFIUSDT', '1INCHUSDT',
    
    # Metaverse & Gaming
    'MANAUSDT', 'SANDUSDT', 'AXSUSDT', 'ENJUSDT', 'GALAUSDT',
    'CHZUSDT', 'THETAUSDT', 'ILVUSDT', 'IMXUSDT', 'GMTUSDT',
    
    # AI & Big Data
    'FETUSDT', 'OCEANUSDT', 'AGIXUSDT', 'RNDR', 'ROSEUSDT',
    
    # Exchange Tokens
    'OKBUSDT', 'HTUSDT', 'FTTUSDT', 'KCSUSDT', 'GTUSDT',
    
    # Yüksek Potansiyelli Altcoinler
    'INJUSDT', 'GMXUSDT', 'LDOUSDT', 'OPUSDT', 'ARBUSDT',
    'SUIUSDT', 'APTUSDT', 'STXUSDT', 'KAVAUSDT', 'ZILUSDT',
    
    # Stablecoin Alternatifleri
    'DAIUSDT', 'USTCUSDT', 'BUSDUSDT', 'TUSDUSDT',
    
    # Eski Güçlü Projeler
    'LTCUSDT', 'EOSUSDT', 'TRXUSDT', 'NEOUSDT', 'VETUSDT',
    'WAVESUSDT', 'DASHUSDT', 'XEMUSDT', 'ZECUSDT', 'XMRUSDT',
    
    # Yeni Trendler
    'STGUSDT', 'SPELLUSDT', 'BELUSDT', 'FLMUSDT', 'RNDRUSDT',
    'HIGHUSDT', 'ASTRUSDT', 'AGLDUSDT', 'BICOUSDT', 'CTSIUSDT',
    
    # Yüksek Hacimli Diğerleri
    'HBARUSDT', 'QTUMUSDT', 'KLAYUSDT', 'IOTXUSDT', 'CELOUSDT',
    'IOTAUSDT', 'ONTUSDT', 'SKLUSDT', 'RVNUSDT', 'STORJUSDT'
]

# Volatilite Limitleri
MIN_SL_DISTANCE = 0.5  # Minimum stop loss mesafesi (%)
MAX_SL_DISTANCE = 2.0  # Maksimum stop loss mesafesi (%)
MAX_VOLATILITY = 10.0  # Maksimum izin verilen volatilite (%)

# Trend Analizi
TREND_EMA_FAST = 9  # Hızlı EMA periyodu
TREND_EMA_SLOW = 21  # Yavaş EMA periyodu
TREND_SUPERTREND_PERIOD = 10  # SuperTrend periyodu
TREND_SUPERTREND_MULTIPLIER = 3  # SuperTrend çarpanı

# Momentum Göstergeleri
STOCH_RSI_PERIOD = 14  # Stochastic RSI periyodu
STOCH_RSI_OVERSOLD = 20  # Stochastic RSI aşırı satım
STOCH_RSI_OVERBOUGHT = 80  # Stochastic RSI aşırı alım

# Hacim Göstergeleri
OBV_TREND_PERIOD = 20  # OBV trend periyodu
VWAP_PERIOD = 14  # VWAP periyodu
