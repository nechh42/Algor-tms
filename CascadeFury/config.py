import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Configuration
API_KEY = os.getenv('BINANCE_API_KEY')
API_SECRET = os.getenv('BINANCE_API_SECRET')

# Trading Parameters
MAX_LEVERAGE = int(os.getenv('MAX_LEVERAGE', 20))
GRID_LEVELS = int(os.getenv('GRID_LEVELS', 5))
POSITION_SIZE_USDT = float(os.getenv('POSITION_SIZE_USDT', 100))
STOP_LOSS_PERCENT = float(os.getenv('STOP_LOSS_PERCENT', 2))
TAKE_PROFIT_PERCENT = float(os.getenv('TAKE_PROFIT_PERCENT', 3))

# Trading Pairs
TRADING_PAIRS = [
    'BTCUSDT',  # Bitcoin
    'ETHUSDT',  # Ethereum
    'BNBUSDT',  # Binance Coin
    'SOLUSDT',  # Solana
    'AVAXUSDT', # Avalanche
    'DOTUSDT',  # Polkadot
    'ADAUSDT',  # Cardano
    'MATICUSDT' # Polygon
]

# Test Mode
TEST_MODE = os.getenv('TEST_MODE', 'true').lower() == 'true'

# Timeframes
TIMEFRAMES = {
    '1m': '1m',
    '5m': '5m',
    '15m': '15m',
    '1h': '1h',
    '4h': '4h',
    '1d': '1d'
}

# Default Timeframe
DEFAULT_TIMEFRAME = TIMEFRAMES['5m']
