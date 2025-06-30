import time
import ccxt
import time
import logging
from datetime import datetime
from core.signals import generate_signal
from config import api_keys
import config.settings as settings

def get_binance_connection():
    """Binance bağlantısını oluşturur (zaman senkronizasyonlu)"""
    exchange = ccxt.binance({
        'apiKey': 'k8Sx3Y27lRIWBJVZ4q9bH65v5p0L9M3dccPpMF7OY8UKke9yPhKfwol3WXTBnuEy',
        'secret': 'r996pp43QLOEhLtXidl49qTGVMkwKDlalJsVf3PiRI6ix1FrJLpJbBkrg8Tr3Cyt',
        'enableRateLimit': True,
        'options': {
            'defaultType': 'spot',
            'recvWindow': 10000  # Zaman sınırını 10 saniyeye çıkar
        },
    })
    
    # Zaman senkronizasyonu
    try:
        server_time = exchange.fetch_time()
        local_time = int(time.time() * 1000)
        exchange.time_difference = server_time - local_time
        exchange.options['adjustForTimeDifference'] = True
    except:
        pass  # Senkronizasyon hatasında bile çalışmaya devam et
        
    return exchange

def get_market_data(symbol, timeframe, limit=100):
    """Piyasa verilerini getirir (zaman aşımı düzeltmeli)"""
    binance = get_binance_connection()
    for _ in range(3):  # 3 deneme
        try:
            return binance.fetch_ohlcv(symbol, timeframe, limit=limit)
        except ccxt.RequestTimeout:
            logging.warning("Zaman aşımı, yeniden deniyor...")
            time.sleep(2)
    return None

def get_current_ticker(symbol):
    """Anlık fiyat bilgisini getirir"""
    binance = get_binance_connection()
    return binance.fetch_ticker(symbol)