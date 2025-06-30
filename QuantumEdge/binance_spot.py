import time
import pandas as pd
import pandas_ta as ta
from binance.client import Client
import requests

# API ayarları
BINANCE_API_KEY = "k8Sx3Y27lRIWBJVZ4q9bH65v5p0L9M3dccPpMF7OY8UKke9yPhKfwol3WXTBnuEy"
BINANCE_API_SECRET = "r996pp43QLOEhLtXidl49qTGVMkwKDlalJsVf3PiRI6ix1FrJLpJbBkrg8Tr3Cyt"
TELEGRAM_BOT_TOKEN = "7976320680:AAFGnunvtSix5WfefIsAdE-ifUAQwk2nIcg" 
TELEGRAM_CHAT_ID = "7858725560"
SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'ADAUSDT']

# İstemciyi oluştur (düzeltildi)
client = Client(api_key=BINANCE_API_KEY, api_secret=BINANCE_API_SECRET)

def send_telegram_message(message):
    """Telegram'a mesaj gönder"""
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message
    }
    try:
        response = requests.post(url, json=payload, timeout=10)
        return response.json()
    except Exception as e:
        print(f"Telegram mesaj gönderme hatası: {e}")

def trading_logic(symbol):
    """EMA crossover stratejisi ile al/sat sinyali üret"""
    try:
        # 1 saatlik mum verilerini al (son 100 mum)
        klines = client.get_klines(symbol=symbol, interval=Client.KLINE_INTERVAL_1HOUR, limit=100)
        df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignore'])
        df['close'] = df['close'].astype(float)
        
        # EMA hesapla
        df['ema12'] = ta.ema(df['close'], length=12)
        df['ema26'] = ta.ema(df['close'], length=26)
        
        # Sinyal belirle
        if not pd.isna(df['ema12'].iloc[-1]) and not pd.isna(df['ema26'].iloc[-1]):
            if df['ema12'].iloc[-1] > df['ema26'].iloc[-1]:
                return 'BUY'
            else:
                return 'SELL'
        return 'HOLD'
    except Exception as e:
        print(f"Strateji hatası ({symbol}): {e}")
        return 'HOLD'

def main():
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - === BINANCE SPOT TRADER BAŞLATILDI ===")
    send_telegram_message("Binance Spot Trader başlatıldı")
    
    while True:
        for symbol in SYMBOLS:
            try:
                # Piyasa verilerini kontrol et
                ticker = client.get_symbol_ticker(symbol=symbol)
                price = float(ticker['price'])
                print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {symbol} Fiyat: {price}")
                
                # Ticaret kararını al
                decision = trading_logic(symbol)
                print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {symbol} Karar: {decision}")
                
                # BUY/SELL işlemleri
                if decision in ['BUY', 'SELL']:
                    message = f"{symbol} - {decision} - Fiyat: {price}"
                    send_telegram_message(message)
                
            except Exception as e:
                error_msg = f"Hata ({symbol}): {e}"
                print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {error_msg}")
                send_telegram_message(error_msg)
        
        # 1 dakika bekle
        time.sleep(60)

if __name__ == "__main__":
    main()