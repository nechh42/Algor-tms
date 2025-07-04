import time
import pandas as pd
import pandas_ta as ta
from binance.client import Client
from binance.enums import *
import requests
import hashlib
import hmac
import urllib.parse

# API ayarları
BINANCE_API_KEY = "k8Sx3Y27lRIWBJVZ4q9bH65v5p0L9M3dccPpMF7OY8UKke9yPhKfwol3WXTBnuEy"
BINANCE_API_SECRET = "r996pp43QLOEhLtXidl49qTGVMkwKDlalJsVf3PiRI6ix1FrJLpJbBkrg8Tr3Cyt"
TELEGRAM_BOT_TOKEN = "7976320680:AAFGnunvtSix5WfefIsAdE-ifUAQwk2nIcg" 
TELEGRAM_CHAT_ID = "7858725560" 
SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'ADAUSDT']
LEVERAGE = 10

# İstemciyi oluştur
client = Client(api_key=BINANCE_API_KEY, api_secret=BINANCE_API_SECRET)

# Özel imza fonksiyonu
def generate_signature(secret, data):
    return hmac.new(secret.encode('utf-8'), data.encode('utf-8'), hashlib.sha256).hexdigest()

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

def setup_leverage(symbol, leverage):
    """Kaldıraç oranını ayarla"""
    try:
        # Özel imzalı istek
        timestamp = int(time.time() * 1000)
        params = f"symbol={symbol}&leverage={leverage}&timestamp={timestamp}"
        signature = generate_signature(BINANCE_API_SECRET, params)
        
        url = f"https://fapi.binance.com/fapi/v1/leverage?{params}&signature={signature}"
        headers = {"X-MBX-APIKEY": BINANCE_API_KEY}
        
        response = requests.post(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        msg = f"{symbol} için {leverage}x kaldıraç ayarlandı"
        print(msg)
        send_telegram_message(msg)
    except Exception as e:
        error_msg = f"Kaldıraç ayarlama hatası ({symbol}): {e}"
        print(error_msg)
        send_telegram_message(error_msg)

def trading_logic(symbol):
    """EMA crossover ve RSI stratejisi"""
    try:
        # 15 dakikalık mum verilerini al (public endpoint)
        klines = client.futures_klines(symbol=symbol, interval=Client.KLINE_INTERVAL_15MINUTE, limit=100)
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume', 
            'close_time', 'quote_asset_volume', 'trades', 
            'taker_buy_base', 'taker_buy_quote', 'ignore'
        ])
        df['close'] = df['close'].astype(float)
        
        # EMA ve RSI hesapla
        df['ema12'] = ta.ema(df['close'], length=12)
        df['ema26'] = ta.ema(df['close'], length=26)
        df['rsi'] = ta.rsi(df['close'], length=14)
        
        # Sinyal
        if not pd.isna(df['ema12'].iloc[-1]) and not pd.isna(df['ema26'].iloc[-1]) and not pd.isna(df['rsi'].iloc[-1]):
            current_rsi = df['rsi'].iloc[-1]
            
            if df['ema12'].iloc[-1] > df['ema26'].iloc[-1] and current_rsi < 70:
                return 'BUY'
            elif df['ema12'].iloc[-1] < df['ema26'].iloc[-1] and current_rsi > 30:
                return 'SELL'
        return 'HOLD'
    except Exception as e:
        print(f"Strateji hatası ({symbol}): {e}")
        return 'HOLD'

def get_position(symbol):
    """Mevcut pozisyonu kontrol et"""
    try:
        # Özel imzalı istek
        timestamp = int(time.time() * 1000)
        params = f"timestamp={timestamp}"
        signature = generate_signature(BINANCE_API_SECRET, params)
        
        url = f"https://fapi.binance.com/fapi/v2/positionRisk?{params}&signature={signature}"
        headers = {"X-MBX-APIKEY": BINANCE_API_KEY}
        
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        positions = response.json()
        
        for pos in positions:
            if pos['symbol'] == symbol:
                return float(pos['positionAmt'])
        return 0.0
    except Exception as e:
        print(f"Pozisyon bilgisi alma hatası ({symbol}): {e}")
        return 0.0

def create_order(symbol, side, quantity=0.001):
    """Futures emri oluştur ve Telegram'a bildir"""
    try:
        current_position = get_position(symbol)
        
        # Pozisyon kapatma işlemleri
        if side == 'SELL' and current_position > 0:
            # Özel imzalı istek
            timestamp = int(time.time() * 1000)
            params = f"symbol={symbol}&side=SELL&type=MARKET&quantity={abs(current_position)}&reduceOnly=true&timestamp={timestamp}"
            signature = generate_signature(BINANCE_API_SECRET, params)
            
            url = f"https://fapi.binance.com/fapi/v1/order?{params}&signature={signature}"
            headers = {"X-MBX-APIKEY": BINANCE_API_KEY}
            
            response = requests.post(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            msg = f"{symbol} Long pozisyon kapatıldı"
            print(msg)
            send_telegram_message(msg)
            
        elif side == 'BUY' and current_position < 0:
            # Özel imzalı istek
            timestamp = int(time.time() * 1000)
            params = f"symbol={symbol}&side=BUY&type=MARKET&quantity={abs(current_position)}&reduceOnly=true&timestamp={timestamp}"
            signature = generate_signature(BINANCE_API_SECRET, params)
            
            url = f"https://fapi.binance.com/fapi/v1/order?{params}&signature={signature}"
            headers = {"X-MBX-APIKEY": BINANCE_API_KEY}
            
            response = requests.post(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            msg = f"{symbol} Short pozisyon kapatıldı"
            print(msg)
            send_telegram_message(msg)
        
        # Yeni pozisyon aç
        if (side == 'BUY' and current_position <= 0) or (side == 'SELL' and current_position >= 0):
            order_side = "BUY" if side == 'BUY' else "SELL"
            
            # Özel imzalı istek
            timestamp = int(time.time() * 1000)
            params = f"symbol={symbol}&side={order_side}&type=MARKET&quantity={quantity}&timestamp={timestamp}"
            signature = generate_signature(BINANCE_API_SECRET, params)
            
            url = f"https://fapi.binance.com/fapi/v1/order?{params}&signature={signature}"
            headers = {"X-MBX-APIKEY": BINANCE_API_KEY}
            
            response = requests.post(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            msg = f"{symbol} {side} emri gönderildi ({quantity})"
            print(msg)
            send_telegram_message(msg)
            
    except Exception as e:
        error_msg = f"Emir gönderme hatası ({symbol}): {e}"
        print(error_msg)
        send_telegram_message(error_msg)

def main():
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - === BINANCE FUTURES TRADER BAŞLATILDI ===")
    send_telegram_message("Binance Futures Trader başlatıldı")
    
    # Kaldıraçları ayarla
    for symbol in SYMBOLS:
        setup_leverage(symbol, LEVERAGE)
    
    while True:
        for symbol in SYMBOLS:
            try:
                # Fiyat bilgisi (public endpoint)
                ticker = client.futures_symbol_ticker(symbol=symbol)
                price = float(ticker['price'])
                print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {symbol} Fiyat: {price}")
                
                # Karar ve pozisyon
                decision = trading_logic(symbol)
                position = get_position(symbol)
                print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {symbol} Karar: {decision}, Pozisyon: {position}")
                
                # Emir gönder (test için yorum satırı yapabilirsiniz)
                # create_order(symbol, decision, quantity=0.001)
                
                # Sadece sinyal bildirimi
                if decision in ['BUY', 'SELL']:
                    message = f"{symbol} - {decision} - Fiyat: {price}"
                    send_telegram_message(message)
                
            except Exception as e:
                error_msg = f"Hata ({symbol}): {e}"
                print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {error_msg}")
                send_telegram_message(error_msg)
        
        # 5 dakika bekle
        print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Bir sonraki tarama için bekleniyor...")
        time.sleep(300)

if __name__ == "__main__":
    main()