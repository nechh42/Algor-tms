import time
import pandas as pd
import pandas_ta as ta
import requests
import hashlib
import hmac
import urllib.parse
import json

# API ayarları
BINANCE_API_KEY = ""
BINANCE_API_SECRET = ""
TELEGRAM_BOT_TOKEN = "7976320680:AAFGnunvtSix5WfefIsAdE-ifUAQwk2nIcg" 
TELEGRAM_CHAT_ID = "7858725560" 
SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'ADAUSDT']
LEVERAGE = 10
RECV_WINDOW = 60000  # 60 saniye

# Binance API endpointleri
BASE_URL = "https://fapi.binance.com"
FUTURES_API = "/fapi/v1"
FUTURES_API_V2 = "/fapi/v2"

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

def generate_signature(data):
    """HMAC SHA256 imzası oluştur"""
    query_string = urllib.parse.urlencode(data)
    return hmac.new(
        BINANCE_API_SECRET.encode('utf-8'),
        query_string.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()

def binance_request(method, endpoint, params=None, signed=False):
    """Binance API'ye istek gönder"""
    url = f"{BASE_URL}{endpoint}"
    headers = {"X-MBX-APIKEY": BINANCE_API_KEY}
    
    try:
        if signed:
            params = params or {}
            params['timestamp'] = int(time.time() * 1000)
            params['recvWindow'] = RECV_WINDOW
            
            # Parametreleri alfabetik sırala
            params = dict(sorted(params.items()))
            
            # İmza oluştur
            params['signature'] = generate_signature(params)
            
            if method == 'GET':
                response = requests.get(url, headers=headers, params=params, timeout=10)
            else:
                response = requests.post(url, headers=headers, params=params, timeout=10)
        else:
            if method == 'GET':
                response = requests.get(url, params=params, timeout=10)
            else:
                response = requests.post(url, data=params, timeout=10)
        
        response.raise_for_status()
        return response.json()
    except Exception as e:
        error_msg = f"Binance istek hatası: {e}"
        if hasattr(e, 'response') and e.response:
            error_msg += f" | Yanıt: {e.response.text}"
        print(error_msg)
        send_telegram_message(error_msg)
        return None

def setup_leverage(symbol, leverage):
    """Kaldıraç oranını ayarla"""
    try:
        params = {
            "symbol": symbol,
            "leverage": leverage
        }
        result = binance_request('POST', f"{FUTURES_API}/leverage", params, signed=True)
        
        if result and 'leverage' in result:
            msg = f"{symbol} için {leverage}x kaldıraç ayarlandı"
            print(msg)
            send_telegram_message(msg)
            return True
        
        return False
    except Exception as e:
        error_msg = f"Kaldıraç ayarlama hatası ({symbol}): {e}"
        print(error_msg)
        send_telegram_message(error_msg)
        return False

def get_klines(symbol, interval='15m', limit=100):
    """Mum verilerini al"""
    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit
    }
    return binance_request('GET', f"{FUTURES_API}/klines", params)

def get_symbol_ticker(symbol):
    """Sembol fiyatını al"""
    params = {"symbol": symbol}
    result = binance_request('GET', f"{FUTURES_API}/ticker/price", params)
    return result if result else {'price': '0'}

def get_position(symbol):
    """Mevcut pozisyonu kontrol et"""
    try:
        positions = binance_request('GET', f"{FUTURES_API_V2}/positionRisk", signed=True)
        if positions:
            for pos in positions:
                if pos['symbol'] == symbol:
                    return float(pos['positionAmt'])
        return 0.0
    except Exception as e:
        print(f"Pozisyon bilgisi alma hatası ({symbol}): {e}")
        return 0.0

def trading_logic(symbol):
    """EMA crossover ve RSI stratejisi"""
    try:
        klines = get_klines(symbol)
        if not klines:
            return 'HOLD'
            
        # DataFrame oluştur
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume', 
            'close_time', 'quote_asset_volume', 'trades', 
            'taker_buy_base', 'taker_buy_quote', 'ignore'
        ])
        
        # Sayısal değerlere çevir
        df['close'] = df['close'].astype(float)
        
        # EMA hesapla
        df['ema12'] = ta.ema(df['close'], length=12)
        df['ema26'] = ta.ema(df['close'], length=26)
        df['rsi'] = ta.rsi(df['close'], length=14)
        
        # Son değerleri al
        last_ema12 = df['ema12'].iloc[-1]
        last_ema26 = df['ema26'].iloc[-1]
        last_rsi = df['rsi'].iloc[-1]
        
        # NaN kontrolü
        if pd.isna(last_ema12) or pd.isna(last_ema26) or pd.isna(last_rsi):
            return 'HOLD'
            
        # Sinyal üret
        if last_ema12 > last_ema26 and last_rsi < 70:
            return 'BUY'
        elif last_ema12 < last_ema26 and last_rsi > 30:
            return 'SELL'
        return 'HOLD'
    except Exception as e:
        print(f"Strateji hatası ({symbol}): {e}")
        return 'HOLD'

def create_order(symbol, side, quantity=0.001):
    """Emir oluştur"""
    try:
        current_position = get_position(symbol)
        quantity = round(quantity, 3)  # Binance hassasiyeti
        
        # Pozisyon kapatma
        if side == 'SELL' and current_position > 0:
            params = {
                "symbol": symbol,
                "side": "SELL",
                "type": "MARKET",
                "quantity": abs(current_position),
                "reduceOnly": "true"
            }
            result = binance_request('POST', f"{FUTURES_API}/order", params, signed=True)
            if result:
                msg = f"{symbol} Long pozisyon kapatıldı"
                print(msg)
                send_telegram_message(msg)
        
        elif side == 'BUY' and current_position < 0:
            params = {
                "symbol": symbol,
                "side": "BUY",
                "type": "MARKET",
                "quantity": abs(current_position),
                "reduceOnly": "true"
            }
            result = binance_request('POST', f"{FUTURES_API}/order", params, signed=True)
            if result:
                msg = f"{symbol} Short pozisyon kapatıldı"
                print(msg)
                send_telegram_message(msg)
        
        # Yeni pozisyon aç
        if (side == 'BUY' and current_position <= 0) or (side == 'SELL' and current_position >= 0):
            params = {
                "symbol": symbol,
                "side": side.upper(),
                "type": "MARKET",
                "quantity": quantity
            }
            result = binance_request('POST', f"{FUTURES_API}/order", params, signed=True)
            if result:
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
                # Fiyat bilgisi
                ticker = get_symbol_ticker(symbol)
                price = float(ticker.get('price', 0))
                print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {symbol} Fiyat: {price}")
                
                # Ticaret kararını al
                decision = trading_logic(symbol)
                print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {symbol} Karar: {decision}")
                
                # Pozisyon bilgisi
                position = get_position(symbol)
                print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {symbol} Pozisyon: {position}")
                
                # Sinyal bildirimi
                if decision in ['BUY', 'SELL']:
                    message = f"{symbol} - {decision} - Fiyat: {price}"
                    send_telegram_message(message)
                
                # Emir gönder (test için yorum satırı yapabilirsiniz)
                # create_order(symbol, decision, quantity=0.001)
                
            except Exception as e:
                error_msg = f"Hata ({symbol}): {e}"
                print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {error_msg}")
                send_telegram_message(error_msg)
        
        # 5 dakika bekle
        print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Bir sonraki tarama için bekleniyor...")
        time.sleep(300)

if __name__ == "__main__":
    main()