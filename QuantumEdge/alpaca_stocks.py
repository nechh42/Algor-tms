import time
import logging
import alpaca_trade_api as tradeapi
import pandas as pd
import pandas_ta as ta
import requests
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

# ===================== KONFİGÜRASYON =====================
ALPACA_API_KEY = "PKSLNCOFRXKBV2HR3KM"
ALPACA_API_SECRET = "Gw0dSWB61LtPfaY2pueLK2hJUGEEI"
TELEGRAM_BOT_TOKEN = "7976320680:AAFGnunvtSix5WfefIsAdE-ifUAQwk2nIcg"  # YENİ BOT TOKEN
TELEGRAM_CHAT_ID = "7858725560"  # YENİ CHAT ID

SYMBOLS = ["TSLA", "AAPL", "NVDA", "AMD"]
TIMEFRAME = "5Min"
INITIAL_BALANCE = 100.0
RISK_PER_TRADE = 0.01
VOLUME_MULTIPLIER = 1.5  # Hacim oranı için eşik değer

# === Teknik indikatör parametreleri ===
EMA_FAST = 9
EMA_SLOW = 21
RSI_PERIOD = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
# =========================================================

# ==================== LOG AYARLARI =======================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger()
# =========================================================

def send_telegram(message):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML"}
        response = requests.post(url, json=payload, timeout=5)
        return response.status_code == 200
    except Exception as e:
        logger.error(f"Telegram error: {str(e)}")
        return False

class AlpacaStockTrader:
    def __init__(self, initial_balance):
        self.balance = initial_balance
        self.positions = {}
        self.trade_count = 0
        self.wins = 0
        self.losses = 0
        self.commission_per_share = 0.005
        
        self.api = tradeapi.REST(
            ALPACA_API_KEY,
            ALPACA_API_SECRET,
            base_url='https://paper-api.alpaca.markets'
        )
        
        send_telegram(f"📈 <b>Alpaca Stock Trader Aktif</b>\nBakiye: ${initial_balance:.2f}")

    def is_market_open(self):
        clock = self.api.get_clock()
        return clock.is_open

    def execute_trade(self, symbol, signal, current_price):
        if symbol in self.positions:
            return False, f"⚠️ {symbol} için zaten pozisyon var"
            
        if not self.is_market_open():
            return False, "⚠️ Piyasa kapalı"
            
        risk_amount = self.balance * RISK_PER_TRADE
        shares = int(risk_amount // current_price)
        
        if shares < 1:
            return False, "⚠️ Yetersiz bakiye"
            
        try:
            if signal == 'long':
                order = self.api.submit_order(
                    symbol=symbol,
                    qty=shares,
                    side='buy',
                    type='market',
                    time_in_force='gtc'
                )
            else:
                order = self.api.submit_order(
                    symbol=symbol,
                    qty=shares,
                    side='sell',
                    type='market',
                    time_in_force='gtc'
                )
        except Exception as e:
            return False, f"⚠️ Emir hatası: {str(e)}"
        
        # Pozisyonu kaydet
        self.positions[symbol] = {
            'direction': signal,
            'entry_price': current_price,
            'shares': shares,
            'entry_time': time.time(),
            'order_id': order.id
        }
        
        commission = shares * self.commission_per_share
        self.balance -= commission
        self.trade_count += 1
        
        msg = (f"🚀 <b>YENİ HİSSE İŞLEMİ</b>\n"
               f"▫️ Sembol: {symbol}\n"
               f"▫️ Yön: {signal.upper()}\n"
               f"▫️ Giriş: ${current_price:.2f}\n"
               f"▫️ Hisse: {shares}")
        send_telegram(msg)
        
        return True, "✅ Pozisyon açıldı"

    # ... (check_exit, close_position fonksiyonları) ...

def get_alpaca_data(api, symbol, timeframe='5Min', limit=100):
    try:
        bars = api.get_bars(symbol, timeframe, limit=limit).df
        if bars.empty:
            return None, None
        return bars, bars['close'].iloc[-1]
    except Exception as e:
        logger.error(f"Alpaca data error ({symbol}): {str(e)}")
        return None, None

def generate_signal(df):
    try:
        if df is None or len(df) < 50:
            return None
            
        # EMA
        df['ema_fast'] = ta.ema(df['close'], EMA_FAST)
        df['ema_slow'] = ta.ema(df['close'], EMA_SLOW)
        
        # RSI
        df['rsi'] = ta.rsi(df['close'], RSI_PERIOD)
        
        # Hacim
        df['volume_ma'] = ta.sma(df['volume'], 20)
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        last = df.iloc[-1]
        prev = df.iloc[-2]
        
        long_condition = (
            last['ema_fast'] > last['ema_slow'] and
            prev['ema_fast'] <= prev['ema_slow'] and
            last['rsi'] < RSI_OVERBOUGHT and
            last['volume_ratio'] > VOLUME_MULTIPLIER
        )
        
        short_condition = (
            last['ema_fast'] < last['ema_slow'] and
            prev['ema_fast'] >= prev['ema_slow'] and
            last['rsi'] > RSI_OVERSOLD and
            last['volume_ratio'] > VOLUME_MULTIPLIER
        )
        
        if long_condition:
            return 'long'
        elif short_condition:
            return 'short'
            
        return None
        
    except Exception as e:
        logger.error(f"Sinyal hatası: {str(e)}")
        return None

def main():
    logger.info("=== ALPACA STOCK TRADER BAŞLATILDI ===")
    trader = AlpacaStockTrader(INITIAL_BALANCE)
    last_check = {symbol: 0 for symbol in SYMBOLS}
    
    while True:
        try:
            if not trader.is_market_open():
                logger.info("Piyasa kapalı, 1 saat bekleniyor...")
                time.sleep(3600)
                continue
                
            for symbol in SYMBOLS:
                if time.time() - last_check[symbol] < 60:
                    continue
                    
                df, current_price = get_alpaca_data(trader.api, symbol, TIMEFRAME)
                if df is None or current_price is None:
                    continue
                    
                # Pozisyon kontrolü
                exit_status, profit, msg = trader.check_exit(symbol, current_price)
                
                # Yeni sinyal
                signal = generate_signal(df)
                if signal and symbol not in trader.positions:
                    status, msg = trader.execute_trade(symbol, signal, current_price)
                
                last_check[symbol] = time.time()
                time.sleep(5)
                
            time.sleep(10)
            
        except Exception as e:
            logger.error(f"Ana hata: {str(e)}")
            time.sleep(30)

if __name__ == "__main__":
    main()