import time
import logging
import ccxt
import pandas as pd
import numpy as np
import pandas_ta as ta
from datetime import datetime, timedelta

# ========== KONFİGÜRASYON (DEĞİŞTİRİN) ==========
BINANCE_API_KEY = "k8Sx3Y27lRIWBJVZ4q9bH65v5p0L9M3dccPpMF7OY8UKke9yPhKfwol3WXTBnuEy"
BINANCE_API_SECRET = "r996pp43QLOEhLtXidl49qTGVMkwKDlalJsVf3PiRI6ix1FrJLpJbBkrg8Tr3Cyt"
SYMBOL = "BTC/USDT"
INITIAL_BALANCE = 100.0  # Başlangıç bakiyesi ($)
RISK_PER_TRADE = 0.01    # İşlem başı risk (%1)
# =================================================

# Strateji Parametreleri (TEST EDİLMİŞ)
EMA_FAST = 5
EMA_SLOW = 13
RSI_PERIOD = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
VOLUME_MULTIPLIER = 1.2
MIN_PROFIT_TARGET = 0.008  # %0.8
MAX_LOSS = 0.005           # %0.5
TRADE_EXPIRY = 300         # 5 dakika (saniye)

# Log Ayarları
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger()

class QuantumTrader:
    def __init__(self, initial_balance):
        self.balance = initial_balance
        self.position = None
        self.entry_price = 0.0
        self.position_size = 0.0
        self.entry_time = None
        self.trade_count = 0
        self.wins = 0
        self.losses = 0
        self.commission_rate = 0.001  # %0.1 Binance komisyonu
        self.spread_factor = 1.00015  # %0.015 spread
        
    def execute_trade(self, signal, current_price):
        if self.position:
            return 0, "Zaten pozisyon var"
            
        # Risk hesapla
        risk_amount = self.balance * RISK_PER_TRADE
        
        # Pozisyon büyüklüğü
        self.position_size = risk_amount / current_price
        
        # Spread uygula
        if signal == 'long':
            self.entry_price = current_price * self.spread_factor
        else:
            self.entry_price = current_price / self.spread_factor
            
        # Pozisyonu aç
        self.position = signal
        self.entry_time = time.time()
        self.trade_count += 1
        
        # Komisyon öde
        commission = self.position_size * self.entry_price * self.commission_rate
        self.balance -= commission
        
        return 1, f"{signal.upper()} pozisyon açıldı | Giriş: ${self.entry_price:.2f}"
    
    def check_exit(self, current_price):
        if not self.position:
            return False, 0, "Pozisyon yok"
            
        current_time = time.time()
        hold_time = current_time - self.entry_time
        
        # Zaman aşımı
        if hold_time > TRADE_EXPIRY:
            return self.close_position(current_price), "Zaman aşımı"
            
        # Kar/zarar hesapla
        if self.position == 'long':
            profit_ratio = (current_price - self.entry_price) / self.entry_price
            if profit_ratio >= MIN_PROFIT_TARGET:
                return self.close_position(current_price), f"Kar hedefi (%{MIN_PROFIT_TARGET*100:.2f})"
            elif profit_ratio <= -MAX_LOSS:
                return self.close_position(current_price), f"Zarar durdur (%{MAX_LOSS*100:.2f})"
        else:
            profit_ratio = (self.entry_price - current_price) / self.entry_price
            if profit_ratio >= MIN_PROFIT_TARGET:
                return self.close_position(current_price), f"Kar hedefi (%{MIN_PROFIT_TARGET*100:.2f})"
            elif profit_ratio <= -MAX_LOSS:
                return self.close_position(current_price), f"Zarar durdur (%{MAX_LOSS*100:.2f})"
                
        return False, 0, "Beklemede"
    
    def close_position(self, current_price):
        # Spread uygula
        if self.position == 'long':
            exit_price = current_price / self.spread_factor
            profit = (exit_price - self.entry_price) * self.position_size
        else:
            exit_price = current_price * self.spread_factor
            profit = (self.entry_price - exit_price) * self.position_size
            
        # Komisyon öde
        commission = self.position_size * exit_price * self.commission_rate
        net_profit = profit - commission
        
        # Bakiyeyi güncelle
        self.balance += net_profit
        
        # İstatistik
        if net_profit > 0:
            self.wins += 1
        else:
            self.losses += 1
            
        # Pozisyonu sıfırla
        self.position = None
        self.entry_price = 0.0
        self.position_size = 0.0
        
        return True, net_profit
        
    def get_performance(self):
        win_rate = self.wins / self.trade_count * 100 if self.trade_count > 0 else 0
        return {
            "balance": self.balance,
            "trades": self.trade_count,
            "win_rate": win_rate,
            "wins": self.wins,
            "losses": self.losses
        }

def get_binance_data(symbol, timeframe='5m', limit=100):
    """Binance'dan veri çek"""
    exchange = ccxt.binance({
        'enableRateLimit': True,
        'options': {'adjustForTimeDifference': True}
    })
    
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    except Exception as e:
        logger.error(f"Veri çekme hatası: {str(e)}")
        return None

def generate_signal(df):
    """Sinyal üret"""
    if df is None or len(df) < 50:
        return None
        
    try:
        # Göstergeleri hesapla
        df['ema_fast'] = ta.ema(df['close'], EMA_FAST)
        df['ema_slow'] = ta.ema(df['close'], EMA_SLOW)
        df['rsi'] = ta.rsi(df['close'], RSI_PERIOD)
        df['volume_ma'] = ta.sma(df['volume'], 20)
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        # Son mum
        last = df.iloc[-1]
        prev = df.iloc[-2]
        
        # 1. EMA Crossover
        ema_cross_up = last['ema_fast'] > last['ema_slow'] and prev['ema_fast'] <= prev['ema_slow']
        ema_cross_down = last['ema_fast'] < last['ema_slow'] and prev['ema_fast'] >= prev['ema_slow']
        
        # 2. RSI Filtresi
        rsi_ok_long = last['rsi'] < RSI_OVERBOUGHT
        rsi_ok_short = last['rsi'] > RSI_OVERSOLD
        
        # 3. Hacim Filtresi
        volume_ok = last['volume_ratio'] >= VOLUME_MULTIPLIER
        
        # Sinyal Koşulları
        if ema_cross_up and rsi_ok_long and volume_ok:
            return 'long'
        elif ema_cross_down and rsi_ok_short and volume_ok:
            return 'short'
            
        return None
        
    except Exception as e:
        logger.error(f"Sinyal üretme hatası: {str(e)}")
        return None

def main():
    logger.info("=== QUANTUM TRADER v2.0 BAŞLATILIYOR ===")
    logger.info(f"Başlangıç Bakiyesi: ${INITIAL_BALANCE:.2f}")
    
    trader = QuantumTrader(INITIAL_BALANCE)
    last_trade_time = 0
    min_trade_interval = 60  # 60 saniye
    
    while True:
        try:
            # 1. Piyasa verilerini al
            df = get_binance_data(SYMBOL)
            if df is None:
                time.sleep(30)
                continue
                
            # 2. Sinyal üret
            signal = generate_signal(df)
            current_price = df['close'].iloc[-1]
            current_time = time.time()
            
            # 3. Pozisyon kontrolü
            exit_status, profit, msg = trader.check_exit(current_price)
            if exit_status:
                logger.info(f"POZİSYON KAPATILDI: {msg} | Kar: ${profit:.4f}")
            
            # 4. Yeni işlem için uygunluk kontrolü
            if (signal and 
                trader.position is None and 
                current_time - last_trade_time > min_trade_interval):
                
                status, msg = trader.execute_trade(signal, current_price)
                if status:
                    logger.info(msg)
                    last_trade_time = current_time
                    
            # 5. Performans raporu (her 10 dakikada)
            if int(time.time()) % 600 == 0:
                perf = trader.get_performance()
                logger.info(
                    f"PERFORMANS | Bakiye: ${perf['balance']:.2f} | "
                    f"İşlem: {perf['trades']} | Başarı: {perf['win_rate']:.1f}%"
                )
            
            # 6. Bekle
            time.sleep(15)
            
        except Exception as e:
            logger.error(f"Ana döngü hatası: {str(e)}")
            time.sleep(60)

if __name__ == "__main__":
    main()