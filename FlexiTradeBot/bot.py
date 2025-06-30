from binance.client import Client
from binance.enums import *
import pandas as pd
import numpy as np
import time
from datetime import datetime
import config
from colorama import init, Fore, Back, Style
import math
import os
from dotenv import load_dotenv
import ta
import threading

# Colorama'yı başlat
init(autoreset=True)

class FlexiTradeBot:
    def __init__(self):
        """Bot'u başlat"""
        # API anahtarlarını yükle
        load_dotenv()
        api_key = os.getenv('BINANCE_API_KEY')
        api_secret = os.getenv('BINANCE_API_SECRET')
        
        if not api_key or not api_secret:
            raise ValueError("API anahtarları bulunamadı!")
        
        # Binance client'ı başlat
        self.client = Client(api_key, api_secret)
        
        # Pozisyon kilidi ve sayacı
        self.position_lock = threading.Lock()
        self.position_counter = 0
        
        # Pozisyon takibi için sözlük
        self.positions = {}
        
        # İşlem geçmişi
        self.trade_history = []  
        
        # Toplam kâr/zarar
        self.total_pnl = 0  
        
        # Margin hesabını kontrol et
        if config.TRADE_MODE == "MARGIN":
            # Margin hesabını etkinleştir
            try:
                self.client.enable_isolated_margin_account()
                print(f"{Fore.GREEN}Izole margin hesabı etkinleştirildi{Style.RESET_ALL}")
            except:
                print(f"{Fore.YELLOW}Izole margin hesabı zaten etkin{Style.RESET_ALL}")
        
        print(f"{Fore.GREEN}Bot başlatıldı!{Style.RESET_ALL}")

    def log_trade(self, symbol, side, entry_price, exit_price=None, amount=None, pnl=None, status="OPEN", reason=""):
        """İşlem geçmişine kaydet"""
        trade = {
            'symbol': symbol,
            'side': side,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'amount': amount,
            'pnl': pnl,
            'status': status,
            'reason': reason,
            'timestamp': datetime.now()
        }
        self.trade_history.append(trade)
        
        # İşlem detaylarını göster
        if status == "OPEN":
            print(f"\n{Fore.GREEN}[{symbol}] Yeni İşlem Açıldı:")
            print(f"  Yön: {side}")
            print(f"  Giriş: {entry_price:.8f}")
            print(f"  Miktar: {amount:.4f}")
            print(f"  Stop Loss: {self.positions[symbol]['stop_loss']:.8f}")
            print(f"  Take Profit: {self.positions[symbol]['take_profit']:.8f}{Style.RESET_ALL}")
        else:
            color = Fore.GREEN if pnl > 0 else Fore.RED
            print(f"\n{color}[{symbol}] İşlem Kapatıldı:")
            print(f"  Yön: {side}")
            print(f"  Giriş: {entry_price:.8f}")
            print(f"  Çıkış: {exit_price:.8f}")
            print(f"  Miktar: {amount:.4f}")
            print(f"  PNL: {pnl:.2f} USDT")
            print(f"  Neden: {reason}{Style.RESET_ALL}")
            
            # Toplam kâr/zararı güncelle
            self.total_pnl += pnl
            
    def show_trade_summary(self):
        """İşlem özetini göster"""
        print(f"\n{Fore.CYAN}{Style.BRIGHT}{'='*20} İşlem Özeti {'='*20}")
        print(f"Toplam PNL: {self.total_pnl:.2f} USDT")
        
        # Son 5 işlemi göster
        print(f"\nSon İşlemler:")
        for trade in sorted(self.trade_history, key=lambda x: x['timestamp'], reverse=True)[:5]:
            color = Fore.GREEN if trade['pnl'] and trade['pnl'] > 0 else Fore.RED
            print(f"{color}{trade['timestamp'].strftime('%H:%M:%S')} {trade['symbol']} {trade['side']}")
            print(f"  {'Kapalı' if trade['status'] == 'CLOSED' else 'Açık'} | {trade['pnl']:.2f} USDT | {trade['reason']}{Style.RESET_ALL}")
        
        print(f"{Fore.CYAN}{Style.BRIGHT}{'='*50}{Style.RESET_ALL}\n")
        
    def get_futures_balance(self):
        try:
            account = self.client.futures_account()
            total_balance = float(account['totalWalletBalance'])
            available_balance = float(account['availableBalance'])
            print(f"{Fore.GREEN}Toplam Bakiye: {Style.BRIGHT}{total_balance} USDT")
            print(f"{Fore.GREEN}Kullanılabilir Bakiye: {Style.BRIGHT}{available_balance} USDT")
            return total_balance, available_balance
        except Exception as e:
            print(f"{Fore.RED}{Style.BRIGHT}Bakiye bilgisi alınamadı: {str(e)}{Style.RESET_ALL}")
            return None, None

    def get_symbol_info(self, symbol):
        """Futures symbol bilgilerini al"""
        try:
            info = self.client.futures_exchange_info()
            for s in info['symbols']:
                if s['symbol'] == symbol:
                    return s
            return None
        except Exception as e:
            print(f"{Fore.RED}Symbol bilgisi alınamadı: {str(e)}{Style.RESET_ALL}")
            return None

    def calculate_position_size(self, symbol, price, leverage):
        """İşlem büyüklüğünü hesapla"""
        try:
            # Bakiyeyi al
            total_balance, available_balance = self.get_futures_balance()
            if not total_balance or not available_balance:
                return None

            # Kullanılabilir USDT miktarı (bakiyenin %95'ini kullan)
            usdt_size = min(total_balance, available_balance) * 0.95
            
            # Minimum işlem kontrolü
            if usdt_size < config.MIN_TRADE_AMOUNT:
                print(f"{Fore.YELLOW}[{symbol}] Yetersiz bakiye: {usdt_size:.2f} < {config.MIN_TRADE_AMOUNT}{Style.RESET_ALL}")
                return None

            # Kaldıraçlı işlem büyüklüğünü hesapla
            leveraged_size = usdt_size * leverage
            
            # Coin miktarını hesapla
            quantity = leveraged_size / price
            
            # Symbol bilgilerini al
            symbol_info = self.get_symbol_info(symbol)
            if not symbol_info:
                return None

            # Quantity precision (miktar hassasiyeti)
            quantity_precision = int(symbol_info['quantityPrecision'])
            
            # Miktarı yuvarla
            quantity = round(quantity, quantity_precision)
            
            # Son kontroller
            actual_size = (quantity * price) / leverage
            if actual_size < config.MIN_TRADE_AMOUNT:
                print(f"{Fore.YELLOW}[{symbol}] İşlem büyüklüğü çok küçük: {actual_size:.2f} USDT{Style.RESET_ALL}")
                return None

            print(f"\n{Fore.CYAN}İşlem Detayları:")
            print(f"Kullanılabilir Bakiye: {usdt_size:.2f} USDT")
            print(f"Kaldıraç: {leverage}x")
            print(f"İşlem Miktarı: {quantity} {symbol}")
            print(f"Gerçek Değer: {actual_size:.2f} USDT")
            print(f"Kaldıraçlı Değer: {(actual_size * leverage):.2f} USDT{Style.RESET_ALL}")

            return quantity

        except Exception as e:
            print(f"{Fore.RED}[{symbol}] İşlem büyüklüğü hesaplanamadı: {str(e)}{Style.RESET_ALL}")
            return None

    def calculate_quantity(self, symbol, usdt_amount, price):
        """Futures için coin miktarını hesapla"""
        try:
            symbol_info = self.get_symbol_info(symbol)
            if not symbol_info:
                return None
            
            # Quantity precision (miktar hassasiyeti)
            quantity_precision = symbol_info['quantityPrecision']
            
            # Price precision (fiyat hassasiyeti)
            price_precision = symbol_info['pricePrecision']
            
            # USDT miktarını coin miktarına çevir
            quantity = usdt_amount / price
            
            # Hassasiyete göre yuvarla
            quantity = round(quantity, quantity_precision)
            
            # Minimum ve maksimum kontrolleri
            min_qty = float(symbol_info.get('minQty', 0))
            max_qty = float(symbol_info.get('maxQty', float('inf')))
            
            if quantity < min_qty:
                print(f"{Fore.YELLOW}[{symbol}] Minimum miktar: {min_qty}{Style.RESET_ALL}")
                return None
            
            if quantity > max_qty:
                print(f"{Fore.YELLOW}[{symbol}] Maksimum miktar: {max_qty}{Style.RESET_ALL}")
                return None
            
            return quantity
            
        except Exception as e:
            print(f"{Fore.RED}Miktar hesaplanamadı: {str(e)}{Style.RESET_ALL}")
            return None

    def get_all_symbols(self):
        """Taranacak coin listesini döndür"""
        try:
            # Sadece config'de belirtilen coinleri tara
            return config.SCAN_SYMBOLS
            
        except Exception as e:
            print(f"{Fore.RED}Coin listesi alınamadı: {str(e)}{Style.RESET_ALL}")
            return []

    def get_volatility(self, symbol, period='1h', window=24):
        """Volatilite hesaplama"""
        try:
            klines = self.client.futures_klines(symbol=symbol, interval=period, limit=window)
            closes = pd.Series([float(k[4]) for k in klines])
            returns = closes.pct_change().dropna()
            volatility = returns.std() * 100  # Yüzde cinsinden volatilite
            return volatility
        except Exception as e:
            print(f"{Fore.RED}[{symbol}] Volatilite hesaplanamadı: {str(e)}{Style.RESET_ALL}")
            return 5.0  # Varsayılan değer
            
    def calculate_optimal_leverage(self, symbol):
        """Volatiliteye göre optimize edilmiş kaldıraç hesaplama"""
        try:
            # Son 24 saatlik volatiliteyi hesapla
            volatility = self.get_volatility(symbol)
            
            # Volatilite bazlı kaldıraç seçimi
            selected_leverage = config.DEFAULT_LEVERAGE  # Varsayılan kaldıraç
            
            for vol_threshold, lev in sorted(config.VOLATILITY_LEVERAGE_RANGES.items()):
                if volatility <= vol_threshold:
                    selected_leverage = lev
                    break
            
            print(f"\n{Fore.CYAN}[{symbol}] Kaldıraç Analizi:")
            print(f"24s Volatilite: {volatility:.2f}%")
            print(f"Seçilen Kaldıraç: {selected_leverage}x{Style.RESET_ALL}")
            
            return selected_leverage
            
        except Exception as e:
            print(f"{Fore.RED}[{symbol}] Kaldıraç hesaplanamadı: {str(e)}{Style.RESET_ALL}")
            return config.DEFAULT_LEVERAGE  # Hata durumunda varsayılan kaldıraç

    def calculate_bollinger_bands(self, closes, period=20, num_std=2):
        sma = closes.rolling(window=period).mean()
        std = closes.rolling(window=period).std()
        upper = sma + (std * num_std)
        lower = sma - (std * num_std)
        return upper, sma, lower

    def calculate_macd(self, closes, fast=12, slow=26, signal=9):
        exp1 = closes.ewm(span=fast, adjust=False).mean()
        exp2 = closes.ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        return macd, signal_line

    def calculate_ema(self, closes, period):
        """Exponential Moving Average hesaplama"""
        return closes.ewm(span=period, adjust=False).mean()
        
    def calculate_supertrend(self, df, period=10, multiplier=3):
        """SuperTrend hesaplama"""
        hl2 = (df['high'] + df['low']) / 2
        atr = df['high'].sub(df['low']).rolling(period).mean()
        
        basic_upperband = hl2 + (multiplier * atr)
        basic_lowerband = hl2 - (multiplier * atr)
        
        final_upperband = basic_upperband.copy()
        final_lowerband = basic_lowerband.copy()
        
        for i in range(period, len(df)):
            if basic_upperband[i] < final_upperband[i-1] and df['close'][i-1] <= final_upperband[i-1]:
                final_upperband[i] = basic_upperband[i]
            else:
                final_upperband[i] = final_upperband[i-1]
                
            if basic_lowerband[i] > final_lowerband[i-1] and df['close'][i-1] >= final_lowerband[i-1]:
                final_lowerband[i] = basic_lowerband[i]
            else:
                final_lowerband[i] = final_lowerband[i-1]
        
        supertrend = pd.Series(index=df.index)
        for i in range(period, len(df)):
            if supertrend[i-1] == final_upperband[i-1] and df['close'][i] <= final_upperband[i]:
                supertrend[i] = final_upperband[i]
            elif supertrend[i-1] == final_upperband[i-1] and df['close'][i] > final_upperband[i]:
                supertrend[i] = final_lowerband[i]
            elif supertrend[i-1] == final_lowerband[i-1] and df['close'][i] >= final_lowerband[i]:
                supertrend[i] = final_lowerband[i]
            elif supertrend[i-1] == final_lowerband[i-1] and df['close'][i] < final_lowerband[i]:
                supertrend[i] = final_upperband[i]
        
        return supertrend
        
    def calculate_stoch_rsi(self, closes, period=14, smooth_k=3, smooth_d=3):
        """Stochastic RSI hesaplama"""
        # Önce normal RSI hesapla
        delta = closes.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        # Stochastic RSI hesapla
        stoch_rsi = (rsi - rsi.rolling(period).min()) / (rsi.rolling(period).max() - rsi.rolling(period).min())
        k = stoch_rsi.rolling(smooth_k).mean() * 100
        d = k.rolling(smooth_d).mean()
        return k, d
        
    def calculate_obv(self, df):
        """On Balance Volume hesaplama"""
        obv = pd.Series(index=df.index)
        obv[0] = 0
        
        for i in range(1, len(df)):
            if df['close'][i] > df['close'][i-1]:
                obv[i] = obv[i-1] + df['volume'][i]
            elif df['close'][i] < df['close'][i-1]:
                obv[i] = obv[i-1] - df['volume'][i]
            else:
                obv[i] = obv[i-1]
        
        return obv
        
    def calculate_vwap(self, df, period=14):
        """Volume Weighted Average Price hesaplama"""
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        vwap = (typical_price * df['volume']).rolling(period).sum() / df['volume'].rolling(period).sum()
        return vwap
        
    def calculate_opportunity_score(self, df, current_price, signals, signal_strength):
        """Fırsat skoru hesaplama (0-100)"""
        score = 0
        
        try:
            # Trend Analizi (0-30 puan)
            ema_fast = self.calculate_ema(df['close'], config.TREND_EMA_FAST)
            ema_slow = self.calculate_ema(df['close'], config.TREND_EMA_SLOW)
            supertrend = self.calculate_supertrend(df, config.TREND_SUPERTREND_PERIOD, config.TREND_SUPERTREND_MULTIPLIER)
            
            # EMA trend puanı
            if ema_fast.iloc[-1] > ema_slow.iloc[-1]:
                if current_price > ema_fast.iloc[-1]:  # Güçlü yukarı trend
                    score += 15
                else:  # Zayıf yukarı trend
                    score += 10
            else:
                if current_price < ema_fast.iloc[-1]:  # Güçlü aşağı trend
                    score += 15
                else:  # Zayıf aşağı trend
                    score += 10
                    
            # SuperTrend puanı
            if current_price > supertrend.iloc[-1]:
                score += 15
            
            # Momentum Analizi (0-30 puan)
            stoch_k, stoch_d = self.calculate_stoch_rsi(df['close'])
            
            # Stochastic RSI puanı
            if stoch_k.iloc[-1] < config.STOCH_RSI_OVERSOLD:
                score += 15
            elif stoch_k.iloc[-1] > config.STOCH_RSI_OVERBOUGHT:
                score += 15
                
            # Sinyal gücü puanı
            score += min(15, signal_strength * 10)
            
            # Hacim Analizi (0-40 puan)
            obv = self.calculate_obv(df)
            vwap = self.calculate_vwap(df)
            
            # OBV trend puanı
            obv_sma = obv.rolling(config.OBV_TREND_PERIOD).mean()
            if obv.iloc[-1] > obv_sma.iloc[-1]:
                score += 20
            
            # VWAP pozisyon puanı
            if len(signals) > 0:  # Sinyal varsa
                if signals[0][0] == 'LONG' and current_price < vwap.iloc[-1]:
                    score += 20  # VWAP altında LONG fırsatı
                elif signals[0][0] == 'SHORT' and current_price > vwap.iloc[-1]:
                    score += 20  # VWAP üstünde SHORT fırsatı
            
            print(f"{Fore.MAGENTA}Fırsat Skoru: {score}/100{Style.RESET_ALL}")
            return score
            
        except Exception as e:
            print(f"{Fore.RED}Fırsat skoru hesaplanamadı: {str(e)}{Style.RESET_ALL}")
            return 0

    def analyze_market(self, symbol):
        """Piyasa analizi yap ve sinyal üret"""
        try:
            # Mum verilerini al
            klines = self.client.futures_klines(symbol=symbol, interval=Client.KLINE_INTERVAL_1MINUTE, limit=100)
            df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time',
                                             'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume',
                                             'taker_buy_quote_asset_volume', 'ignore'])
            
            # Veri tiplerini dönüştür
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Veri kontrolü
            if df['close'].isna().any() or df['volume'].isna().any():
                print(f"[{symbol}] Veri eksik, analiz atlanıyor...")
                return None, None, 0
            
            if df['volume'].sum() == 0 or len(df['close'].unique()) < 10:
                print(f"[{symbol}] Yetersiz veri veya sıfır hacim, analiz atlanıyor...")
                return None, None, 0
            
            # Teknik göstergeleri hesapla
            current_price = float(df['close'].iloc[-1])
            
            # RSI
            rsi = ta.momentum.rsi(df['close'], window=config.RSI_PERIOD)
            current_rsi = float(rsi.iloc[-1])
            
            # MACD
            macd = ta.trend.macd_diff(df['close'])
            current_macd = float(macd.iloc[-1])
            
            # Bollinger Bands
            bb = ta.volatility.BollingerBands(df['close'], window=config.BB_PERIOD, window_dev=config.BB_STD)
            bb_upper = float(bb.bollinger_hband().iloc[-1])
            bb_middle = float(bb.bollinger_mavg().iloc[-1])
            bb_lower = float(bb.bollinger_lband().iloc[-1])
            
            # EMA'lar
            ema20 = ta.trend.ema_indicator(df['close'], window=20)
            ema50 = ta.trend.ema_indicator(df['close'], window=50)
            ema200 = ta.trend.ema_indicator(df['close'], window=200)
            
            # Trend analizi
            trend = "DOWN" if ema20.iloc[-1] < ema50.iloc[-1] and ema50.iloc[-1] < ema200.iloc[-1] else "UP"
            
            # Hacim analizi
            volume_sma = df['volume'].rolling(window=20).mean()
            volume_ratio = df['volume'].iloc[-1] / volume_sma.iloc[-1] if volume_sma.iloc[-1] > 0 else 0
            
            print(f"\n[{symbol}] Teknik Analiz:")
            print(f"Fiyat: {current_price:.8f}")
            print(f"RSI: {current_rsi:.2f}")
            print(f"MACD: {current_macd:.8f}")
            print(f"BB Üst: {bb_upper:.8f}")
            print(f"BB Orta: {bb_middle:.8f}")
            print(f"BB Alt: {bb_lower:.8f}")
            print(f"Trend: {trend}")
            print(f"Hacim Oranı: {volume_ratio:.2f}x")
            
            # Fırsat skoru hesapla
            long_score = 0
            short_score = 0
            signal = None
            
            # LONG Sinyalleri
            if trend == "UP":
                # RSI bazlı
                if current_rsi <= config.RSI_OVERSOLD:
                    long_score += 30
                
                # MACD bazlı
                if current_macd > 0:
                    long_score += 20
                
                # BB bazlı
                if current_price <= bb_lower:
                    long_score += 25
                
                # Hacim bazlı
                if volume_ratio >= config.MIN_VOLUME_RATIO:
                    long_score += 25
            
            # SHORT Sinyalleri
            elif trend == "DOWN":
                # RSI bazlı
                if current_rsi >= config.RSI_OVERBOUGHT:
                    short_score += 30
                
                # MACD bazlı
                if current_macd < 0:
                    short_score += 20
                
                # BB bazlı
                if current_price >= bb_upper:
                    short_score += 25
                
                # Hacim bazlı
                if volume_ratio >= config.MIN_VOLUME_RATIO:
                    short_score += 25
                
                # Ek SHORT sinyalleri
                if current_price > bb_upper * 1.002:  # Üst bandın %0.2 üstünde
                    short_score += 10
                if current_rsi > 70:  # Aşırı alım
                    short_score += 10
                if ema20.iloc[-1] < ema20.iloc[-2]:  # Kısa vadeli düşüş
                    short_score += 10
            
            # En yüksek skoru al
            final_score = max(long_score, short_score)
            
            # Sinyal üret
            if final_score >= config.MIN_OPPORTUNITY_SCORE:
                signal = "LONG" if long_score > short_score else "SHORT"
                print(f"Sinyal: {signal} (Skor: {final_score})")
            else:
                print(f"Sinyal yok (Skor: {final_score})")
            
            return signal, current_price, final_score
            
        except Exception as e:
            print(f"[{symbol}] Analiz hatası: {str(e)}")
            return None, None, 0

    def get_account_info(self):
        try:
            account = self.client.futures_account()
            total_balance = float(account['totalWalletBalance'])
            available_balance = float(account['availableBalance'])
            positions = {p['symbol']: p for p in account['positions']}
            return {
                'total_balance': total_balance,
                'available_balance': available_balance,
                'positions': positions
            }
        except Exception as e:
            print(f"{Fore.RED}Hesap bilgileri alınamadı: {str(e)}{Style.RESET_ALL}")
            return None

    def place_order(self, symbol, side, quantity, current_price):
        """Futures emir yerleştir"""
        try:
            if not quantity:
                print(f"{Fore.RED}[{symbol}] Geçersiz işlem miktarı{Style.RESET_ALL}")
                return False

            # Kaldıracı ayarla
            leverage = self.calculate_optimal_leverage(symbol)
            self.client.futures_change_leverage(symbol=symbol, leverage=leverage)
            
            # Marjin tipini ayarla (ISOLATED)
            try:
                self.client.futures_change_margin_type(symbol=symbol, marginType='ISOLATED')
            except:
                pass  # Zaten ISOLATED ise hata verebilir
            
            # Stop loss ve take profit hesapla
            if side == "BUY":
                stop_loss = round(current_price * (1 - config.STOP_LOSS_PERCENTAGE / 100), 8)
                take_profit_1 = round(current_price * (1 + config.TAKE_PROFIT_LEVELS[50] / 100), 8)
                take_profit_2 = round(current_price * (1 + config.TAKE_PROFIT_LEVELS[100] / 100), 8)
            else:
                stop_loss = round(current_price * (1 + config.STOP_LOSS_PERCENTAGE / 100), 8)
                take_profit_1 = round(current_price * (1 - config.TAKE_PROFIT_LEVELS[50] / 100), 8)
                take_profit_2 = round(current_price * (1 - config.TAKE_PROFIT_LEVELS[100] / 100), 8)

            # Ana pozisyonu aç
            order = self.client.futures_create_order(
                symbol=symbol,
                side=side,
                type='MARKET',
                quantity=quantity
            )

            # Stop loss emri
            sl_order = self.client.futures_create_order(
                symbol=symbol,
                side='SELL' if side == 'BUY' else 'BUY',
                type='STOP_MARKET',
                stopPrice=stop_loss,
                closePosition=True,
                timeInForce='GTC'
            )

            # İlk take profit (%50)
            tp1_qty = round(quantity * 0.5, 8)
            tp1_order = self.client.futures_create_order(
                symbol=symbol,
                side='SELL' if side == 'BUY' else 'BUY',
                type='LIMIT',
                timeInForce='GTC',
                price=take_profit_1,
                quantity=tp1_qty
            )

            # İkinci take profit (%50)
            tp2_order = self.client.futures_create_order(
                symbol=symbol,
                side='SELL' if side == 'BUY' else 'BUY',
                type='LIMIT',
                timeInForce='GTC',
                price=take_profit_2,
                quantity=tp1_qty
            )

            # Trailing stop (kârda ise)
            if config.TRAILING_STOP:
                activation_price = round(current_price * (1 + config.TRAILING_STOP_ACTIVATION / 100), 8) if side == 'BUY' else round(current_price * (1 - config.TRAILING_STOP_ACTIVATION / 100), 8)
                
                ts_order = self.client.futures_create_order(
                    symbol=symbol,
                    side='SELL' if side == 'BUY' else 'BUY',
                    type='TRAILING_STOP_MARKET',
                    callbackRate=config.TRAILING_STOP_DISTANCE,
                    activationPrice=activation_price,
                    quantity=quantity
                )

            print(f"\n{Fore.GREEN}[{symbol}] İşlem Detayları:")
            print(f"Yön: {side}")
            print(f"Giriş: {current_price}")
            print(f"Stop Loss: {stop_loss} (%{config.STOP_LOSS_PERCENTAGE})")
            print(f"Take Profit 1: {take_profit_1} (%{config.TAKE_PROFIT_LEVELS[50]})")
            print(f"Take Profit 2: {take_profit_2} (%{config.TAKE_PROFIT_LEVELS[100]})")
            print(f"Trailing Stop: Aktif (Aktivasyon: %{config.TRAILING_STOP_ACTIVATION}, Mesafe: %{config.TRAILING_STOP_DISTANCE})")
            print(f"Kaldıraç: {leverage}x")
            print(f"Miktar: {quantity}{Style.RESET_ALL}")
            
            return True
            
        except Exception as e:
            print(f"{Fore.RED}[{symbol}] Emir hatası: {str(e)}{Style.RESET_ALL}")
            return False

    def get_active_position_count(self):
        """Aktif pozisyon sayısını döndür"""
        with self.position_lock:
            try:
                positions = self.client.futures_position_information()
                active_count = len([p for p in positions if float(p['positionAmt']) != 0])
                print(f"{Fore.YELLOW}Aktif Pozisyon Sayısı: {active_count}{Style.RESET_ALL}")
                return active_count
            except Exception as e:
                print(f"{Fore.RED}Pozisyon sayısı alınamadı: {str(e)}{Style.RESET_ALL}")
                return config.MAX_OPEN_POSITIONS  # Hata durumunda güvenli tarafta kal

    def close_position(self, symbol):
        """Belirli bir pozisyonu kapat"""
        try:
            position = next((p for p in self.client.futures_position_information() if p['symbol'] == symbol), None)
            
            if position and float(position['positionAmt']) != 0:
                amount = float(position['positionAmt'])
                side = 'SELL' if amount > 0 else 'BUY'
                
                # Market emri ile pozisyonu kapat
                self.client.futures_create_order(
                    symbol=symbol,
                    side=side,
                    type='MARKET',
                    quantity=abs(amount),
                    reduceOnly=True
                )
                
                print(f"{Fore.GREEN}[{symbol}] Pozisyon kapatıldı{Style.RESET_ALL}")
                return True
                
        except Exception as e:
            print(f"{Fore.RED}[{symbol}] Pozisyon kapatılamadı: {str(e)}{Style.RESET_ALL}")
        
        return False

    def close_all_positions(self):
        """Tüm açık pozisyonları kapat"""
        try:
            positions = self.client.futures_position_information()
            closed_count = 0
            
            for position in positions:
                symbol = position['symbol']
                amount = float(position['positionAmt'])
                
                if amount == 0:  # Pozisyon zaten kapalı
                    continue
                    
                side = 'SELL' if amount > 0 else 'BUY'  # Pozisyonu kapatmak için ters yönde işlem
                abs_amount = abs(amount)
                
                try:
                    # Market emri ile pozisyonu kapat
                    self.client.futures_create_order(
                        symbol=symbol,
                        side=side,
                        type='MARKET',
                        quantity=abs_amount,
                        reduceOnly=True  # Sadece pozisyonu kapat, yeni pozisyon açma
                    )
                    
                    # İşlem geçmişine kaydet
                    entry_price = float(position['entryPrice'])
                    exit_price = float(position['markPrice'])
                    pnl = float(position['unRealizedProfit'])
                    
                    self.log_trade(
                        symbol=symbol,
                        side=side,
                        entry_price=entry_price,
                        exit_price=exit_price,
                        amount=abs_amount,
                        pnl=pnl,
                        status="CLOSED",
                        reason="Manuel Kapatma"
                    )
                    
                    closed_count += 1
                    print(f"{Fore.GREEN}[{symbol}] Pozisyon kapatıldı. PNL: {pnl:.2f} USDT{Style.RESET_ALL}")
                    
                except Exception as e:
                    print(f"{Fore.RED}[{symbol}] Pozisyon kapatılamadı: {str(e)}{Style.RESET_ALL}")
            
            print(f"\n{Fore.GREEN}Toplam {closed_count} pozisyon kapatıldı.{Style.RESET_ALL}")
            return closed_count
            
        except Exception as e:
            print(f"{Fore.RED}Pozisyonlar kapatılırken hata oluştu: {str(e)}{Style.RESET_ALL}")
            return 0

    def check_and_close_positions(self):
        """Kârdaki pozisyonları kontrol et ve kapat"""
        try:
            positions = self.client.futures_position_information()
            closed_positions = []
            
            for position in positions:
                symbol = position['symbol']
                amount = float(position['positionAmt'])
                
                if amount != 0:  # Açık pozisyon varsa
                    entry_price = float(position['entryPrice'])
                    current_price = float(position['markPrice'])
                    unrealized_pnl = float(position['unRealizedProfit'])
                    
                    # Kâr yüzdesini hesapla
                    if amount > 0:  # Long pozisyon
                        profit_percentage = ((current_price - entry_price) / entry_price) * 100
                        side = "LONG"
                    else:  # Short pozisyon
                        profit_percentage = ((entry_price - current_price) / entry_price) * 100
                        side = "SHORT"
                    
                    # Take Profit'e ulaşıldıysa pozisyonu kapat
                    if profit_percentage >= config.TAKE_PROFIT_PERCENTAGE:
                        close_side = 'SELL' if amount > 0 else 'BUY'
                        
                        # Market emirle pozisyonu kapat
                        self.client.futures_create_order(
                            symbol=symbol,
                            side=close_side,
                            type='MARKET',
                            quantity=abs(amount),
                            reduceOnly=True
                        )
                        
                        # Stop Loss ve Take Profit emirlerini iptal et
                        self.client.futures_cancel_all_open_orders(symbol=symbol)
                        closed_positions.append(symbol)
                        
                        # İşlem geçmişine kaydet
                        self.log_trade(
                            symbol=symbol,
                            side=side,
                            entry_price=entry_price,
                            exit_price=current_price,
                            amount=abs(amount),
                            pnl=unrealized_pnl,
                            status="CLOSED",
                            reason=f"Take Profit (%{profit_percentage:.1f})"
                        )
                            
                        print(f"{Fore.GREEN}[{symbol}] Kâr Alındı! Profit: {unrealized_pnl:.2f} USDT (%{profit_percentage:.1f}){Style.RESET_ALL}")
                        time.sleep(2)  # İşlemin yerleşmesini bekle
            
            # Kapatılan pozisyonları listeden çıkar
            for symbol in closed_positions:
                if symbol in self.positions:
                    del self.positions[symbol]
            
            return len(closed_positions)  # Kapatılan pozisyon sayısını döndür
                        
        except Exception as e:
            print(f"{Fore.RED}Pozisyon kontrolü sırasında hata: {str(e)}{Style.RESET_ALL}")
            return 0

    def strictly_check_position_limit(self):
        """Pozisyon limitini kesin olarak kontrol et"""
        try:
            positions = self.client.futures_position_information()
            active_positions = [p for p in positions if float(p['positionAmt']) != 0]
            active_count = len(active_positions)
            
            if active_count >= config.MAX_OPEN_POSITIONS:
                print(f"\n{Fore.RED}DİKKAT: Pozisyon limiti aşıldı!")
                print(f"Aktif Pozisyonlar ({active_count}):{Style.RESET_ALL}")
                
                # Tüm aktif pozisyonları göster
                for pos in active_positions:
                    symbol = pos['symbol']
                    amount = float(pos['positionAmt'])
                    pnl = float(pos['unRealizedProfit'])
                    print(f"{Fore.YELLOW}{symbol}: {abs(amount):.4f} | PNL: {pnl:.2f} USDT{Style.RESET_ALL}")
                
                # Fazla pozisyonları kapat
                if active_count > config.MAX_OPEN_POSITIONS:
                    excess = active_count - config.MAX_OPEN_POSITIONS
                    print(f"\n{Fore.RED}Fazla pozisyonlar kapatılıyor ({excess} adet)...{Style.RESET_ALL}")
                    
                    # PNL'e göre sırala ve en kötüleri kapat
                    sorted_positions = sorted(active_positions, key=lambda x: float(x['unRealizedProfit']))
                    
                    for pos in sorted_positions[:excess]:
                        symbol = pos['symbol']
                        amount = float(pos['positionAmt'])
                        side = 'SELL' if amount > 0 else 'BUY'
                        
                        try:
                            self.client.futures_create_order(
                                symbol=symbol,
                                side=side,
                                type='MARKET',
                                quantity=abs(amount),
                                reduceOnly=True
                            )
                            print(f"{Fore.GREEN}[{symbol}] Fazla pozisyon kapatıldı{Style.RESET_ALL}")
                            
                        except Exception as e:
                            print(f"{Fore.RED}[{symbol}] Pozisyon kapatılamadı: {str(e)}{Style.RESET_ALL}")
                
                return False
            
            return True
            
        except Exception as e:
            print(f"{Fore.RED}Pozisyon limiti kontrol edilemedi: {str(e)}{Style.RESET_ALL}")
            return False

    def run(self):
        """Bot'u çalıştır"""
        print(f"\n{Fore.CYAN}{Style.BRIGHT}{'='*50}")
        print(f"FlexiTradeBot Başlatıldı - {datetime.now()}")
        print(f"{'='*50}{Style.RESET_ALL}\n")
        
        while True:
            try:
                # ADIM 1: Mevcut pozisyonları kontrol et
                positions = self.client.futures_position_information()
                active_positions = [p for p in positions if float(p['positionAmt']) != 0]
                active_count = len(active_positions)
                
                print(f"\n{Fore.YELLOW}Aktif Pozisyon Kontrolü:")
                print(f"Pozisyon Sayısı: {active_count}/{config.MAX_OPEN_POSITIONS}{Style.RESET_ALL}")
                
                # ADIM 2: Yeni işlem açılabilir mi kontrol et
                if active_count >= config.MAX_OPEN_POSITIONS:
                    print(f"{Fore.YELLOW}Maksimum pozisyon sayısına ulaşıldı. {config.SCAN_INTERVAL} saniye bekleniyor...{Style.RESET_ALL}")
                    time.sleep(config.SCAN_INTERVAL)
                    continue
                
                # ADIM 3: Hesap bilgilerini al
                account = self.get_account_info()
                if not account:
                    print(f"{Fore.RED}Hesap bilgileri alınamadı. 30 saniye bekleniyor...{Style.RESET_ALL}")
                    time.sleep(30)
                    continue
                
                print(f"\n{Fore.CYAN}Hesap Durumu:")
                print(f"Toplam Bakiye: {account['total_balance']:.2f} USDT")
                print(f"Kullanılabilir Bakiye: {account['available_balance']:.2f} USDT{Style.RESET_ALL}")
                
                # ADIM 4: Tüm coinleri analiz et
                symbols = self.get_all_symbols()
                best_opportunity = {'score': 0, 'symbol': None, 'signal': None, 'price': None}
                
                print(f"\n{Fore.CYAN}Coin Analizi Başlıyor...")
                print(f"Toplam {len(symbols)} coin taranacak{Style.RESET_ALL}")
                
                for symbol in symbols:
                    # Bu coin'de açık pozisyon var mı?
                    if any(p['symbol'] == symbol for p in active_positions):
                        continue
                    
                    # Piyasa analizi yap
                    signal, price, score = self.analyze_market(symbol)
                    
                    # Daha iyi fırsat varsa kaydet
                    if signal and price and score > best_opportunity['score']:
                        best_opportunity = {
                            'score': score,
                            'symbol': symbol,
                            'signal': signal,
                            'price': price
                        }
                    
                    time.sleep(0.1)  # Rate limit'e takılmamak için
                
                # ADIM 5: En iyi fırsatı değerlendir
                if best_opportunity['symbol'] and best_opportunity['score'] >= config.MIN_OPPORTUNITY_SCORE:
                    print(f"\n{Fore.GREEN}En İyi Fırsat Bulundu!")
                    print(f"Coin: {best_opportunity['symbol']}")
                    print(f"Sinyal: {best_opportunity['signal']}")
                    print(f"Skor: {best_opportunity['score']:.1f}")
                    print(f"Fiyat: {best_opportunity['price']:.8f}{Style.RESET_ALL}")
                    
                    # İşlemi aç
                    if self.place_order(
                        best_opportunity['symbol'],
                        'BUY' if best_opportunity['signal'] == 'LONG' else 'SELL',
                        self.calculate_position_size(best_opportunity['symbol'], best_opportunity['price'], self.calculate_optimal_leverage(best_opportunity['symbol'])),
                        best_opportunity['price']
                    ):
                        print(f"{Fore.GREEN}İşlem başarıyla açıldı!{Style.RESET_ALL}")
                        time.sleep(5)  # İşlemin yerleşmesini bekle
                else:
                    print(f"\n{Fore.YELLOW}Uygun fırsat bulunamadı.{Style.RESET_ALL}")
                
                print(f"\n{Fore.CYAN}{config.SCAN_INTERVAL} saniye bekleniyor...{Style.RESET_ALL}")
                time.sleep(config.SCAN_INTERVAL)
                
            except Exception as e:
                print(f"{Fore.RED}{Style.BRIGHT}Bot çalışırken hata oluştu: {str(e)}{Style.RESET_ALL}")
                time.sleep(30)

if __name__ == "__main__":
    bot = FlexiTradeBot()
    bot.run()
