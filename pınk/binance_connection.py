import os
import sys
import time
import logging
import statistics
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Union, Optional
from dotenv import load_dotenv
import ccxt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from binance.client import Client
from binance.exceptions import BinanceAPIException
from binance.um_futures import UMFutures  # Futures için

# TensorFlow uyarılarını gizle
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Loglama ayarları
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_bot.log', mode='w', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Global değişkenler
TRADING_MODE = None  # '1': Spot, '2': Futures, '3': Both
MAX_RETRY_COUNT = 3
RETRY_DELAY = 5  # saniye

class BinanceConnection:
    def __init__(self):
        # .env dosyasından API anahtarlarını yükle
        load_dotenv()
        self.api_key = os.getenv('BINANCE_API_KEY')
        self.api_secret = os.getenv('BINANCE_API_SECRET')
        self.logger = self.logger_ayarla()
        self.trading_mode = int(os.getenv('TRADING_MODE', 1))  # 1: Spot, 2: Futures, 3: Both
        
        try:
            # Spot client
            self.spot_client = Client(self.api_key, self.api_secret)
            
            # Futures client
            self.futures_client = UMFutures(
                key=self.api_key,
                secret=self.api_secret
            )
            
            # Geriye dönük uyumluluk için
            self.client = self.spot_client
            
            self.test_connection()
            self.logger.info("Binance bağlantısı başarılı (Spot ve Futures)")
        except Exception as e:
            self.logger.error(f"Binance bağlantı hatası: {str(e)}")
            raise

    def logger_ayarla(self):
        logger = logging.getLogger('binance_connection')
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        
        # Dosyaya logla
        fh = logging.FileHandler('binance_connection.log')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        
        # Konsola logla
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        
        return logger

    def test_connection(self):
        try:
            self.client.ping()
            return True
        except BinanceAPIException as e:
            self.logger.error(f"Bağlantı testi başarısız: {str(e)}")
            return False

    def get_symbol_data(self, symbol, interval='1h', limit=100):
        """
        Belirli bir sembol için kline verilerini getirir
        """
        try:
            klines = self.client.get_klines(
                symbol=symbol,
                interval=interval,
                limit=limit
            )
            
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 
                'volume', 'close_time', 'quote_volume', 'trades',
                'taker_base', 'taker_quote', 'ignore'
            ])
            
            # Veri tiplerini dönüştür
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)
                
            return df
            
        except BinanceAPIException as e:
            self.logger.error(f"Veri çekme hatası ({symbol}): {str(e)}")
            return None

    def get_balances(self):
        """
        Hem spot hem de futures bakiyelerini getirir
        """
        balances = {
            'spot': [],
            'futures': []
        }
        
        try:
            # Spot bakiyeleri
            if self.trading_mode in [1, 3]:
                spot_account = self.spot_client.get_account()
                for balance in spot_account['balances']:
                    free = float(balance['free'])
                    locked = float(balance['locked'])
                    if free > 0 or locked > 0:
                        balances['spot'].append({
                            'asset': balance['asset'],
                            'free': free,
                            'locked': locked,
                            'total': free + locked
                        })
                self.logger.info(f"Spot Bakiyeleri: {balances['spot']}")

            # Futures bakiyeleri
            if self.trading_mode in [2, 3]:
                try:
                    futures_account = self.futures_client.account()
                    self.logger.info(f"Futures Hesap Bilgisi: {futures_account}")
                    
                    # Tüm varlıkları kontrol et
                    for asset in futures_account.get('assets', []):
                        wallet_balance = float(asset.get('walletBalance', 0))
                        unrealized_profit = float(asset.get('unrealizedProfit', 0))
                        
                        # Sıfırdan büyük tüm bakiyeleri ekle
                        if wallet_balance > 0 or unrealized_profit != 0:
                            balances['futures'].append({
                                'asset': asset['asset'],
                                'wallet_balance': wallet_balance,
                                'unrealized_profit': unrealized_profit,
                                'total': wallet_balance + unrealized_profit
                            })
                    
                    self.logger.info(f"Futures Bakiyeleri: {balances['futures']}")
                    
                    # Eğer hiç bakiye yoksa bilgilendir
                    if not balances['futures']:
                        self.logger.warning("Futures hesabında bakiye bulunamadı!")
                    
                except Exception as futures_error:
                    self.logger.error(f"Futures bakiye sorgulama hatası: {str(futures_error)}")
                    
            return balances
            
        except Exception as e:
            self.logger.error(f"Genel bakiye sorgulama hatası: {str(e)}")
            return balances

    def create_order(self, symbol, side, order_type, quantity, price=None, is_futures=False):
        """
        Spot veya Futures emir oluşturur
        """
        try:
            params = {
                'symbol': symbol,
                'side': side,
                'type': order_type,
                'quantity': quantity
            }
            
            if price:
                params['price'] = price
                
            if is_futures:
                order = self.futures_client.new_order(**params)
            else:
                order = self.spot_client.create_order(**params)
                
            self.logger.info(f"Emir oluşturuldu ({('Futures' if is_futures else 'Spot')}): {order}")
            return order
            
        except Exception as e:
            self.logger.error(f"Emir oluşturma hatası: {str(e)}")
            return None

    def get_all_usdt_pairs(self):
        """
        Tüm USDT çiftlerini getirir
        """
        try:
            tickers = self.client.get_all_tickers()
            return [t['symbol'] for t in tickers if t['symbol'].endswith('USDT')]
        except BinanceAPIException as e:
            self.logger.error(f"USDT çiftleri getirme hatası: {str(e)}")
            return []

    def get_symbol_price(self, symbol):
        """
        Anlık fiyat bilgisini getirir
        """
        try:
            if symbol.endswith('USDT'):
                ticker = self.spot_client.get_ticker(symbol=symbol)
            else:
                ticker = self.futures_client.ticker_price(symbol=symbol)
            return float(ticker['price'])
        except BinanceAPIException as e:
            self.logger.error(f"Fiyat sorgulama hatası ({symbol}): {str(e)}")
            return None

    def get_order_status(self, symbol, order_id):
        """
        Emir durumunu kontrol eder
        """
        try:
            order = self.client.get_order(symbol=symbol, orderId=order_id)
            return order['status']
        except BinanceAPIException as e:
            self.logger.error(f"Emir durumu sorgulama hatası: {str(e)}")
            return None

    def cancel_order(self, symbol, order_id):
        """
        Emri iptal eder
        """
        try:
            result = self.client.cancel_order(symbol=symbol, orderId=order_id)
            self.logger.info(f"Emir iptal edildi: {result}")
            return True
        except BinanceAPIException as e:
            self.logger.error(f"Emir iptal hatası: {str(e)}")
            return False

    def get_exchange_info(self, symbol):
        """
        Sembol için borsa bilgilerini getirir
        """
        try:
            if symbol.endswith('USDT'):
                info = self.spot_client.get_exchange_info()
            else:
                info = self.futures_client.exchange_info()
            for s in info['symbols']:
                if s['symbol'] == symbol:
                    return s
            return None
        except BinanceAPIException as e:
            self.logger.error(f"Borsa bilgisi getirme hatası: {str(e)}")
            return None

    def check_and_handle_errors(self, response):
        """
        API yanıtlarını kontrol eder ve hataları yönetir
        """
        if 'code' in response:
            error_msg = f"API Hatası: {response['code']} - {response['msg']}"
            self.logger.error(error_msg)
            raise BinanceAPIException(error_msg)
        return response

    def get_spot_balance(self):
        """
        Spot hesap bakiyelerini getirir
        """
        try:
            balances = []
            account = self.spot_client.get_account()
            
            for balance in account['balances']:
                free_amount = float(balance['free'])
                locked_amount = float(balance['locked'])
                
                if free_amount > 0 or locked_amount > 0:
                    balances.append({
                        'asset': balance['asset'],
                        'free': free_amount,
                        'locked': locked_amount,
                        'total': free_amount + locked_amount
                    })
            
            return balances
            
        except BinanceAPIException as e:
            self.logger.error(f"Spot bakiye sorgulama hatası: {str(e)}")
            return []

    def get_open_orders(self, symbol=None):
        """
        Açık emirleri getirir
        """
        try:
            if symbol:
                return self.spot_client.get_open_orders(symbol=symbol)
            return self.spot_client.get_open_orders()
        except BinanceAPIException as e:
            self.logger.error(f"Açık emir sorgulama hatası: {str(e)}")
            return []

    def get_all_positions(self):
        """
        Tüm pozisyonları getirir
        """
        try:
            positions = []
            account = self.spot_client.get_account()
            
            for balance in account['balances']:
                free = float(balance['free'])
                locked = float(balance['locked'])
                total = free + locked
                
                if total > 0:
                    positions.append({
                        'symbol': balance['asset'],
                        'amount': total,
                        'free': free,
                        'locked': locked
                    })
                    
            return positions
            
        except BinanceAPIException as e:
            self.logger.error(f"Pozisyon sorgulama hatası: {str(e)}")
            return []

    def close_all_positions(self, market_type='both'):
        """
        Tüm pozisyonları kapatır
        market_type: 'spot', 'futures', 'both'
        """
        closed = {
            'spot': [],
            'futures': []
        }
        
        try:
            # Spot pozisyonları kapat
            if market_type in ['spot', 'both'] and self.trading_mode in [1, 3]:
                spot_closed = super().close_all_positions()  # Mevcut spot kapatma fonksiyonu
                closed['spot'] = spot_closed

            # Futures pozisyonları kapat
            if market_type in ['futures', 'both'] and self.trading_mode in [2, 3]:
                futures_positions = self.futures_client.get_position_risk()
                for pos in futures_positions:
                    amount = float(pos['positionAmt'])
                    if amount != 0:
                        # Pozisyon yönünün tersine market emri ver
                        side = 'SELL' if amount > 0 else 'BUY'
                        order = self.futures_client.new_order(
                            symbol=pos['symbol'],
                            side=side,
                            type='MARKET',
                            quantity=abs(amount)
                        )
                        closed['futures'].append(order)
                        
            return closed
            
        except Exception as e:
            self.logger.error(f"Pozisyon kapatma hatası: {str(e)}")
            return closed

    def get_leverage_brackets(self, symbol):
        """
        Futures sembolü için kaldıraç aralıklarını getirir
        """
        try:
            return self.futures_client.get_leverage_brackets(symbol=symbol)
        except Exception as e:
            self.logger.error(f"Kaldıraç bilgisi getirme hatası: {str(e)}")
            return None

    def change_leverage(self, symbol, leverage):
        """
        Futures sembolü için kaldıracı değiştirir
        """
        try:
            return self.futures_client.change_leverage(
                symbol=symbol, 
                leverage=leverage
            )
        except Exception as e:
            self.logger.error(f"Kaldıraç değiştirme hatası: {str(e)}")
            return None

# Global Binance bağlantısı
binance = BinanceConnection()

class PerformanceTracker:
    def __init__(self):
        self.trades = []
        self.total_profit = 0
        self.winning_trades = 0
        self.losing_trades = 0
        
    def add_trade(self, symbol, side, entry_price, exit_price, position_size, trade_type):
        profit = (exit_price - entry_price) * position_size if side == 1 else (entry_price - exit_price) * position_size
        profit_percent = ((exit_price - entry_price) / entry_price * 100) * (1 if side == 1 else -1)
        
        trade = {
            'symbol': symbol,
            'side': 'LONG' if side == 1 else 'SHORT',
            'entry_price': entry_price,
            'exit_price': exit_price,
            'position_size': position_size,
            'profit_usdt': profit,
            'profit_percent': profit_percent,
            'trade_type': trade_type,
            'timestamp': datetime.now()
        }
        
        self.trades.append(trade)
        self.total_profit += profit
        
        if profit > 0:
            self.winning_trades += 1
        else:
            self.losing_trades += 1
    
    def get_statistics(self):
        total_trades = len(self.trades)
        if total_trades == 0:
            return "No trades yet"
            
        win_rate = (self.winning_trades / total_trades * 100) if total_trades > 0 else 0
        avg_profit = sum(t['profit_percent'] for t in self.trades) / total_trades
        
        last_24h_trades = [t for t in self.trades if (datetime.now() - t['timestamp']).total_seconds() <= 86400]
        last_24h_profit = sum(t['profit_usdt'] for t in last_24h_trades)
        
        stats = f"\nPerformance Statistics:"
        stats += f"\n- Total Trades: {total_trades}"
        stats += f"\n- Win Rate: {win_rate:.2f}%"
        stats += f"\n- Total Profit: {self.total_profit:.2f} USDT"
        stats += f"\n- Average Profit per Trade: {avg_profit:.2f}%"
        stats += f"\n- Last 24h Profit: {last_24h_profit:.2f} USDT"
        stats += f"\n- Best Trade: {max(t['profit_percent'] for t in self.trades):.2f}%" if self.trades else "N/A"
        stats += f"\n- Worst Trade: {min(t['profit_percent'] for t in self.trades):.2f}%" if self.trades else "N/A"
        
        return stats

class MoneyManager:
    def __init__(self):
        self.daily_profit = 0
        self.daily_loss = 0
        self.weekly_profit = 0
        self.weekly_loss = 0
        self.min_trade_amount = 10  # Minimum işlem miktarı 10 USDT
        
        # Dinamik zarar limitleri (bakiyeye göre)
        self.max_daily_loss_percent = 1.5  # Günlük maksimum %1.5 zarar
        self.max_weekly_loss_percent = 4  # Haftalık maksimum %4 zarar
        self.daily_profit_target_percent = 3  # Günlük %3 kar hedefi
        
        self.strategy_performance = {}
        self.last_reset = datetime.now()
        
    def update_limits(self, balance):
        # Bakiyeye göre limitleri güncelle
        self.max_daily_loss = -(balance * self.max_daily_loss_percent / 100)
        self.max_weekly_loss = -(balance * self.max_weekly_loss_percent / 100)
        self.daily_profit_target = balance * self.daily_profit_target_percent / 100
        
    def check_risk_limits(self, symbol, amount, market_type):
        """
        Risk limitlerini kontrol et
        """
        try:
            # Günlük ve haftalık limitleri kontrol et
            if abs(self.daily_loss) > self.max_daily_loss_percent:
                logger.warning(f"Günlük zarar limiti aşıldı! ({self.daily_loss:.2f}%)")
                return False
                
            if abs(self.weekly_loss) > self.max_weekly_loss_percent:
                logger.warning(f"Haftalık zarar limiti aşıldı! ({self.weekly_loss:.2f}%)")
                return False
                
            if self.daily_profit > self.daily_profit_target_percent:
                logger.info(f"Günlük kar hedefine ulaşıldı! ({self.daily_profit:.2f}%)")
                return False
                
            # Bakiye kontrolü
            if market_type == 'spot':
                balance = float(binance.spot_client.get_account()['total']['USDT'])
            else:  # futures
                balance = float(binance.futures_client.account()['USDT']['total'])
                
            if balance < amount:
                logger.warning(f"Yetersiz bakiye! Mevcut: {balance:.2f} USDT, Gerekli: {amount:.2f} USDT")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Risk limit kontrolü hatası: {str(e)}")
            return False

    def update_profits(self, profit, strategy):
        # Günlük/haftalık kar/zarar güncelle
        self.daily_profit += profit
        self.weekly_profit += profit
        
        # Strateji performansını güncelle
        if strategy not in self.strategy_performance:
            self.strategy_performance[strategy] = {'profit': 0, 'trades': 0}
        self.strategy_performance[strategy]['profit'] += profit
        self.strategy_performance[strategy]['trades'] += 1
        
        # Günlük reset kontrol
        if (datetime.now() - self.last_reset).days >= 1:
            self.daily_profit = profit
            self.daily_loss = min(0, profit)
            self.last_reset = datetime.now()
            
    def get_position_size_multiplier(self, strategy):
        if strategy not in self.strategy_performance:
            return 1.0
            
        perf = self.strategy_performance[strategy]
        if perf['trades'] < 5:  # Minimum trade sayısı
            return 1.0
            
        avg_profit = perf['profit'] / perf['trades']
        if avg_profit > 0:
            return min(1.5, 1 + avg_profit/100)  # Maksimum 1.5x
        else:
            return max(0.5, 1 + avg_profit/100)  # Minimum 0.5x

class MarketRegimeAnalyzer:
    def __init__(self):
        self.lookback_period = 100
        
    def analyze_regime(self, symbol, timeframe='1h'):
        try:
            ohlcv = binance.futures_client.fetch_ohlcv(symbol, timeframe, limit=self.lookback_period)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # Trend Analizi
            df['sma20'] = df['close'].rolling(20).mean()
            df['sma50'] = df['close'].rolling(50).mean()
            
            current_price = df['close'].iloc[-1]
            sma20 = df['sma20'].iloc[-1]
            sma50 = df['sma50'].iloc[-1]
            
            # Volatilite Analizi
            df['returns'] = df['close'].pct_change()
            current_volatility = df['returns'].std() * np.sqrt(24)
            
            # Range Analizi
            df['high_low_range'] = df['high'] - df['low']
            avg_range = df['high_low_range'].mean()
            current_range = df['high_low_range'].iloc[-1]
            
            # Rejim Tespiti
            regime = {
                'trend': 'uptrend' if current_price > sma20 > sma50 else 'downtrend' if current_price < sma20 < sma50 else 'sideways',
                'volatility': 'high' if current_volatility > 0.03 else 'low',
                'range': 'expanding' if current_range > avg_range * 1.2 else 'contracting' if current_range < avg_range * 0.8 else 'normal'
            }
            
            return regime
        except Exception as e:
            print(f"Error in regime analysis: {e}")
            return None

class MultiTimeframeAnalyzer:
    def __init__(self):
        self.timeframes = ['5m', '15m', '1h', '4h']
        
    def analyze(self, symbol):
        signals = {}
        total_score = 0
        
        for tf in self.timeframes:
            try:
                data = self.get_data(symbol, tf)
                if data is None:
                    continue
                    
                signals[tf] = self.analyze_timeframe(data)
                total_score += signals[tf]['score']
                
            except Exception as e:
                print(f"Error analyzing {tf}: {e}")
                
        # Ağırlıklı skor hesapla
        weighted_score = total_score / len(self.timeframes)
        
        return {
            'signals': signals,
            'weighted_score': weighted_score
        }
        
    def get_data(self, symbol, timeframe):
        try:
            ohlcv = binance.futures_client.fetch_ohlcv(symbol, timeframe, limit=100)
            return pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        except:
            return None
            
    def analyze_timeframe(self, data):
        # Teknik indikatörler
        rsi = calculate_rsi(data['close'])
        macd = calculate_macd(data['close'])
        
        # Momentum ve trend skorları
        momentum_score = 1 if rsi[-1] > 50 else -1
        trend_score = 1 if macd['macd'][-1] > macd['signal'][-1] else -1
        
        return {
            'rsi': rsi[-1],
            'macd': macd['macd'][-1],
            'signal': macd['signal'][-1],
            'score': (momentum_score + trend_score) / 2
        }

class ProfitOptimizer:
    def __init__(self):
        self.partial_tp_levels = [2, 3, 5]  # Kısmi kar alma seviyeleri
        self.position_close_percent = [0.3, 0.3, 0.4]  # Her seviyede kapatılacak miktar
        
    def check_and_take_profits(self, symbol, side, entry_price, position_size):
        try:
            current_price = float(binance.futures_client.fetch_ticker(symbol)['last'])
            profit_percent = ((current_price - entry_price) / entry_price * 100) * (1 if side == 1 else -1)
            
            for level, close_percent in zip(self.partial_tp_levels, self.position_close_percent):
                if profit_percent >= level:
                    close_size = position_size * close_percent
                    if close_size > 0:
                        print(f"\nTaking partial profits for {symbol}")
                        print(f"Profit: {profit_percent:.2f}%")
                        print(f"Closing {close_percent*100}% of position")
                        
                        # Pozisyonu kısmen kapat
                        binance.futures_client.create_order(
                            symbol=symbol,
                            type='MARKET',
                            side='sell' if side == 1 else 'buy',
                            amount=close_size
                        )
                        
                        return True
            return False
        except Exception as e:
            print(f"Error in profit optimization: {e}")
            return False

class AITrader:
    def __init__(self):
        self.scaler = StandardScaler()
        self.training_data = []
        self.training_labels = []
        self.rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.lstm_model = self._build_lstm_model()
        self.min_confidence = 0.65  # Minimum tahmin güvenilirliği
        self.validation_window = 20  # Son 20 tahminin doğruluğunu takip et
        self.prediction_history = []  # Tahmin geçmişi
        self.accuracy_threshold = 0.60  # Minimum doğruluk oranı
        
    def _build_lstm_model(self):
        model = Sequential([
            LSTM(50, input_shape=(1, 8), return_sequences=True),
            Dropout(0.2),
            LSTM(50),
            Dropout(0.2),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model
        
    def prepare_data(self, data):
        features = np.array([
            float(data['price_change']),
            float(data['volume_change']),
            float(data['rsi']),
            float(data['macd']),
            float(data['macd_signal']),
            float(data['macd_hist']),
            float(data['bb_upper']),
            float(data['bb_lower'])
        ]).reshape(1, -1)
        
        return self.scaler.fit_transform(features)
        
    def predict_rf(self, features):
        if len(self.training_data) < 100:
            return 0.5
            
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        pred_prob = self.rf_model.predict_proba(features_scaled)[0][1]
        
        # Güvenilirlik kontrolü
        if abs(pred_prob - 0.5) < 0.15:  # Kararsız tahmin
            return 0.5
            
        return pred_prob
        
    def predict_lstm(self, features):
        if len(self.training_data) < 100:
            return 0.5
            
        features_scaled = self.scaler.transform(features)
        features_reshaped = features_scaled.reshape((1, 1, 8))
        pred = self.lstm_model.predict(features_reshaped)[0][0]
        
        # Güvenilirlik kontrolü
        if abs(pred - 0.5) < 0.15:
            return 0.5
            
        return pred
        
    def update_models(self, features, actual_label):
        self.training_data.append(features)
        self.training_labels.append(actual_label)
        
        # Tahmin geçmişini güncelle
        if len(self.prediction_history) >= self.validation_window:
            self.prediction_history.pop(0)
        self.prediction_history.append(actual_label)
        
        # Modelleri güncelle
        if len(self.training_data) >= 100:
            # Son 1000 veriyi kullan
            X = np.array(self.training_data[-1000:])
            y = np.array(self.training_labels[-1000:])
            
            # Random Forest güncelleme
            self.rf_model.fit(X, y)
            
            # LSTM güncelleme (her 100 örnekte bir)
            if len(self.training_data) % 100 == 0:
                X_lstm = X.reshape((len(X), 1, 8))
                self.lstm_model.fit(X_lstm, y, epochs=10, batch_size=32, verbose=0)
                
    def get_prediction_accuracy(self):
        if len(self.prediction_history) < self.validation_window:
            return 0.5
            
        correct_predictions = sum(self.prediction_history)
        return correct_predictions / len(self.prediction_history)

def calculate_ai_signal(data):
    """
    AI tabanlı alım/satım sinyalleri
    """
    try:
        features = ai_trader.prepare_data(data)
        
        # Model tahminleri
        rf_pred = ai_trader.predict_rf(features[-1])
        lstm_pred = ai_trader.predict_lstm(features)
        
        # Tahmin güvenilirliği kontrolü
        if rf_pred < ai_trader.min_confidence and lstm_pred < ai_trader.min_confidence:
            return {'signal': 0, 'confidence': 0}
            
        # Model doğruluk kontrolü
        accuracy = ai_trader.get_prediction_accuracy()
        if accuracy < ai_trader.accuracy_threshold:
            return {'signal': 0, 'confidence': 0}
            
        # Ağırlıklı toplam sinyal
        ensemble_pred = (rf_pred * 0.6) + (lstm_pred * 0.4)
        signal_strength = abs(ensemble_pred - 0.5) * 2
        
        signal = {
            'signal': signal_strength if ensemble_pred > 0.5 else -signal_strength,
            'confidence': max(rf_pred, lstm_pred)
        }
        
        # Gerçek sonucu kaydet (bir sonraki güncelleme için)
        if data['price_change'] > 0:
            actual_direction = 1
        else:
            actual_direction = 0
            
        ai_trader.update_models(features[-1], actual_direction)
        
        return signal
        
    except Exception as e:
        logger.error(f"AI sinyal hesaplama hatası: {str(e)}")
        return {'signal': 0, 'confidence': 0}

def manage_risk(symbol, side, quantity, entry_price):
    """
    Gelişmiş risk yönetimi
    """
    try:
        # Risk parametreleri
        account_size = float(get_balance())
        risk_per_trade = 0.02  # Hesap büyüklüğünün %2'si
        max_loss_amount = account_size * risk_per_trade
        
        # Stop loss hesaplama (ATR tabanlı)
        atr = calculate_atr(symbol)
        if side == 'BUY':
            stop_loss = entry_price - (atr * 2)  # 2 ATR altı
            take_profit = entry_price + (atr * 3)  # 3 ATR üstü
        else:
            stop_loss = entry_price + (atr * 2)
            take_profit = entry_price - (atr * 3)
            
        # Pozisyon büyüklüğü optimizasyonu
        price_volatility = calculate_volatility(symbol)
        if price_volatility > 0.05:  # Yüksek volatilite
            quantity = quantity * 0.7  # Pozisyon büyüklüğünü azalt
            
        # Risk/Ödül oranı kontrolü
        risk_amount = abs(entry_price - stop_loss) * quantity
        reward_amount = abs(take_profit - entry_price) * quantity
        
        if risk_amount > max_loss_amount or (reward_amount / risk_amount) < 2:
            logger.warning(f"{symbol} - Risk/Ödül oranı uygun değil")
            return False
            
        # Trailing stop ayarla
        trailing_percent = min(2.0, price_volatility * 100)  # Volatiliteye göre ayarla
        
        return {
            'quantity': quantity,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'trailing_percent': trailing_percent
        }
        
    except Exception as e:
        logger.error(f"Risk yönetimi hatası: {str(e)}")
        return False

def execute_trade(symbol, direction, amount, market_type='spot'):
    """
    İşlem gerçekleştir
    """
    try:
        logger.info("\n=== İŞLEM BAŞLATILIYOR ===")
        logger.info(f"Symbol: {symbol}")
        logger.info(f"Yön: {'LONG/BUY' if direction == 1 else 'SHORT/SELL'}")
        logger.info(f"Miktar: {amount} USDT")
        logger.info(f"Piyasa: {market_type.upper()}")
        
        # Minimum gereksinimleri kontrol et
        if not check_min_trade_requirements(symbol, amount, market_type):
            return False
            
        # Risk kontrolü
        if not money_manager.check_risk_limits(symbol, amount, market_type):
            return False
            
        # Client'ı seç
        client = binance.futures_client if market_type == 'futures' else binance.spot_client
        
        # Market bilgilerini al
        market = client.load_markets()
        if symbol not in market:
            logger.error(f"Symbol bulunamadı: {symbol}")
            return False
            
        # Güncel fiyatı al
        if symbol.endswith('USDT'):
            ticker = client.get_ticker(symbol=symbol)
        else:
            ticker = client.ticker_price(symbol=symbol)
        current_price = float(ticker['price'])
        
        # İşlem miktarını hesapla
        quantity = amount / current_price
        
        # Stop-loss ve take-profit hesapla
        if market_type == 'futures':
            leverage = calculate_optimal_leverage(symbol, market_data['ai_confidence'])
            client.set_leverage(leverage, symbol)
            
            stop_loss, take_profit = calculate_dynamic_stop_loss(
                symbol=symbol,
                side='buy' if direction == 1 else 'sell',
                entry_price=current_price
            )
            
            logger.info(f"Kaldıraç: {leverage}x")
            logger.info(f"Stop-Loss: {stop_loss:.8f}")
            logger.info(f"Take-Profit: {take_profit:.8f}")
        
        # İşlemi gerçekleştir
        order = client.create_order(
            symbol=symbol,
            type='MARKET',
            side='buy' if direction == 1 else 'sell',
            amount=quantity
        )
        
        logger.info("\n=== İŞLEM BAŞARILI ===")
        logger.info(f"Order ID: {order['id']}")
        logger.info(f"İşlem Fiyatı: {float(order['price']) if order.get('price') else current_price:.8f}")
        logger.info(f"Gerçekleşen Miktar: {float(order['filled']):.8f}")
        
        # Stop-loss ve take-profit orderlarını ekle
        if market_type == 'futures':
            # Stop-loss order
            client.create_order(
                symbol=symbol,
                type='STOP_MARKET',
                side='sell' if direction == 1 else 'buy',
                amount=quantity,
                params={'stopPrice': stop_loss}
            )
            
            # Take-profit order
            client.create_order(
                symbol=symbol,
                type='TAKE_PROFIT_MARKET',
                side='sell' if direction == 1 else 'buy',
                amount=quantity,
                params={'stopPrice': take_profit}
            )
            
            logger.info("Stop-loss ve take-profit orderları eklendi")
        
        # İşlemi kaydet
        trade_info = {
            'symbol': symbol,
            'side': 'buy' if direction == 1 else 'sell',
            'amount': amount,
            'price': current_price,
            'market_type': market_type,
            'timestamp': datetime.now().timestamp()
        }
        
        if market_type == 'futures':
            trade_info.update({
                'leverage': leverage,
                'stop_loss': stop_loss,
                'take_profit': take_profit
            })
            
        save_trade_history(trade_info)
        
        return True
        
    except Exception as e:
        logger.error(f"\n=== İŞLEM HATASI ===")
        logger.error(f"Hata: {str(e)}")
        logger.error(f"Detay: {e.__class__.__name__}")
        return False

def manage_futures_risk(balance, entry_price, volatility, leverage=10, max_risk=0.02):
    # Calculate position size based on max risk
    position_value = balance * max_risk
    position_size = position_value / (entry_price * leverage)
    print(f"Calculated Position Size: {position_size}")

    # Dynamic leverage adjustment based on volatility
    if volatility > 0.03:
        leverage = max(5, leverage - 1)
    elif volatility < 0.01:
        leverage = min(20, leverage + 1)

    print(f"Adjusted Leverage: {leverage}x")
    return position_size, leverage

def signal_handler(signum, frame):
    print("\nStopping the trading bot...")
    sys.exit(0)

def continuous_market_update(interval=60):
    # Register signal handler for Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)
    
    print("\nTrading bot started. Press Ctrl+C to stop.")
    while True:
        try:
            # Update the list of cryptocurrencies to monitor
            cryptos_to_monitor = update_crypto_list_based_on_criteria()
            print("\nMonitoring these pairs:", cryptos_to_monitor)

            # Analyze and trade for each cryptocurrency
            analyze_and_trade_for_all_cryptos(trading_mode=mode)

            print(f"\nWaiting {interval} seconds before next update...")
            time.sleep(interval)
        except Exception as e:
            print(f"\nError during market update: {e}")
            time.sleep(interval)

def start_trading_bot():
    """
    Trading botunu başlat
    """
    try:
        # Bot instance'ı oluştur
        bot = TradingBot()
        
        # Trading modunu ayarla
        mode = choose_trading_mode()
        bot.set_trading_mode(mode)
        
        # Futures bakiyesini kontrol et
        try:
            futures_balance = binance.futures_client.get_account()
            print(f"Futures Bakiyesi: {futures_balance['USDT']['total']}")
        except:
            print("Futures bakiyesi bulunamadı")
            return
        
        # Başlangıç mesajı
        logger.info("\n=== Trading Bot Başlatılıyor ===")
        logger.info(f"Trading Modu: {'Spot' if mode == '1' else 'Futures' if mode == '2' else 'Her İkisi'}")
        
        # Ana döngü
        while True:
            try:
                # Market analizi yap
                analyze_and_trade_for_all_cryptos(mode)
                
                # 5 dakika bekle
                logger.info("\n5 dakika bekleniyor...")
                time.sleep(300)
                
            except KeyboardInterrupt:
                logger.info("\nBot durduruluyor...")
                break
                
            except Exception as e:
                logger.error(f"\nHata: {str(e)}")
                time.sleep(2)
                
    except Exception as e:
        logger.error(f"\nBot başlatma hatası: {str(e)}")

def choose_trading_mode():
    """
    Trading modunu seç
    """
    while True:
        print("\nTrading Modu Seçin:")
        print("1. Spot Trading")
        print("2. Futures Trading")
        print("3. Her İkisi")
        
        try:
            mode = int(input("Seçiminiz (1-3): "))
            if mode in [1, 2, 3]:
                # .env dosyasını güncelle
                with open('.env', 'a') as f:
                    f.write(f"\nTRADING_MODE={mode}")
                
                logger.info(f"Trading modu seçildi: {mode}")
                return mode
            else:
                print("Geçersiz seçim. Lütfen 1, 2 veya 3 girin.")
        except ValueError:
            print("Lütfen bir sayı girin.")

def fetch_real_time_data(symbol):
    try:
        if symbol.endswith('USDT'):
            ticker = binance.spot_client.get_ticker(symbol=symbol)
        else:
            ticker = binance.futures_client.ticker_price(symbol=symbol)
        return ticker
    except Exception as e:
        print(f"Error fetching real-time data for {symbol}: {e}")
        return None

def update_indicators_with_real_time(data, real_time_data):
    if real_time_data:
        data['close'].iloc[-1] = real_time_data['price']
        data = calculate_technical_indicators(data)
    return data

def optimize_portfolio_and_manage_risk(balance, symbols, max_risk_per_trade=0.02):
    portfolio_value = balance
    position_sizes = {}
    for symbol in symbols:
        real_time_data = fetch_real_time_data(symbol)
        if real_time_data:
            entry_price = real_time_data['price']
            position_value = portfolio_value * max_risk_per_trade
            position_size = position_value / entry_price
            position_sizes[symbol] = position_size
            print(f"Calculated position size for {symbol}: {position_size}")
    return position_sizes

def get_account_balance(mode):
    try:
        balances = {'spot': None, 'futures': None}
        
        if mode in ['1', '3']:  # Spot or Both
            spot_balance = binance.spot_client.get_account()
            print("\nSpot Wallet Balances:")
            for asset in spot_balance['total']:
                if float(spot_balance['total'][asset]) > 0:
                    print(f"{asset}: {float(spot_balance['total'][asset])}")
            balances['spot'] = spot_balance
            
        if mode in ['2', '3']:  # Futures or Both
            futures_balance = binance.futures_client.get_account()
            print("\nFutures Wallet Balances:")
            usdt_balance = futures_balance['USDT']['total'] if 'USDT' in futures_balance else 0
            print(f"USDT: {float(usdt_balance):.8f}")
            balances['futures'] = futures_balance
            
        return balances
    except Exception as e:
        print(f"Error fetching balance: {e}")
        return None

def calculate_volatility(symbol, timeframe='1h', window=20):
    """
    Belirli bir zaman dilimi için volatilite hesapla
    """
    try:
        ohlcv = binance.futures_client.fetch_ohlcv(symbol, timeframe, limit=window)
        closes = [x[4] for x in ohlcv]
        returns = np.diff(np.log(closes))
        return np.std(returns) * np.sqrt(24)  # Günlük volatiliteye çevir
    except Exception as e:
        print(f"Error calculating volatility: {e}")
        return 0.05  # Default orta-yüksek volatilite

def calculate_optimal_leverage(symbol, confidence_score):
    """
    Market koşullarına göre optimal kaldıraç hesapla
    """
    try:
        volatility = calculate_volatility(symbol)
        base_leverage = 5  # Temel kaldıraç
        
        # Volatilite ve güven skoruna göre kaldıraç ayarla
        if volatility < 0.02 and abs(confidence_score) > 70:
            optimal_leverage = min(10, base_leverage * 2)
        elif volatility > 0.05:
            optimal_leverage = max(2, base_leverage / 2)
        else:
            optimal_leverage = base_leverage
            
        print(f"Volatility: {volatility:.4f}")
        print(f"Optimal Leverage: {optimal_leverage}x")
        return optimal_leverage
    except Exception as e:
        print(f"Error calculating leverage: {e}")
        return 3  # Hata durumunda güvenli kaldıraç

def add_trailing_stop(symbol, side, entry_price, position_size):
    """
    Karda olan pozisyonlar için trailing stop ekle
    """
    try:
        current_price = float(binance.futures_client.fetch_ticker(symbol)['last'])
        profit_percent = ((current_price - entry_price) / entry_price * 100) * (1 if side == 1 else -1)
        
        if profit_percent > 2:  # %2'den fazla karda
            print(f"\nAdding Trailing Stop for {symbol}")
            print(f"Current Profit: {profit_percent:.2f}%")
            
            binance.futures_client.create_order(
                symbol=symbol,
                type='TRAILING_STOP_MARKET',
                side='sell' if side == 1 else 'buy',
                amount=position_size,
                params={'callbackRate': 1.0, 'positionSide': 'LONG' if side == 1 else 'SHORT'}
            )
            print("Trailing Stop Added Successfully")
    except Exception as e:
        print(f"Error adding trailing stop: {e}")

def setup_grid_orders(symbol, base_price, position_size, grid_size=5, price_distance=0.5):
    """
    Bir fiyat seviyesi etrafında grid orderlar oluştur
    """
    try:
        print(f"\nSetting up Grid Orders for {symbol}")
        print(f"Base Price: {base_price}")
        print(f"Grid Size: {grid_size}")
        print(f"Price Distance: {price_distance}%")
        
        orders = []
        for i in range(grid_size):
            buy_price = base_price * (1 - (price_distance * (i+1) / 100))
            sell_price = base_price * (1 + (price_distance * (i+1) / 100))
            
            # Alış emirleri
            buy_order = binance.futures_client.create_order(
                symbol=symbol,
                type='LIMIT',
                side='buy',
                amount=position_size/grid_size,
                price=buy_price
            )
            orders.append(buy_order)
            
            # Satış emirleri
            sell_order = binance.futures_client.create_order(
                symbol=symbol,
                type='LIMIT',
                side='sell',
                amount=position_size/grid_size,
                price=sell_price
            )
            orders.append(sell_order)
            
        print(f"Created {len(orders)} grid orders")
        return orders
    except Exception as e:
        print(f"Error setting up grid orders: {e}")
        return []

def analyze_time_conditions():
    """
    Saat ve gün bazlı trading koşullarını analiz et
    """
    now = datetime.now()
    hour = now.hour
    day = now.weekday()
    
    conditions = []
    score = 0
    
    # Pazar günü riski azalt
    if day == 6:
        score -= 20
        conditions.append("Sunday: Reduced Risk")
    
    # Yüksek volatilite saatleri
    if 2 <= hour <= 4:
        score += 10
        conditions.append("Asian Session Opening")
    elif 9 <= hour <= 11:
        score += 10
        conditions.append("European Session Opening")
    elif 15 <= hour <= 17:
        score += 10
        conditions.append("US Session Opening")
        
    print("\nTime Analysis:")
    for condition in conditions:
        print(f"- {condition}")
    print(f"Time Score: {score}")
    
    return score

def calculate_correlation_score(symbol1, symbol2, timeframe='1h'):
    """
    İki kripto arasındaki korelasyonu hesapla
    """
    try:
        ohlcv1 = binance.futures_client.fetch_ohlcv(symbol1, timeframe)
        ohlcv2 = binance.futures_client.fetch_ohlcv(symbol2, timeframe)
        
        prices1 = [x[4] for x in ohlcv1]
        prices2 = [x[4] for x in ohlcv2]
        
        correlation = np.corrcoef(prices1, prices2)[0,1]
        return correlation
    except Exception as e:
        print(f"Error calculating correlation: {e}")
        return 1

def check_portfolio_diversity(symbol, open_positions):
    """
    Portföy çeşitliliğini kontrol et
    """
    try:
        # Açık pozisyonlar arasındaki korelasyonları kontrol et
        for pos in open_positions:
            if pos['symbol'] != symbol:
                correlation = calculate_correlation_score(symbol, pos['symbol'])
                if correlation > 0.8:  # Yüksek korelasyon
                    print(f"\nHigh correlation ({correlation:.2f}) between {symbol} and {pos['symbol']}")
                    return False
        return True
    except Exception as e:
        print(f"Error checking portfolio diversity: {e}")
        return True

def calculate_dynamic_stop_loss(symbol, side, entry_price):
    try:
        # ATR (Average True Range) hesapla
        ohlcv = binance.futures_client.fetch_ohlcv(symbol, '15m', limit=20)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        tr_list = []
        for i in range(1, len(df)):
            high_low = df['high'][i] - df['low'][i]
            high_close = abs(df['high'][i] - df['close'][i-1])
            low_close = abs(df['low'][i] - df['close'][i-1])
            tr = max(high_low, high_close, low_close)
            tr_list.append(tr)
        
        atr = sum(tr_list) / len(tr_list)
        
        # Volatiliteye göre stop loss hesapla
        volatility = calculate_volatility(symbol)
        
        # Temel stop loss mesafesi (ATR'nin 1.5 katı)
        base_stop = atr * 1.5
        
        # Volatiliteye göre ayarla
        if volatility > 0.05:  # Yüksek volatilite
            base_stop = base_stop * 1.2
        elif volatility < 0.02:  # Düşük volatilite
            base_stop = base_stop * 0.8
        
        # Maksimum stop loss mesafesi (giriş fiyatının %2'si)
        max_stop = entry_price * 0.02
        
        # Stop loss mesafesini sınırla
        stop_distance = min(base_stop, max_stop)
        
        if side == 1:  # Long pozisyon
            stop_loss = entry_price - stop_distance
        else:  # Short pozisyon
            stop_loss = entry_price + stop_distance
        
        return stop_loss, (abs(stop_loss - entry_price) / entry_price) * 100
        
    except Exception as e:
        print(f"Stop loss hesaplama hatası: {e}")
        # Default değerlere dön
        return entry_price * (0.98 if side == 1 else 1.02), 2

def execute_trade(symbol, direction, amount):
    """
    İşlem gerçekleştir
    :param symbol: İşlem yapılacak sembol
    :param direction: 1 (alış) veya -1 (satış)
    :param amount: İşlem miktarı (USDT)
    """
    try:
        # Risk kontrolü
        if not money_manager.check_risk_limits():
            print("Risk limitleri aşıldı, işlem yapılamıyor")
            return False
            
        # Kaldıraç optimize et
        optimal_leverage = calculate_optimal_leverage(symbol, {'total_score': 0})  # Güvenli kaldıraç
        try:
            binance.futures_client.fapiPrivate_post_leverage({
                'symbol': symbol.replace('/', ''),
                'leverage': optimal_leverage
            })
        except Exception as e:
            print(f"Kaldıraç ayarlama hatası: {e}")
        
        side_text = 'buy' if direction == 1 else 'sell'
        
        # Futures pozisyonu aç
        order = binance.futures_client.create_order(
            symbol=symbol,
            type='MARKET',
            side=side_text,
            amount=amount,
            params={'reduceOnly': False}
        )
        
        print(f"\nİşlem Gerçekleşti:")
        print(f"Sembol: {symbol}")
        print(f"Yön: {'LONG/BUY' if direction == 1 else 'SHORT/SELL'}")
        print(f"Miktar: {amount} USDT")
        print(f"Kaldıraç: {optimal_leverage}x")
        
        return True
        
    except Exception as e:
        print(f"İşlem hatası ({symbol}): {e}")
        return False

def analyze_and_trade_for_all_cryptos(trading_mode):
    """
    Tüm kriptoları analiz et ve uygun işlemleri aç
    Gelişmiş analiz, ML ve risk yönetimi kullanır
    """
    try:
        # Market ve zaman koşullarını analiz et
        regime_analyzer = MarketRegimeAnalyzer()
        mtf_analyzer = MultiTimeframeAnalyzer()
        ai_trader = AITrader()
        
        # En aktif coinleri al
        target_pairs = get_active_trading_pairs()
        
        logger.info("\nPiyasa Analizi Başlıyor...")
        logger.info(f"Analiz Edilecek Coin Sayısı: {len(target_pairs)}")
        
        for symbol in target_pairs:
            try:
                # Market türünü belirle
                market_type = 'futures' if ':USDT' in symbol else 'spot'
                
                # Temel analiz
                market_conditions = calculate_market_conditions(symbol)
                if not market_conditions:
                    continue
                    
                # Toplam skoru hesapla
                total_score = calculate_total_score(symbol, market_conditions)
                
                # Sonuçları logla
                logger.info(f"\n{symbol} Analiz Sonuçları:")
                logger.info(f"Fiyat Değişimi (24s): %{market_conditions['price_change']:.2f}")
                logger.info(f"Hacim Değişimi: %{market_conditions['volume_change']:.2f}")
                logger.info(f"RSI: {market_conditions['rsi']:.2f}")
                logger.info(f"Toplam Puan: {total_score}")
                
                # İşlem sinyali varsa
                if abs(total_score) >= 2:  # Alış sinyali
                    logger.info(f"\n{symbol} için {'ALIŞ' if total_score > 0 else 'SATIŞ'} sinyali!")
                    
                    # İşlemi gerçekleştir
                    execute_trade(symbol, 1 if total_score > 0 else -1, 10, market_type)
                    
            except Exception as e:
                logger.error(f"{symbol} analiz hatası: {str(e)}")
                continue
                
        logger.info("\n5 dakika bekleniyor...")
        time.sleep(300)
        
    except Exception as e:
        logger.error(f"Genel analiz hatası: {str(e)}")

def get_active_trading_pairs():
    """
    En aktif trade edilen coinleri seç (hem spot hem futures)
    Hacim, fiyat değişimi ve volatiliteye göre sırala
    """
    try:
        logger.info("Aktif coin listesi güncelleniyor...")
        pair_metrics = []
        
        # SPOT marketleri al
        try:
            spot_markets = binance.spot_client.get_exchange_info()
            for symbol, market in spot_markets['symbols'].items():
                if (symbol.endswith('USDT') and 
                    market.get('status', '') == 'TRADING' and
                    market.get('isSpotTradingAllowed', False)):
                    try:
                        ticker = binance.spot_client.get_ticker(symbol=symbol)
                        
                        # Gerekli verilerin varlığını kontrol et
                        if (ticker.get('quoteVolume') is None or 
                            ticker.get('priceChangePercent') is None or 
                            ticker.get('highPrice') is None or 
                            ticker.get('lowPrice') is None):
                            continue
                        
                        volume_usdt = float(ticker['quoteVolume'])
                        price_change = float(ticker['priceChangePercent'])
                        high = float(ticker['highPrice'])
                        low = float(ticker['lowPrice'])
                        
                        if low == 0:
                            continue
                            
                        volatility = (high - low) / low * 100
                        
                        # Minimum hacim filtresi (100,000 USDT)
                        if volume_usdt < 100000:
                            continue
                        
                        # Metrik skoru hesapla
                        volume_score = min(volume_usdt / 1000000, 10)
                        change_score = abs(price_change)
                        volatility_score = min(volatility, 20)
                        
                        total_score = volume_score + change_score + volatility_score
                        
                        pair_metrics.append({
                            'symbol': symbol,
                            'volume': volume_usdt,
                            'price_change': price_change,
                            'volatility': volatility,
                            'score': total_score,
                            'type': 'spot'
                        })
                        
                    except Exception as e:
                        logger.debug(f"Spot coin metrik hesaplama atlandı ({symbol}): {str(e)}")
                        continue
                        
        except Exception as e:
            logger.error(f"Spot market verisi alınamadı: {str(e)}")
        
        # FUTURES marketleri al
        try:
            futures_markets = binance.futures_client.exchange_info()
            for market in futures_markets['symbols']:
                symbol = market['symbol']
                if (market['contractType'] == 'PERPETUAL' and 
                    market['status'] == 'TRADING' and
                    'BULL' not in symbol and
                    'BEAR' not in symbol and
                    'UP' not in symbol and
                    'DOWN' not in symbol):
                    try:
                        ticker = binance.futures_client.ticker_price(symbol=symbol)
                        
                        # Gerekli verilerin varlığını kontrol et
                        if (ticker.get('price') is None or 
                            ticker.get('priceChangePercent') is None or 
                            ticker.get('highPrice') is None or 
                            ticker.get('lowPrice') is None):
                            continue
                        
                        volume_usdt = float(market['volume'])
                        price_change = float(market['priceChangePercent'])
                        high = float(market['highPrice'])
                        low = float(market['lowPrice'])
                        
                        if low == 0:
                            continue
                            
                        volatility = (high - low) / low * 100
                        
                        # Minimum hacim filtresi (100,000 USDT)
                        if volume_usdt < 100000:
                            continue
                        
                        # Metrik skoru hesapla
                        volume_score = min(volume_usdt / 1000000, 10)
                        change_score = abs(price_change)
                        volatility_score = min(volatility, 20)
                        
                        total_score = volume_score + change_score + volatility_score
                        
                        pair_metrics.append({
                            'symbol': symbol,
                            'volume': volume_usdt,
                            'price_change': price_change,
                            'volatility': volatility,
                            'score': total_score,
                            'type': 'futures'
                        })
                        
                    except Exception as e:
                        logger.debug(f"Futures coin metrik hesaplama atlandı ({symbol}): {str(e)}")
                        continue
                        
        except Exception as e:
            logger.error(f"Futures market verisi alınamadı: {str(e)}")
        
        # Skorlara göre sırala
        pair_metrics.sort(key=lambda x: x['score'], reverse=True)
        
        # En iyi 100 coini seç
        selected_pairs = pair_metrics[:100]
        
        if not selected_pairs:
            logger.warning("Hiç uygun coin bulunamadı! Varsay��lan listeyi kullanıyorum...")
            return ["BTC/USDT", "ETH/USDT", "BNB/USDT"]
        
        # Sonuçları logla
        logger.info("\nSeçilen En İyi 10 Coin:")
        for pair in selected_pairs[:10]:
            logger.info(f"{pair['symbol']} ({pair['type']}):")
            logger.info(f"  24s Hacim: {pair['volume']:,.0f} USDT")
            logger.info(f"  Fiyat Değişimi: %{pair['price_change']:.2f}")
            logger.info(f"  Volatilite: %{pair['volatility']:.2f}")
            logger.info(f"  Toplam Skor: {pair['score']:.2f}")
        
        # Market tiplerine göre sayıları logla
        spot_count = sum(1 for p in selected_pairs if p['type'] == 'spot')
        futures_count = sum(1 for p in selected_pairs if p['type'] == 'futures')
        logger.info(f"\nToplam seçilen coin sayısı: {len(selected_pairs)}")
        logger.info(f"Spot: {spot_count}, Futures: {futures_count}")
        
        return [pair['symbol'] for pair in selected_pairs]
        
    except Exception as e:
        logger.error(f"Coin listesi güncelleme hatası: {str(e)}")
        return ["BTC/USDT", "ETH/USDT", "BNB/USDT"]

def fetch_and_analyze_data(symbol):
    try:
        ohlcv = binance.futures_client.fetch_ohlcv(symbol, '1h', limit=100)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df = calculate_technical_indicators(df)
        return df
    except Exception as e:
        print(f"Error fetching and analyzing data for {symbol}: {e}")
        return None

def calculate_market_conditions(symbol):
    """
    Market koşullarını hesapla
    """
    try:
        # 24 saatlik ticker bilgisi al
        if symbol.endswith('USDT'):
            ticker = binance.spot_client.get_ticker(symbol=symbol)
        else:
            ticker = binance.futures_client.ticker_price(symbol=symbol)
        
        # RSI hesapla
        ohlcv = binance.futures_client.fetch_ohlcv(symbol, '1h', limit=100)
        if not ohlcv:
            return None
            
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        rsi = calculate_rsi(df['close'])
        
        # Market koşullarını döndür
        return {
            'price_change': float(ticker['priceChangePercent']),  # 24s fiyat değişimi
            'volume_change': float(ticker['quoteVolume']),  # 24s hacim değişimi
            'rsi': float(rsi)  # RSI değeri
        }
        
    except Exception as e:
        logger.error(f"Market koşulları hesaplama hatası ({symbol}): {str(e)}")
        return None

def calculate_total_score(symbol, market_data, market_type='spot'):
    """Coin için toplam skoru hesapla"""
    try:
        total_score = 0
        weights = {
            'technical': 0.4,    # Teknik analiz ağırlığı
            'ai': 0.3,          # AI tahmin ağırlığı
            'market': 0.3       # Market koşulları ağırlığı
        }
        
        # Log ekle
        logger.info(f"Hesaplanan market verileri: {market_data}")
        
        # Teknik analiz skoru (0-100)
        technical_score = 0
        if float(market_data['rsi']) < 30:
            technical_score += 30  # Aşırı satım
        elif float(market_data['rsi']) > 70:
            technical_score += 20  # Aşırı alım
            
        logger.info(f"Teknik Skor: {technical_score}")
        
        if float(market_data['price_change']) > 0:
            technical_score += min(40, float(market_data['price_change']) * 2)
            
        if float(market_data['volume_change']) > 50:
            technical_score += 30
        
        # AI tahmin skoru (0-100)
        ai_score = float(market_data.get('ai_confidence', 0)) * 100
        logger.info(f"AI Skoru: {ai_score}")
        
        # Market koşulları skoru (0-100)
        market_score = 0
        volatility = calculate_volatility(symbol)
        if 0.5 <= volatility <= 2.0:  # İdeal volatilite aralığı
            market_score += 40
            
        trend_strength = calculate_trend_strength(market_data)
        market_score += trend_strength * 30
        
        if analyze_time_conditions()['suitable_for_trading']:
            market_score += 30
        
        # Toplam skoru hesapla
        total_score = (
            weights['technical'] * technical_score +
            weights['ai'] * ai_score +
            weights['market'] * market_score
        )
        
        # Log ekle
        logger.info(f"Hesaplanan toplam puan: {total_score}")
        
        return max(0, min(100, total_score))  # 0-100 aralığında sınırla
        
    except Exception as e:
        logger.error(f"Skor hesaplama hatası ({symbol}): {str(e)}")
        return 0

# Global instance'ları oluştur
money_manager = MoneyManager()
regime_analyzer = MarketRegimeAnalyzer()
mtf_analyzer = MultiTimeframeAnalyzer()
profit_optimizer = ProfitOptimizer()
performance_tracker = PerformanceTracker()
ai_trader = AITrader()

class TradingBot:
    def __init__(self):
        self.is_running = False
        self.total_profit = 0
        self.open_positions = []
        self.mode = None  # '1': Spot, '2': Futures, '3': Both
        
    def set_trading_mode(self, mode):
        """
        Trading modunu ayarla ve doğrula
        
        :param mode: Trading modu (1: Spot, 2: Futures, 3: Her İkisi)
        """
        if mode not in [1, 2, 3]:
            logger.error(f"Geçersiz trading modu: {mode}")
            raise ValueError("Trading modu 1, 2 veya 3 olmalıdır")
        
        self.mode = mode
        
        # .env dosyasını güncelle
        with open('.env', 'r') as f:
            lines = f.readlines()
        
        with open('.env', 'w') as f:
            for line in lines:
                if not line.startswith('TRADING_MODE='):
                    f.write(line)
            
            f.write(f"\nTRADING_MODE={mode}")
        
        logger.info(f"Trading modu başarıyla ayarlandı: {mode}")
    
    def display_balances(self):
        """
        Hesap bakiyelerini göster
        """
        try:
            # Bakiyeleri al
            balances = binance.get_balances()
            
            print("\n=== HESAP BAKİYELERİ ===")
            
            # Spot hesabı bakiyeleri
            if balances['spot']:
                print("\nSPOT HESABI:")
                for balance in balances['spot']:
                    print(f"{balance['asset']}: "
                          f"Serbest: {balance['free']:.4f}, "
                          f"Kilitli: {balance['locked']:.4f}, "
                          f"Toplam: {balance['total']:.4f}")
            else:
                print("\nSPOT HESABI: Bakiye bulunamadı.")
            
            # Futures hesabı bakiyeleri
            if balances['futures']:
                print("\nFUTURES HESABI:")
                for balance in balances['futures']:
                    print(f"{balance['asset']}: "
                          f"Cüzdan Bakiyesi: {balance['wallet_balance']:.4f}, "
                          f"Gerçekleşmemiş Kar: {balance['unrealized_profit']:.4f}")
            else:
                print("\nFUTURES HESABI: Bakiye bulunamadı.")
        
        except Exception as e:
            logger.error(f"Bakiye görüntüleme hatası: {str(e)}")
            print("Bakiyeleri görüntülerken bir hata oluştu. Lütfen daha sonra tekrar deneyin.")
    
    def display_open_positions(self):
        """Açık pozisyonları göster"""
        try:
            print("\n=== AÇIK POZİSYONLAR ===")
            
            # Futures pozisyonları
            if self.mode in ['2', '3']:
                positions = binance.futures_client.fetch_positions()
                active_positions = [p for p in positions if float(p['contracts']) > 0]
                
                if not active_positions:
                    logger.info("\nFutures: Açık pozisyon bulunmuyor")
                else:
                    logger.info("\nFUTURES POZİSYONLARI:")
                    for pos in active_positions:
                        entry_price = float(pos['entryPrice'])
                        current_price = float(binance.futures_client.fetch_ticker(pos['symbol'])['last'])
                        contracts = float(pos['contracts'])
                        
                        if pos['side'] == 'long':
                            pnl_percent = ((current_price - entry_price) / entry_price * 100) * (1 if pos['side'] == 'long' else -1)
                        else:
                            pnl_percent = ((entry_price - current_price) / entry_price * 100) * (1 if pos['side'] == 'long' else -1)
                        
                        logger.info(f"\nSembol: {pos['symbol']}")
                        logger.info(f"Yön: {'LONG' if pos['side'] == 'long' else 'SHORT'}")
                        logger.info(f"Miktar: {contracts}")
                        logger.info(f"Giriş Fiyatı: {entry_price:.8f}")
                        logger.info(f"Mevcut Fiyat: {current_price:.8f}")
                        logger.info(f"Kar/Zarar: %{pnl_percent:.2f}")
            
            # Spot pozisyonları
            if self.mode in ['1', '3']:
                spot_balances = binance.spot_client.get_account()
                logger.info("\nSPOT POZİSYONLARI:")
                for currency, balance in spot_balances['total'].items():
                    if float(balance) > 0 and currency != 'USDT':
                        ticker = binance.spot_client.get_ticker(f"{currency}/USDT")
                        current_price = float(ticker['last'])
                        position_value = float(balance) * current_price
                        logger.info(f"\n{currency}:")
                        logger.info(f"Miktar: {float(balance):.8f}")
                        logger.info(f"Mevcut Fiyat: {current_price:.8f}")
                        logger.info(f"Toplam Değer: {position_value:.2f} USDT")
            
            print("\n=====================")
        except Exception as e:
            logger.error(f"Pozisyon görüntüleme hatası: {str(e)}")
    
    def close_all_positions(self):
        """Tüm pozisyonları kapat"""
        try:
            # Futures pozisyonlarını kapat
            if self.mode in ['2', '3']:
                positions = binance.futures_client.fetch_positions()
                active_positions = [p for p in positions if float(p['contracts']) > 0]
                
                if not active_positions:
                    logger.info("\nKapatılacak futures pozisyonu yok")
                else:
                    logger.info("\n=== FUTURES POZİSYONLARI KAPATILIYOR ===")
                    for pos in active_positions:
                        try:
                            close_side = 'sell' if pos['side'] == 'long' else 'buy'
                            
                            order = binance.futures_client.create_order(
                                symbol=pos['symbol'],
                                type='MARKET',
                                side=close_side,
                                amount=float(pos['contracts'])
                            )
                            
                            logger.info(f"{pos['symbol']} pozisyonu kapatıldı")
                        except Exception as e:
                            logger.error(f"{pos['symbol']} pozisyonu kapatılamadı: {e}")
            
            # Spot pozisyonlarını kapat
            if self.mode in ['1', '3']:
                spot_balances = binance.spot_client.get_account()
                logger.info("\n=== SPOT POZİSYONLARI KAPATILIYOR ===")
                
                for currency, balance in spot_balances['total'].items():
                    if float(balance) > 0 and currency != 'USDT':
                        try:
                            order = binance.spot_client.create_market_sell_order(
                                symbol=f"{currency}/USDT",
                                amount=float(balance)
                            )
                            logger.info(f"{currency} spot pozisyonu kapatıldı")
                        except Exception as e:
                            logger.error(f"{currency} spot pozisyonu kapatılamadı: {e}")
            
            logger.info("\nTüm pozisyonlar kapatıldı")
            print("=====================")
        except Exception as e:
            logger.error(f"Pozisyon kapatma hatası: {str(e)}")
    
    def display_total_pnl(self):
        """Toplam kar/zarar durumunu göster"""
        try:
            total_pnl = 0
            
            # Futures kar/zarar
            if self.mode in ['2', '3']:
                positions = binance.futures_client.fetch_positions()
                futures_pnl = sum([float(p['unrealizedPnl']) for p in positions])
                total_pnl += futures_pnl
                logger.info("\n=== FUTURES KAR/ZARAR ===")
                logger.info(f"Unrealized P/L: {futures_pnl:.2f} USDT")
            
            # Spot kar/zarar
            if self.mode in ['1', '3']:
                spot_balances = binance.spot_client.get_account()
                spot_pnl = 0
                for currency, balance in spot_balances['total'].items():
                    if float(balance) > 0 and currency != 'USDT':
                        ticker = binance.spot_client.get_ticker(f"{currency}/USDT")
                        current_value = float(balance) * float(ticker['last'])
                        spot_pnl += current_value
                
                total_pnl += spot_pnl
                logger.info("\n=== SPOT KAR/ZARAR ===")
                logger.info(f"Toplam Değer: {spot_pnl:.2f} USDT")
            
            logger.info("\n=== TOPLAM KAR/ZARAR ===")
            logger.info(f"Toplam: {total_pnl:.2f} USDT")
            logger.info(f"Realized P/L: {self.total_profit:.2f} USDT")
            logger.info(f"Genel Toplam: {(total_pnl + self.total_profit):.2f} USDT")
            print("=====================")
        except Exception as e:
            logger.error(f"Kar/Zarar hesaplama hatası: {str(e)}")

    def start_bot(self):
        """Botu başlat"""
        if not self.mode:
            logger.error("Trading modu ayarlanmamış!")
            return
            
        if self.is_running:
            logger.warning("Bot zaten çalışıyor!")
            return
            
        self.is_running = True
        logger.info("\nBot başlatılıyor...")
        logger.info(f"İşlem stratejisi: {'Spot' if self.mode == '1' else 'Futures' if self.mode == '2' else 'Spot + Futures'}")
        logger.info("Minimum işlem: 10 USDT")
        logger.info("AI Trader: Aktif")
        logger.info("Risk Yönetimi: Aktif")
        
        while self.is_running:
            try:
                # Mevcut durumu göster
                self.display_balances()
                self.display_open_positions()
                self.display_total_pnl()
                
                # Sembol listesini güncelle
                symbols = get_active_trading_pairs()
                logger.info("\nPiyasa Analizi Başlıyor...")
                logger.info(f"Analiz Edilecek Coin Sayısı: {len(symbols)}")
                
                for symbol in symbols:
                    try:
                        # Market türünü belirle
                        market_type = 'futures' if ':USDT' in symbol else 'spot'
                        
                        # Temel analiz
                        market_conditions = calculate_market_conditions(symbol)
                        if not market_conditions:
                            continue
                            
                        # Toplam skoru hesapla
                        total_score = calculate_total_score(symbol, market_conditions)
                        
                        # Sonuçları logla
                        logger.info(f"\n{symbol} Analiz Sonuçları:")
                        logger.info(f"Fiyat Değişimi (24s): %{market_conditions['price_change']:.2f}")
                        logger.info(f"Hacim Değişimi: %{market_conditions['volume_change']:.2f}")
                        logger.info(f"RSI: {market_conditions['rsi']:.2f}")
                        logger.info(f"Toplam Puan: {total_score}")
                        
                        # İşlem sinyali varsa
                        if abs(total_score) >= 2:  # Alış sinyali
                            logger.info(f"\n{symbol} için {'ALIŞ' if total_score > 0 else 'SATIŞ'} sinyali!")
                            
                            # İşlemi gerçekleştir
                            execute_trade(symbol, 1 if total_score > 0 else -1, 10, market_type)
                    except Exception as e:
                        logger.error(f"{symbol} analiz hatası: {str(e)}")
                        continue
                
                logger.info("\n5 dakika bekleniyor...")
                time.sleep(300)
                
            except Exception as e:
                logger.error(f"Bot çalışma hatası: {e}")
                time.sleep(10)
                
                # Bağlantıyı kontrol et ve gerekirse yeniden bağlan
                try:
                    binance.spot_client.fetch_balance()
                except:
                    logger.warning("Bağlantı koptu, yeniden bağlanılıyor...")
                    binance.reconnect()
    
    def stop_bot(self):
        """Botu durdur"""
        if not self.is_running:
            logger.warning("Bot zaten durdurulmuş!")
            return
            
        self.is_running = False
        logger.info("Bot durduruluyor...")
        logger.info("Lütfen açık pozisyonları kontrol edin")

if __name__ == "__main__":
    try:
        # Bot instance'ı oluştur
        bot = TradingBot()
        
        # Trading modunu ayarla
        bot.set_trading_mode(choose_trading_mode())  # Otomatik mod seçimi
        
        while True:
            try:
                print("\n=== TRADING BOT MENU ===")
                print("1. Botu Başlat")
                print("2. Botu Durdur")
                print("3. Bakiyeleri Göster")
                print("4. Açık Pozisyonları Göster")
                print("5. Tüm Pozisyonları Kapat")
                print("6. Toplam Kar/Zarar")
                print("7. Çıkış")
                print("=====================")
                
                choice = input("\nSeçiminiz (1-7): ")
                
                if choice == '1':
                    bot.start_bot()
                elif choice == '2':
                    bot.stop_bot()
                elif choice == '3':
                    bot.display_balances()
                elif choice == '4':
                    bot.display_open_positions()
                elif choice == '5':
                    confirm = input("Tüm pozisyonları kapatmak istediğinize emin misiniz? (e/h): ")
                    if confirm.lower() == 'e':
                        bot.close_all_positions()
                elif choice == '6':
                    bot.display_total_pnl()
                elif choice == '7':
                    logger.info("\nBot kapatılıyor...")
                    if bot.is_running:
                        bot.stop_bot()
                    break
                else:
                    logger.warning("\nGeçersiz seçim!")
                    
            except Exception as e:
                logger.error(f"\nHata: {e}")
                time.sleep(2)
                
    except KeyboardInterrupt:
        logger.info("\nBot kullanıcı tarafından durduruldu.")
        if bot.is_running:
            bot.stop_bot()
    except Exception as e:
        logger.error(f"\nKritik hata: {e}")
        if bot and bot.is_running:
            bot.stop_bot()

def check_min_trade_requirements(symbol, amount, market_type='spot'):
    """
    Minimum işlem gereksinimlerini kontrol et
    """
    try:
        if market_type == 'spot':
            client = binance.spot_client
        else:
            client = binance.futures_client
            
        # Market bilgilerini al
        market = client.load_markets()
        symbol_info = market[symbol]
        
        # Minimum işlem miktarı kontrolü
        min_amount = symbol_info.get('limits', {}).get('amount', {}).get('min', 0)
        min_cost = symbol_info.get('limits', {}).get('cost', {}).get('min', 0)
        
        # Güncel fiyatı al
        if symbol.endswith('USDT'):
            ticker = client.get_ticker(symbol=symbol)
        else:
            ticker = client.ticker_price(symbol=symbol)
        current_price = float(ticker['price'])
        
        # İşlem miktarını hesapla
        quantity = amount / current_price
        
        # Kontroller
        if quantity < min_amount:
            logger.warning(f"Minimum işlem miktarı: {min_amount} {symbol.split('/')[0]}")
            return False
            
        if amount < min_cost:
            logger.warning(f"Minimum işlem tutarı: {min_cost} USDT")
            return False
            
        # Bakiye kontrolü
        if market_type == 'spot':
            balance = float(binance.spot_client.get_account()['total']['USDT'])
        else:
            balance = float(binance.futures_client.get_account()['USDT']['total'])
            
        if balance < amount:
            logger.warning(f"Yetersiz bakiye! Mevcut: {balance:.2f} USDT, Gerekli: {amount:.2f} USDT")
            return False
            
        return True
        
    except Exception as e:
        logger.error(f"İşlem gereksinimi kontrolü hatası: {str(e)}")
        return False

def execute_trade(symbol, direction, amount, market_type='spot'):
    """
    İşlem gerçekleştir
    """
    try:
        logger.info("\n=== İŞLEM BAŞLATILIYOR ===")
        logger.info(f"Symbol: {symbol}")
        logger.info(f"Yön: {'LONG/BUY' if direction == 1 else 'SHORT/SELL'}")
        logger.info(f"Miktar: {amount} USDT")
        logger.info(f"Piyasa: {market_type.upper()}")
        
        # Hesap bilgilerini logla
        debug_account_info()
        
        # Minimum gereksinimleri kontrol et
        if not check_min_trade_requirements(symbol, amount, market_type):
            return False
            
        # Risk kontrolü
        if not money_manager.check_risk_limits(symbol, amount, market_type):
            return False
            
        if market_type == 'futures':
            client = binance.futures_client
        else:
            client = binance.spot_client
            
        # Market verilerini al
        if symbol.endswith('USDT'):
            ticker = client.get_ticker(symbol=symbol)
        else:
            ticker = client.ticker_price(symbol=symbol)
        current_price = ticker['price']
        
        # İşlem tipi ve miktarı belirle
        side = 'buy' if direction == 1 else 'sell'
        quantity = amount / current_price
        
        # İşlemi gerçekleştir
        order = client.create_order(
            symbol=symbol,
            type='MARKET',
            side=side,
            amount=quantity
        )
        
        logger.info("\n=== İŞLEM BAŞARILI ===")
        logger.info(f"Order ID: {order['id']}")
        logger.info(f"İşlem Fiyatı: {float(order['price']) if order.get('price') else current_price:.8f}")
        logger.info(f"Gerçekleşen Miktar: {float(order['filled']):.8f}")
        
        # Stop-loss ve take-profit orderlarını ekle
        if market_type == 'futures':
            # Stop-loss order
            client.create_order(
                symbol=symbol,
                type='STOP_MARKET',
                side='sell' if direction == 1 else 'buy',
                amount=quantity,
                params={'stopPrice': stop_loss}
            )
            
            # Take-profit order
            client.create_order(
                symbol=symbol,
                type='TAKE_PROFIT_MARKET',
                side='sell' if direction == 1 else 'buy',
                amount=quantity,
                params={'stopPrice': take_profit}
            )
            
            logger.info("Stop-loss ve take-profit orderları eklendi")
        
        # İşlemi kaydet
        trade_info = {
            'symbol': symbol,
            'side': 'buy' if direction == 1 else 'sell',
            'amount': amount,
            'price': current_price,
            'market_type': market_type,
            'timestamp': datetime.now().timestamp()
        }
        
        if market_type == 'futures':
            trade_info.update({
                'leverage': leverage,
                'stop_loss': stop_loss,
                'take_profit': take_profit
            })
            
        save_trade_history(trade_info)
        
        return True
        
    except Exception as e:
        logger.error(f"\n=== İŞLEM HATASI ===")
        logger.error(f"Hata: {str(e)}")
        logger.error(f"Detay: {e.__class__.__name__}")
        return False

def debug_account_info():
    """Hesap bilgilerini detaylı logla"""
    try:
        # Spot bakiye
        spot_balance = binance.spot_client.get_account()
        logger.info("\n=== SPOT HESAP BİLGİLERİ ===")
        logger.info(f"Spot Balance Type: {type(spot_balance)}")
        logger.info(f"Spot Balance Raw Data: {spot_balance}")  # Debug raw data
        
        # Debugging: Print out all keys and their types
        if isinstance(spot_balance, dict):
            for key, value in spot_balance.items():
                logger.info(f"Key: {key}, Type: {type(value)}")
        
        # Safely handle different possible data structures
        if isinstance(spot_balance, dict) and isinstance(spot_balance.get('balances', []), list):
            for asset_info in spot_balance['balances']:
                free_balance = float(asset_info.get('free', 0))
                locked_balance = float(asset_info.get('locked', 0))
                total_balance = free_balance + locked_balance
                if total_balance > 0:
                    logger.info(f"{asset_info.get('asset', 'Unknown')}: Free: {free_balance:.8f}, Locked: {locked_balance:.8f}")
        else:
            logger.warning(f"Unexpected spot balance format: {type(spot_balance)}")
                
        # Futures bakiye
        futures_balance = binance.futures_client.get_account()
        logger.info("\n=== FUTURES HESAP BİLGİLERİ ===")
        logger.info(f"Futures Balance Type: {type(futures_balance)}")
        logger.info(f"Futures Balance Raw Data: {futures_balance}")  # Debug raw data
        
        # Safely handle futures balance
        if isinstance(futures_balance, dict) and 'USDT' in futures_balance:
            logger.info(f"USDT Bakiye: {float(futures_balance['USDT']['total']):.2f}")
        else:
            logger.warning(f"Unexpected futures balance format: {type(futures_balance)}")
        
        # Açık pozisyonlar
        positions = binance.futures_client.fetch_positions()
        logger.info("\n=== AÇIK POZİSYONLAR ===")
        logger.info(f"Positions Type: {type(positions)}")
        logger.info(f"Positions Raw Data: {positions}")  # Debug raw data
        
        for pos in positions:
            if isinstance(pos, dict) and float(pos.get('contracts', 0)) > 0:
                logger.info(f"Symbol: {pos.get('symbol', 'Unknown')}")
                logger.info(f"Yön: {'LONG' if pos.get('side') == 'long' else 'SHORT'}")
                logger.info(f"Miktar: {float(pos.get('contracts', 0)):.8f}")
                logger.info(f"Kaldıraç: {pos.get('leverage', 'N/A')}x")
                logger.info(f"PNL: {float(pos.get('unrealizedPnl', 0)):.2f} USDT")
                
    except Exception as e:
        logger.error(f"Hesap bilgileri alınamadı: {str(e)}", exc_info=True)  # Added exc_info for full traceback

class TradingBot:
    def __init__(self):
        self.is_running = False
        self.total_profit = 0
        self.open_positions = []
        self.mode = None  # '1': Spot, '2': Futures, '3': Both
        
        # Alt sistemleri başlat
        self.money_manager = MoneyManager()
        self.regime_analyzer = MarketRegimeAnalyzer()
        self.mtf_analyzer = MultiTimeframeAnalyzer()
        self.profit_optimizer = ProfitOptimizer()
        self.ai_trader = AITrader()
        
    def analyze_market(self, symbol: str) -> dict:
        """
        Tüm market analizlerini tek bir fonksiyonda topla
        """
        data = fetch_and_analyze_data(symbol)
        
        # Teknik analiz
        technical_indicators = calculate_technical_indicators(data)
        
        # AI sinyalleri
        ai_signals = self.ai_trader.predict_rf(technical_indicators)
        
        # Market rejimi
        market_regime = self.regime_analyzer.analyze_regime(symbol)
        
        # Çoklu zaman dilimi analizi
        mtf_signals = self.mtf_analyzer.analyze(symbol)
        
        return {
            'technical': technical_indicators,
            'ai_signals': ai_signals,
            'market_regime': market_regime,
            'mtf_signals': mtf_signals
        }
    
    def should_trade(self, symbol: str, analysis_results: dict) -> tuple:
        """
        Tüm trading koşullarını kontrol et
        """
        # Zaman koşulları
        if not analyze_time_conditions():
            return False, None
            
        # Market koşulları
        market_score = calculate_market_conditions(symbol)
        if market_score < 50:
            return False, None
            
        # Trading sinyalleri
        signals = generate_signals(symbol, analysis_results)
        if not signals['should_trade']:
            return False, None
            
        return True, signals['direction']
    
    def execute_strategy(self, symbol: str):
        """
        Trading stratejisini uygula
        """
        try:
            # Market analizi
            analysis = self.analyze_market(symbol)
            
            # Trading kararı
            should_trade, direction = self.should_trade(symbol, analysis)
            if not should_trade:
                return
                
            # Risk yönetimi
            balance = get_account_balance(self.mode)
            position_size = self.money_manager.calculate_position_size(balance)
            
            # İşlemi gerçekleştir
            if execute_trade(symbol, direction, position_size, 'spot' if self.mode == '1' else 'futures'):
                logger.info(f"Strategy executed successfully for {symbol}")
                
        except Exception as e:
            logger.error(f"Strategy execution error for {symbol}: {str(e)}")

def generate_signals(symbol, data, timeframe='1h'):
    """
    Teknik analiz sinyalleri üret
    """
    try:
        signals = {
            'LONG': False,
            'SHORT': False,
            'strength': 0,
            'reasons': []
        }
        
        # Temel veriler
        price_change = float(data['price_change'])
        volume_change = float(data['volume_change'])
        rsi = float(data['rsi'])
        
        # Log ekle
        logger.info(f"RSI: {rsi}, Price Change: {price_change}, Volume Change: {volume_change}")
        
        # RSI Sinyalleri
        if rsi < 30:
            signals['LONG'] = True
            signals['strength'] += 2
            signals['reasons'].append(f"RSI aşırı satım ({rsi:.2f})")
        elif rsi > 70:
            signals['SHORT'] = True
            signals['strength'] += 2
            signals['reasons'].append(f"RSI aşırı alım ({rsi:.2f})")
            
        # Fiyat Değişimi Sinyalleri
        if price_change > 5:  # %5'den fazla artış
            if volume_change > 50:  # Hacim de artıyorsa
                signals['LONG'] = True
                signals['strength'] += 1.5
                signals['reasons'].append(f"Güçlü yükseliş trendi (%{price_change:.1f})")
        elif price_change < -5:  # %5'den fazla düşüş
            if volume_change > 50:  # Hacim artışıyla
                signals['SHORT'] = True
                signals['strength'] += 1.5
                signals['reasons'].append(f"Güçlü düşüş trendi (%{price_change:.1f})")
                
        # MACD Sinyalleri
        macd_data = calculate_macd(symbol, timeframe)
        if macd_data['signal'] == 'BUY':
            signals['LONG'] = True
            signals['strength'] += 1.5
            signals['reasons'].append("MACD kesişimi (LONG)")
        elif macd_data['signal'] == 'SELL':
            signals['SHORT'] = True
            signals['strength'] += 1.5
            signals['reasons'].append("MACD kesişimi (SHORT)")
            
        # Hareketli Ortalama Sinyalleri
        ma_data = check_moving_averages(symbol, timeframe)
        if ma_data['signal'] == 'BUY':
            signals['LONG'] = True
            signals['strength'] += 1
            signals['reasons'].append(f"MA kesişimi (LONG)")
        elif ma_data['signal'] == 'SELL':
            signals['SHORT'] = True
            signals['strength'] += 1
            signals['reasons'].append(f"MA kesişimi (SHORT)")
            
        # Bollinger Bant Sinyalleri
        bb_data = check_bollinger_bands(symbol, timeframe)
        if bb_data['signal'] == 'BUY':
            signals['LONG'] = True
            signals['strength'] += 1
            signals['reasons'].append("Bollinger alt bant teması")
        elif bb_data['signal'] == 'SELL':
            signals['SHORT'] = True
            signals['strength'] += 1
            signals['reasons'].append("Bollinger üst bant teması")
            
        return signals
        
    except Exception as e:
        logger.error(f"Sinyal üretme hatası: {str(e)}")
        return {'LONG': False, 'SHORT': False, 'strength': 0, 'reasons': []}

def check_moving_averages(symbol, timeframe='1h'):
    """
    Hareketli ortalama kesişimlerini kontrol et
    """
    try:
        # Veri al
        data = binance.spot_client.fetch_ohlcv(symbol, timeframe, limit=100)
        if not data:
            return {'signal': 'NEUTRAL', 'fast_ma': 0, 'slow_ma': 0}
            
        # Kapanış fiyatlarını al
        closes = [x[4] for x in data]
        
        # Hareketli ortalamaları hesapla
        fast_ma = calculate_moving_average(closes, 10)  # 10 periyot
        mid_ma = calculate_moving_average(closes, 20)   # 20 periyot
        slow_ma = calculate_moving_average(closes, 50)  # 50 periyot
        
        # Kesişimleri kontrol et
        if fast_ma > mid_ma and fast_ma > slow_ma:
            return {'signal': 'BUY', 'fast_ma': fast_ma, 'slow_ma': slow_ma}
            
        elif fast_ma < mid_ma and fast_ma < slow_ma:
            return {'signal': 'SELL', 'fast_ma': fast_ma, 'slow_ma': slow_ma}
            
        return {'signal': 'NEUTRAL', 'fast_ma': 0, 'slow_ma': 0}
        
    except Exception as e:
        logger.error(f"MA kontrolü hatası: {str(e)}")
        return {'signal': 'NEUTRAL', 'fast_ma': 0, 'slow_ma': 0}

def check_bollinger_bands(symbol, timeframe='1h'):
    """
    Bollinger Bant sinyalleri
    """
    try:
        period = 20
        std_dev = 2
        
        # Fiyat verileri
        ohlcv = binance.spot_client.fetch_ohlcv(symbol, timeframe)
        closes = [x[4] for x in ohlcv]
        
        # BB hesaplamaları
        middle_band = calculate_moving_average(closes, period)
        std = statistics.stdev(closes[-period:])
        
        upper_band = middle_band + (std_dev * std)
        lower_band = middle_band - (std_dev * std)
        
        current_price = closes[-1]
        
        # Sinyal kontrolleri
        if current_price <= lower_band:
            return {'signal': 'BUY', 'price': current_price, 'lower': lower_band}
        elif current_price >= upper_band:
            return {'signal': 'SELL', 'price': current_price, 'upper': upper_band}
            
        return {'signal': 'NEUTRAL', 'price': current_price}
        
    except Exception as e:
        logger.error(f"Bollinger hesaplama hatası: {str(e)}")
        return {'signal': 'NEUTRAL', 'price': 0}

def should_open_position(symbol, data, market_type='spot'):
    """
    İşlem açma koşullarını kontrol et
    """
    try:
        # Sinyal kontrolü
        signals = generate_signals(symbol, data)
        
        if signals['strength'] < 3:  # Minimum sinyal gücü
            logger.info(f"{symbol} - Yetersiz sinyal gücü: {signals['strength']}")
            return False
            
        # Yön belirleme
        if signals['LONG'] and not signals['SHORT']:
            direction = 1  # LONG
        elif signals['SHORT'] and not signals['LONG']:
            direction = -1  # SHORT
        else:
            logger.info(f"{symbol} - Net sinyal yok")
            return False
            
        # Diğer kontroller...
        min_volume = 1000000  # Minimum 24s hacim
        volume = float(data['volume'])
        
        if volume < min_volume:
            logger.info(f"{symbol} - Yetersiz hacim: {volume:.2f}")
            return False
            
        # Market koşullarını kontrol et
        market_conditions = calculate_market_conditions(symbol)
        if market_conditions:
            # Skor hesapla
            score = calculate_score(
                symbol=symbol,
                price_change=market_conditions['price_change'],
                volume_change=market_conditions['volume_change'],
                rsi=market_conditions['rsi']
            )
            
            if score < 30:  # Minimum skor şartı
                logger.info(f"{symbol} - Yetersiz skor: {score}")
                return False
                
        # Sinyal detaylarını logla
        logger.info(f"\n=== {symbol} SİNYAL DETAYLARI ===")
        logger.info(f"Sinyal Yönü: {'LONG' if direction == 1 else 'SHORT'}")
        logger.info(f"Sinyal Gücü: {signals['strength']}")
        logger.info("Sebepler:")
        for reason in signals['reasons']:
            logger.info(f"- {reason}")
            
        return direction
        
    except Exception as e:
        logger.error(f"İşlem koşulu kontrolü hatası: {str(e)}")
        return 0

def calculate_score(symbol: str, price_change: float, volume_change: float, rsi: float, time_score: int = 0) -> float:
    """
    Coin için toplam skoru hesapla
    
    Parametreler:
    - symbol: Coin sembolü
    - price_change: Fiyat değişimi (%)
    - volume_change: Hacim değişimi (%)
    - rsi: RSI değeri
    - time_score: Zaman bazlı skor (opsiyonel)
    
    Dönüş:
    - float: 0-100 arası toplam skor
    """
    try:
        # Fiyat değişimi skoru (0-40 puan)
        price_score = min(max(abs(price_change) * 4, 0), 40)
        
        # Hacim değişimi skoru (0-30 puan)
        volume_score = min(max(volume_change * 3, 0), 30)
        
        # RSI skoru (0-20 puan)
        if rsi <= 30:  # Aşırı satım
            rsi_score = 20
        elif rsi >= 70:  # Aşırı alım
            rsi_score = 0
        else:
            rsi_score = 10
            
        # Zaman skoru (0-10 puan)
        time_score = min(max(time_score, 0), 10)
        
        # Toplam skor
        total_score = price_score + volume_score + rsi_score + time_score
        
        logger.info(f"\n=== SKOR DETAYLARI ({symbol}) ===")
        logger.info(f"Fiyat Skoru: {price_score:.2f}/40")
        logger.info(f"Hacim Skoru: {volume_score:.2f}/30")
        logger.info(f"RSI Skoru: {rsi_score}/20")
        logger.info(f"Zaman Skoru: {time_score}/10")
        logger.info(f"Toplam Skor: {total_score:.2f}/100")
        
        return total_score
        
    except Exception as e:
        logger.error(f"Skor hesaplama hatası: {str(e)}")
        return 0
