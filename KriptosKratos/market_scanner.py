import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
import joblib
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import EMAIndicator, MACD, ADXIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator
import ccxt
from concurrent.futures import ThreadPoolExecutor
import os
from sklearn.ensemble import RandomForestClassifier

class MarketScanner:
    def __init__(self, exchange, config):
        self.exchange = exchange
        self.config = config
        self.scaler = StandardScaler()
        self.ml_model = self._load_or_create_model()
        self.opportunity_scores = {}
        self.model = None

    def _load_or_create_model(self):
        """ML modelini yükle veya oluştur"""
        model_path = 'ml_model.h5'
        if os.path.exists(model_path):
            return load_model(model_path)
        else:
            return self._create_model()

    def _create_model(self):
        """Yeni ML modeli oluştur"""
        model = Sequential([
            LSTM(100, return_sequences=True, input_shape=(60, 15)),
            Dropout(0.2),
            LSTM(100, return_sequences=False),
            Dropout(0.2),
            Dense(50, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def get_all_futures_symbols(self):
        """Tüm vadeli işlem sembollerini al"""
        try:
            markets = self.exchange.fetch_markets()
            futures = []
            for market in markets:
                if market['type'] == 'future' and market['quote'] == 'USDT':
                    futures.append(market['symbol'])
            print(f"Bulunan vadeli işlem sembolleri: {len(futures)}")
            return futures
        except Exception as e:
            print(f"Sembol listesi alınamadı: {e}")
            return []

    def calculate_volatility_score(self, df):
        """Volatilite skoru hesapla"""
        try:
            atr = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'])
            df['atr'] = atr.average_true_range()
            return (df['atr'].iloc[-1] / df['close'].iloc[-1]) * 100
        except:
            return 0

    def calculate_volume_score(self, df):
        """Hacim skoru hesapla"""
        try:
            avg_volume = df['volume'].mean()
            current_volume = df['volume'].iloc[-1]
            return (current_volume / avg_volume) * 100
        except:
            return 0

    def calculate_momentum_score(self, df):
        """Momentum skoru hesapla"""
        try:
            # RSI
            rsi_ind = RSIIndicator(close=df['close'])
            df['rsi'] = rsi_ind.rsi()
            
            # MACD
            macd = MACD(close=df['close'])
            df['macd'] = macd.macd()
            
            # Momentum (fiyat değişimi)
            df['momentum'] = df['close'].pct_change(periods=14)
            
            rsi_score = (df['rsi'].iloc[-1] - 50) / 50
            macd_score = 1 if df['macd'].iloc[-1] > 0 else -1
            mom_score = 1 if df['momentum'].iloc[-1] > 0 else -1
            
            return (rsi_score + macd_score + mom_score) / 3 * 100
        except:
            return 0

    def prepare_ml_features(self, df):
        """ML özellikleri hazırla"""
        try:
            features = pd.DataFrame()
            
            # RSI
            rsi = RSIIndicator(close=df['close'])
            features['rsi'] = rsi.rsi()
            
            # MACD
            macd = MACD(close=df['close'])
            features['macd'] = macd.macd()
            
            # ATR
            atr = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'])
            features['atr'] = atr.average_true_range()
            
            # Momentum
            features['mom'] = df['close'].pct_change(periods=14)
            
            # Stochastic
            stoch = StochasticOscillator(high=df['high'], low=df['low'], close=df['close'])
            features['stoch_k'] = stoch.stoch()
            
            # ADX
            adx = ADXIndicator(high=df['high'], low=df['low'], close=df['close'])
            features['adx'] = adx.adx()
            
            # Williams %R (benzer mantıkla hesaplanabilir)
            features['willr'] = (df['high'].rolling(14).max() - df['close']) / (df['high'].rolling(14).max() - df['low'].rolling(14).min()) * -100
            
            # Fiyat değişimleri
            features['price_change'] = df['close'].pct_change()
            features['high_low_range'] = (df['high'] - df['low']) / df['low']
            features['volume_change'] = df['volume'].pct_change()
            
            # Hareketli ortalamalar
            ema20 = EMAIndicator(close=df['close'], window=20)
            ema50 = EMAIndicator(close=df['close'], window=50)
            features['ema_20'] = ema20.ema_indicator()
            features['ema_50'] = ema50.ema_indicator()
            
            # NaN değerleri temizle
            features = features.fillna(0)
            
            # Normalize et
            features_scaled = self.scaler.fit_transform(features)
            
            # Son 60 veriyi al
            if len(features_scaled) > 60:
                return features_scaled[-60:].reshape(1, 60, 15)
            return None
        except:
            return None

    def predict_opportunity(self, df):
        """ML modeli ile fırsat tahmini yap"""
        try:
            X = self.prepare_ml_features(df)
            if X is not None:
                return self.ml_model.predict(X, verbose=0)[0][0]
            return 0
        except:
            return 0

    def calculate_opportunity_score(self, symbol):
        """Her sembol için fırsat skoru hesapla"""
        try:
            # Market verilerini al
            ohlcv = self.exchange.fetch_ohlcv(symbol, '5m', limit=100)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # Skorları hesapla
            volatility_score = self.calculate_volatility_score(df)
            volume_score = self.calculate_volume_score(df)
            momentum_score = self.calculate_momentum_score(df)
            ml_score = self.calculate_ml_prediction(df)
            
            # Son 24 saatlik hacim kontrolü
            if df['volume'].sum() * df['close'].iloc[-1] < self.config.MIN_VOLUME:
                return None
            
            # Toplam skor
            total_score = (
                volatility_score * 0.4 +  # Volatiliteye daha fazla ağırlık
                volume_score * 0.3 +      # Hacime daha fazla ağırlık
                momentum_score * 0.3      # Momentum'a normal ağırlık
            )
            
            # Minimum skor kontrolü
            if total_score < 20:  # Daha düşük minimum skor
                return None
                
            return {
                'symbol': symbol,
                'total_score': total_score,
                'volatility': volatility_score,
                'volume': volume_score,
                'momentum': momentum_score,
                'ml_prediction': ml_score,
                'last_price': df['close'].iloc[-1]
            }
        except Exception as e:
            print(f"Hata {symbol}: {e}")
            return None

    def scan_markets(self):
        """Tüm piyasaları tara"""
        symbols = self.get_all_futures_symbols()
        opportunities = []
        
        print(f"Toplam {len(symbols)} vadeli işlem sembolü bulundu...")
        
        # Her sembol için fırsat hesapla
        for symbol in symbols:
            try:
                # Market verilerini al
                ohlcv = self.exchange.fetch_ohlcv(symbol, '5m', limit=100)
                if not ohlcv or len(ohlcv) < 100:
                    continue
                    
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                
                # Teknik indikatörleri hesapla
                df = self.calculate_all_indicators(df)
                
                # Skorları hesapla
                volatility_score = self.calculate_volatility_score(df)
                volume_score = self.calculate_volume_score(df)
                momentum_score = self.calculate_momentum_score(df)
                
                # ML tahminini al
                ml_score = self.calculate_ml_prediction(df)
                
                # Son fiyat değişimi (%)
                price_change = ((df['close'].iloc[-1] - df['close'].iloc[-2]) / df['close'].iloc[-2]) * 100
                
                # Son 24 saatlik hacim kontrolü
                daily_volume = df['volume'].sum() * df['close'].iloc[-1]
                if daily_volume < self.config.MIN_VOLUME:
                    continue
                
                # Toplam skor
                total_score = (
                    volatility_score * 0.3 +  # Volatilite
                    volume_score * 0.2 +      # Hacim
                    momentum_score * 0.2 +    # Momentum
                    ml_score * 0.3            # ML tahmini
                )
                
                # Minimum skor kontrolü
                if total_score < 20:
                    continue
                    
                opportunity = {
                    'symbol': symbol,
                    'total_score': total_score,
                    'volatility': volatility_score,
                    'volume': volume_score,
                    'momentum': momentum_score,
                    'ml_prediction': ml_score,
                    'price_change': price_change,
                    'daily_volume': daily_volume,
                    'last_price': df['close'].iloc[-1],
                    'rsi': df['rsi'].iloc[-1],
                    'macd': df['macd'].iloc[-1],
                    'signal': df['macd_signal'].iloc[-1]
                }
                
                opportunities.append(opportunity)
                print(f"Fırsat bulundu: {symbol} (Skor: {total_score:.2f}, Değişim: {price_change:.2f}%, Hacim: {daily_volume/1000000:.1f}M USDT)")
                
            except Exception as e:
                print(f"Hata {symbol}: {e}")
                continue
        
        # Sonuçları sırala
        opportunities.sort(key=lambda x: x['total_score'], reverse=True)
        
        # En iyi fırsatları sakla
        self.opportunity_scores = {
            opp['symbol']: opp for opp in opportunities[:20]
        }
        
        print(f"\nToplam {len(opportunities)} fırsat bulundu")
        print("\nEn İyi 5 Fırsat:")
        for opp in opportunities[:5]:
            print(
                f"{opp['symbol']}:\n"
                f"  Skor: {opp['total_score']:.2f}\n"
                f"  Değişim: {opp['price_change']:.2f}%\n"
                f"  Hacim: {opp['daily_volume']/1000000:.1f}M USDT\n"
                f"  RSI: {opp['rsi']:.2f}\n"
                f"  ML Tahmini: {opp['ml_prediction']:.2f}%\n"
            )
        
        return opportunities[:20]  # En iyi 20 fırsatı döndür

    def get_best_opportunities(self, min_score=50):
        """En iyi fırsatları getir"""
        opportunities = self.scan_markets()
        return [opp for opp in opportunities if opp['total_score'] > min_score]

    def train_ml_model(self):
        """ML modelini eğit"""
        try:
            print("ML modeli eğitimi başlıyor...")
            symbols = self.get_all_futures_symbols()
            training_data = []
            
            for symbol in symbols[:10]:  # İlk 10 sembol ile başla
                try:
                    # Son 500 mum verisi al
                    ohlcv = self.exchange.fetch_ohlcv(symbol, '5m', limit=500)
                    if not ohlcv or len(ohlcv) < 500:
                        continue
                        
                    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    
                    # Teknik indikatörleri hesapla
                    df = self.calculate_all_indicators(df)
                    
                    # Etiketleri hazırla (gelecek 3 mum için yükseliş/düşüş)
                    df['target'] = (df['close'].shift(-3) > df['close']).astype(int)
                    
                    # NaN değerleri temizle
                    df = df.dropna()
                    
                    training_data.append(df)
                    print(f"{symbol} verileri hazırlandı")
                    
                except Exception as e:
                    print(f"Hata {symbol}: {e}")
                    continue
            
            if not training_data:
                print("Eğitim verisi toplanamadı")
                return
                
            # Tüm verileri birleştir
            full_data = pd.concat(training_data)
            
            # Özellikleri ve hedefi ayır
            X = full_data[self.config.FEATURE_COLUMNS]
            y = full_data['target']
            
            # Modeli eğit
            self.model = RandomForestClassifier(n_estimators=100)
            self.model.fit(X, y)
            
            # Modeli kaydet
            joblib.dump(self.model, 'ml_model.joblib')
            print("ML modeli eğitildi ve kaydedildi")
            
        except Exception as e:
            print(f"ML model eğitimi hatası: {e}")

    def calculate_ml_prediction(self, df):
        """ML tahminini hesapla"""
        try:
            if self.model is None:
                if os.path.exists('ml_model.joblib'):
                    self.model = joblib.load('ml_model.joblib')
                else:
                    self.train_ml_model()
                    if self.model is None:
                        return 50  # Model yoksa nötr değer döndür
            
            # Son veriyi al
            last_data = df[self.config.FEATURE_COLUMNS].iloc[-1:]
            
            # Tahmin yap
            pred = self.model.predict_proba(last_data)[0]
            
            # Yükseliş olasılığını döndür
            return pred[1] * 100
            
        except Exception as e:
            print(f"ML tahmin hatası: {e}")
            return 50

    def save_scaler(self):
        """Scaler'ı kaydet"""
        joblib.dump(self.scaler, 'scaler.save')

    def load_scaler(self):
        """Scaler'ı yükle"""
        if os.path.exists('scaler.save'):
            self.scaler = joblib.load('scaler.save')
