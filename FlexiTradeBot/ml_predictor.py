import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import config
import ta
import time
import os
import pickle

class MLPredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.load_model()
    
    def load_model(self):
        """Eğitilmiş modeli yükle"""
        try:
            self.model = joblib.load('model/trading_model.pkl')
            self.scaler = joblib.load('model/scaler.pkl')
        except:
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
    
    def prepare_features(self, df):
        """Öznitelikleri hazırla"""
        features = pd.DataFrame()
        
        # Teknik göstergelerden öznitelikler
        features['rsi'] = df['RSI']
        features['macd'] = df['MACD']
        features['bb_position'] = (df['close'] - df['BB_LOWER']) / (df['BB_UPPER'] - df['BB_LOWER'])
        features['ema_cross'] = df['EMA_CROSS']
        features['vwap_position'] = (df['close'] - df['VWAP']) / df['VWAP']
        features['volume_ratio'] = df['VOL_RATIO']
        
        # Fiyat değişimi
        features['price_change'] = df['close'].pct_change()
        features['volatility'] = df['ATR'] / df['close']
        
        # Trend özellikleri
        features['trend_strength'] = (df['EMA9'] - df['EMA21']) / df['EMA21']
        
        return features.fillna(0)
    
    def predict(self, df):
        """Tahmin yap"""
        try:
            # Öznitelikleri hazırla
            features = self.prepare_features(df)
            
            # Son veriyi al
            current_features = features.iloc[-1:].values
            
            # Normalize et
            current_features = self.scaler.fit_transform(current_features)
            
            # Tahmin yap
            prediction = self.model.predict_proba(current_features)[0]
            
            # Sonuçları döndür
            return {
                'buy_probability': prediction[1],
                'sell_probability': prediction[0]
            }
            
        except Exception as e:
            print(f"ML tahmin hatası: {str(e)}")
            return None
    
    def train(self, df, labels):
        """Modeli eğit"""
        try:
            # Öznitelikleri hazırla
            features = self.prepare_features(df)
            
            # Normalize et
            X = self.scaler.fit_transform(features)
            
            # Modeli eğit
            self.model.fit(X, labels)
            
            # Modeli kaydet
            joblib.dump(self.model, 'model/trading_model.pkl')
            joblib.dump(self.scaler, 'model/scaler.pkl')
            
            print("Model başarıyla eğitildi ve kaydedildi!")
            return True
            
        except Exception as e:
            print(f"Model eğitim hatası: {str(e)}")
            return False
    
    def train_model(self, symbol, interval='5m', lookback_days=30):
        """Modeli eğit"""
        try:
            # Geçmiş verileri al
            end_time = int(time.time() * 1000)
            start_time = end_time - (lookback_days * 24 * 60 * 60 * 1000)
            
            klines = self.client.futures_historical_klines(
                symbol=symbol,
                interval=interval,
                start_str=str(start_time),
                end_str=str(end_time)
            )
            
            # DataFrame oluştur
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])
            
            # Veri tiplerini dönüştür
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Teknik göstergeleri hesapla
            df['RSI'] = ta.momentum.rsi(df['close'], window=14)
            
            # MACD
            macd = ta.trend.macd(df['close'])
            df['MACD'] = macd.macd()
            df['MACD_Signal'] = macd.macd_signal()
            df['MACD_Hist'] = macd.macd_diff()
            
            # Bollinger Bands
            bollinger = ta.volatility.BollingerBands(df['close'])
            df['BB_Upper'] = bollinger.bollinger_hband()
            df['BB_Lower'] = bollinger.bollinger_lband()
            df['BB_Middle'] = bollinger.bollinger_mavg()
            
            # Diğer göstergeler
            df['EMA_9'] = ta.trend.ema_indicator(df['close'], window=9)
            df['SMA_20'] = ta.trend.sma_indicator(df['close'], window=20)
            df['ATR'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'])
            
            # Hedef değişkeni oluştur (gelecek 3 mum için yön)
            df['Target'] = np.where(
                df['close'].shift(-3) > df['close'] * 1.005,  # %0.5 yukarı
                1,  # LONG
                np.where(
                    df['close'].shift(-3) < df['close'] * 0.995,  # %0.5 aşağı
                    -1,  # SHORT
                    0  # NOTR
                )
            )
            
            # NaN değerleri temizle
            df.dropna(inplace=True)
            
            # Özellik ve hedef değişkenleri ayır
            features = [
                'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist',
                'BB_Upper', 'BB_Lower', 'BB_Middle',
                'EMA_9', 'SMA_20', 'ATR'
            ]
            
            X = df[features]
            y = df['Target']
            
            # Veriyi ölçeklendir
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            
            # Modeli eğit
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
            
            self.model.fit(X_scaled, y)
            
            # Modeli kaydet
            if not os.path.exists('models'):
                os.makedirs('models')
                
            model_path = f'models/{symbol}_model.pkl'
            scaler_path = f'models/{symbol}_scaler.pkl'
            
            with open(model_path, 'wb') as f:
                pickle.dump(self.model, f)
            
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
            
            print(f"[{symbol}] Model başarıyla eğitildi ve kaydedildi!")
            return True
            
        except Exception as e:
            print(f"[{symbol}] Model eğitimi hatası: {str(e)}")
            return False
