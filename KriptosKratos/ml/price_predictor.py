"""
Fiyat tahmin modülü - LSTM tabanlı
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import os

# TensorFlow uyarılarını kapat
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

class PricePredictor:
    def __init__(self):
        self.model = None
        self.scaler = MinMaxScaler()
        self.lookback_period = 60  # 60 periyot geçmiş veri
        self.prediction_period = 12  # 12 periyot tahmin
        self.retrain_interval = 1000  # 1000 veri noktasında bir yeniden eğit
        self.data_points = 0
        self.trained_pairs = set()
        
    def preprocess_data(self, data):
        """Veriyi modele uygun formata dönüştür"""
        scaled_data = self.scaler.fit_transform(data[['close', 'volume', 'rsi', 'macd', 'bb_width']].values)
        X, y = [], []
        
        for i in range(len(scaled_data) - self.lookback_period - self.prediction_period):
            X.append(scaled_data[i:(i + self.lookback_period)])
            y.append(scaled_data[i + self.lookback_period:i + self.lookback_period + self.prediction_period, 0])
            
        return np.array(X), np.array(y)
        
    def build_model(self, input_shape):
        """LSTM modeli oluştur"""
        model = Sequential([
            LSTM(100, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(self.prediction_period)
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001),
                     loss='mse',
                     metrics=['mae'])
        return model
        
    def train(self, data, symbol):
        """Modeli eğit"""
        try:
            print(f"\n{symbol} için AI modeli eğitiliyor...")
            X, y = self.preprocess_data(data)
            
            if len(X) < 100:  # Minimum veri kontrolü
                print(f"{symbol} için yeterli veri yok")
                return False
                
            if self.model is None:
                self.model = self.build_model(input_shape=(X.shape[1], X.shape[2]))
                
            # Modeli eğit
            history = self.model.fit(X, y,
                                   epochs=50,
                                   batch_size=32,
                                   validation_split=0.1,
                                   verbose=0)
                                   
            self.trained_pairs.add(symbol)
            print(f"{symbol} için AI modeli eğitildi - Loss: {history.history['loss'][-1]:.4f}")
            return True
            
        except Exception as e:
            print(f"{symbol} eğitimi sırasında hata: {str(e)}")
            return False
            
    def predict(self, data, symbol):
        """Fiyat tahmini yap"""
        try:
            if symbol not in self.trained_pairs:
                if not self.train(data, symbol):
                    return None
                    
            # Veriyi hazırla
            scaled_data = self.scaler.transform(data[['close', 'volume', 'rsi', 'macd', 'bb_width']].values[-self.lookback_period:])
            X = np.array([scaled_data])
            
            # Tahmin yap
            scaled_pred = self.model.predict(X, verbose=0)
            predictions = self.scaler.inverse_transform(np.column_stack((scaled_pred[0], np.zeros((self.prediction_period, 4)))))
            
            return predictions[:, 0]  # Sadece fiyat tahminlerini döndür
            
        except Exception as e:
            print(f"{symbol} tahmini sırasında hata: {str(e)}")
            return None
            
    def get_prediction_metrics(self, data, symbol):
        """Tahmin metriklerini hesapla"""
        predictions = self.predict(data, symbol)
        if predictions is None:
            return None
            
        current_price = data['close'].iloc[-1]
        predicted_prices = predictions
        
        return {
            'current_price': current_price,
            'predicted_prices': predicted_prices,
            'predicted_direction': 'UP' if predicted_prices[-1] > current_price else 'DOWN',
            'predicted_change': ((predicted_prices[-1] - current_price) / current_price) * 100,
            'confidence_score': self.calculate_confidence_score(data, predictions)
        }
        
    def calculate_confidence_score(self, data, predictions):
        """Tahmin güven skorunu hesapla"""
        # Son tahminlerin doğruluk oranına göre güven skoru
        try:
            actual = data['close'].iloc[-self.prediction_period:].values
            previous_pred = predictions[:len(actual)]
            
            # RMSE bazlı güven skoru
            rmse = np.sqrt(np.mean((actual - previous_pred) ** 2))
            max_rmse = np.std(data['close']) * 2
            confidence = max(0, min(100, (1 - rmse/max_rmse) * 100))
            
            return confidence
            
        except:
            return 50  # Varsayılan güven skoru
