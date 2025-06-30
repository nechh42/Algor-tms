import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

class PricePredictor:
    def __init__(self):
        self.model = None
        self.scaler = MinMaxScaler()
        
    def create_model(self, sequence_length):
        """LSTM modelini oluştur"""
        model = Sequential([
            LSTM(units=50, return_sequences=True, input_shape=(sequence_length, 1)),
            Dropout(0.2),
            LSTM(units=50, return_sequences=False),
            Dropout(0.2),
            Dense(units=25),
            Dense(units=1)
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        self.model = model
        return model
        
    def prepare_data(self, df, sequence_length):
        """Veriyi model için hazırla"""
        # Fiyat verisini normalize et
        scaled_data = self.scaler.fit_transform(df['close'].values.reshape(-1, 1))
        
        X, y = [], []
        for i in range(len(scaled_data) - sequence_length):
            X.append(scaled_data[i:(i + sequence_length)])
            y.append(scaled_data[i + sequence_length])
            
        return np.array(X), np.array(y)
        
    def train(self, df, sequence_length=60, epochs=50, batch_size=32):
        """Modeli eğit"""
        X, y = self.prepare_data(df, sequence_length)
        
        if self.model is None:
            self.create_model(sequence_length)
            
        self.model.fit(X, y, epochs=epochs, batch_size=batch_size, validation_split=0.1)
        
    def predict(self, df, sequence_length=60):
        """Gelecek fiyat tahmini yap"""
        if self.model is None:
            raise ValueError("Model henüz eğitilmemiş!")
            
        # Son sequence_length kadar veriyi al
        last_sequence = df['close'].values[-sequence_length:]
        scaled_sequence = self.scaler.transform(last_sequence.reshape(-1, 1))
        
        # Tahmin yap
        prediction = self.model.predict(scaled_sequence.reshape(1, sequence_length, 1))
        
        # Tahmini orijinal ölçeğe geri dönüştür
        predicted_price = self.scaler.inverse_transform(prediction)[0][0]
        
        return predicted_price
