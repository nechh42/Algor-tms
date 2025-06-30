import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from sklearn.preprocessing import MinMaxScaler
import logging
import pandas as pd

class DeepLearningModel:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.y_scaler = None
        self.setup_logging()
        self.setup_model()
        
    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def setup_model(self):
        """LSTM modelini oluştur"""
        try:
            # L2 regularization
            l2_reg = l2(1e-6)
            
            self.model = Sequential([
                # First LSTM layer with more units
                Bidirectional(LSTM(128, return_sequences=True, kernel_regularizer=l2_reg), input_shape=(60, 8)),
                BatchNormalization(),
                Dropout(0.2),
                
                # Second LSTM layer with moderate units
                Bidirectional(LSTM(64, return_sequences=True, kernel_regularizer=l2_reg)),
                Dropout(0.2),
                
                # Final LSTM layer
                Bidirectional(LSTM(32, kernel_regularizer=l2_reg)),
                Dropout(0.2),
                
                # Dense layers with residual connections
                Dense(32, activation='relu', kernel_regularizer=l2_reg),
                Dense(16, activation='relu', kernel_regularizer=l2_reg),
                Dense(1)
            ])
            
            # Use Adam optimizer with reduced learning rate
            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(
                    learning_rate=0.0005,
                    clipnorm=1.0  # Add gradient clipping
                ),
                loss='huber'  # Huber loss is more robust to outliers
            )
            
            self.logger.info("Model başarıyla oluşturuldu")
            
        except Exception as e:
            self.logger.error(f"Model oluşturma hatası: {str(e)}")
            
    def prepare_data(self, data, is_training=True):
        """Veriyi model için hazırla"""
        try:
            # Özellik seçimi - test_model.py ile aynı sırada
            features = ['close_pct', 'volume', 'rsi', 'macd', 'bb_width', 'price_position', 'atr', 'obv']
            X = data[features].values
            
            # Initialize scaler if not exists
            if self.scaler is None:
                self.scaler = MinMaxScaler()
            
            # Veriyi normalize et
            if is_training:
                # Training modunda fit_transform kullan
                X = self.scaler.fit_transform(X)
            else:
                # Prediction modunda sadece transform kullan
                X = self.scaler.transform(X)
                
            # Sekansları oluştur
            X_seq = []
            y_seq = []
            
            if len(X) < 60:
                # Tahmin için tek bir sekans oluştur
                X_seq = [X]
                y_seq = [0]  # Dummy hedef değer
            else:
                # Eğitim için sekanslar oluştur
                for i in range(60, len(X)):
                    X_seq.append(X[i-60:i])
                    if is_training:
                        y_seq.append(data['close'].iloc[i])  # Use actual close price as target
                    else:
                        y_seq.append(0)  # Dummy value for prediction
                
            return np.array(X_seq), np.array(y_seq)
            
        except Exception as e:
            self.logger.error(f"Veri hazırlama hatası: {str(e)}")
            return None, None
            
    def train_model(self, X_train, y_train):
        """Modeli eğit"""
        try:
            # Scale the target values
            y_scaler = MinMaxScaler()
            y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).reshape(-1)
            
            # Early stopping callback
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True
            )
            
            # Learning rate reduction callback
            reduce_lr = ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-5
            )
            
            # Model checkpoint callback
            checkpoint = ModelCheckpoint(
                'models/best_model.h5',
                monitor='val_loss',
                save_best_only=True,
                mode='min',
                verbose=0
            )
            
            # Train the model
            history = self.model.fit(
                X_train,
                y_train_scaled,
                epochs=150,
                batch_size=32,
                validation_split=0.2,
                callbacks=[early_stopping, reduce_lr, checkpoint],
                verbose=1
            )
            
            # Store the y_scaler for later use
            self.y_scaler = y_scaler
            
            return history
            
        except Exception as e:
            logging.error(f"Model eğitimi hatası: {str(e)}")
            return None
            
    def predict_next_price(self, X):
        """Bir sonraki fiyatı tahmin et"""
        try:
            # Ensure X is a pandas DataFrame
            if not isinstance(X, pd.DataFrame):
                X = pd.DataFrame(X)
            
            # Prepare features in the correct order
            features = ['close_pct', 'volume', 'rsi', 'macd', 'bb_width', 'price_position', 'atr', 'obv']
            
            # Check if we have enough data
            if len(X) < 60:
                raise ValueError("Not enough data points. Need at least 60 rows.")
            
            # Get the last 60 rows of features
            X = X.tail(60)
            
            # Check if all required features are present
            missing_features = [f for f in features if f not in X.columns]
            if missing_features:
                raise ValueError(f"Missing required features: {missing_features}")
            
            # Select and order features
            X = X[features]
            
            # Convert to numpy array and reshape for LSTM input
            X_array = X.values
            X_array = self.scaler.transform(X_array)
            X_seq = np.array([X_array])  # Shape: (1, 60, 8)
            
            # Make prediction
            prediction_scaled = self.model.predict(X_seq, verbose=0)
            
            # Inverse transform the prediction
            prediction = self.y_scaler.inverse_transform(prediction_scaled.reshape(-1, 1)).reshape(-1)[0]
            
            return prediction
            
        except Exception as e:
            logging.error(f"Tahmin sırasında hata: {str(e)}")
            return None
            
    def save_model(self, path='model.keras'):
        """Modeli kaydet"""
        try:
            self.model.save(path)
            self.logger.info(f"Model başarıyla kaydedildi: {path}")
            
        except Exception as e:
            self.logger.error(f"Model kaydetme hatası: {str(e)}")
            
    def load_model(self, path='model.keras'):
        """Modeli yükle"""
        try:
            self.model = load_model(path)
            self.logger.info(f"Model başarıyla yüklendi: {path}")
            
        except Exception as e:
            self.logger.error(f"Model yükleme hatası: {str(e)}")
