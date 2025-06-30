import pandas as pd
import pandas_ta as ta
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import logging

class AdvancedStrategy:
    def __init__(self, config):
        """Strateji başlat"""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # AI modelini yükle
        self.model = self._load_ai_model()
        self.scaler = StandardScaler()

    def _load_ai_model(self):
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(60, 7)),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def calculate_fibonacci_levels(self, df):
        """Fibonacci seviyeleri hesapla"""
        try:
            # Son trend için yüksek ve düşük noktaları bul
            high = df['high'].rolling(window=20).max().iloc[-1]
            low = df['low'].rolling(window=20).min().iloc[-1]
            
            # Fibonacci seviyeleri
            diff = high - low
            levels = {
                'fib_236': high - (diff * 0.236),
                'fib_382': high - (diff * 0.382),
                'fib_500': high - (diff * 0.500),
                'fib_618': high - (diff * 0.618),
                'fib_786': high - (diff * 0.786)
            }
            return levels
        except Exception as e:
            print(f"Fibonacci hesaplama hatası: {e}")
            return None

    def calculate_all_indicators(self, df):
        """Tüm teknik göstergeleri hesapla"""
        try:
            # RSI
            df['rsi'] = ta.rsi(df['close'], length=self.config.RSI_PERIOD)
            
            # MACD
            macd = ta.macd(df['close'], fast=12, slow=26, signal=9)
            df['macd'] = macd['MACD_12_26_9']
            df['macd_signal'] = macd['SIGNAL_12_26_9']
            
            # EMA'lar
            df['ema_short'] = ta.ema(df['close'], length=self.config.EMA_SHORT)
            df['ema_long'] = ta.ema(df['close'], length=self.config.EMA_LONG)
            
            # Bollinger Bantları
            bb = ta.bbands(df['close'], length=20, std=2)
            df['bb_upper'] = bb['BBU_20_2.0']
            df['bb_lower'] = bb['BBL_20_2.0']
            df['bb_middle'] = bb['BBM_20_2.0']
            
            # Stochastic RSI
            stoch = ta.stochrsi(df['close'], length=14, rsi_length=14, smooth=3)
            df['stoch_k'] = stoch['StochRSI_14_14_3_3']
            df['stoch_d'] = stoch['StochRSId_14_14_3_3']
            
            # ADX
            adx = ta.adx(df['high'], df['low'], df['close'], length=14)
            df['adx'] = adx['ADX_14']
            df['adx_pos'] = adx['ADX_POS_14']
            df['adx_neg'] = adx['ADX_NEG_14']
            
            # ATR
            df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
            
            # CCI
            df['cci'] = ta.cci(df['high'], df['low'], df['close'], length=14)
            
            # OBV
            df['obv'] = ta.obv(df['close'], df['volume'])
            
            # Williams %R
            df['williams_r'] = ta.wr(df['high'], df['low'], df['close'], length=14)
            
            # Ichimoku Cloud
            high_9 = df['high'].rolling(window=9).max()
            low_9 = df['low'].rolling(window=9).min()
            df['tenkan_sen'] = (high_9 + low_9) / 2
            
            high_26 = df['high'].rolling(window=26).max()
            low_26 = df['low'].rolling(window=26).min()
            df['kijun_sen'] = (high_26 + low_26) / 2
            
            df['senkou_span_a'] = ((df['tenkan_sen'] + df['kijun_sen']) / 2).shift(26)
            
            high_52 = df['high'].rolling(window=52).max()
            low_52 = df['low'].rolling(window=52).min()
            df['senkou_span_b'] = ((high_52 + low_52) / 2).shift(26)
            
            df['chikou_span'] = df['close'].shift(-26)

            # Pivot Noktaları
            df['pivot'] = (df['high'].rolling(2).max() + df['low'].rolling(2).min() + df['close']) / 3
            df['r1'] = 2 * df['pivot'] - df['low'].rolling(2).min()
            df['s1'] = 2 * df['pivot'] - df['high'].rolling(2).max()
            
            # Momentum Göstergeleri
            # ROC (Rate of Change)
            df['roc'] = ((df['close'] - df['close'].shift(10)) / df['close'].shift(10)) * 100
            
            # MFI (Money Flow Index)
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            money_flow = typical_price * df['volume']
            
            positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0).rolling(14).sum()
            negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0).rolling(14).sum()
            
            mfi_ratio = positive_flow / negative_flow
            df['mfi'] = 100 - (100 / (1 + mfi_ratio))
            
            # Supertrend
            atr = ta.atr(df['high'], df['low'], df['close'], length=10)
            
            upperband = ((df['high'] + df['low']) / 2) + (2 * atr)
            lowerband = ((df['high'] + df['low']) / 2) - (2 * atr)
            
            df['supertrend'] = np.where(df['close'] > upperband, 1, 
                                      np.where(df['close'] < lowerband, -1, 0))
            
            # VWAP Hesaplama
            df['vwap'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()
            
            # Hacim Profili
            price_bins = pd.qcut(df['close'], 10, labels=False)
            volume_profile = df.groupby(price_bins)['volume'].sum()
            df['volume_profile'] = price_bins.map(volume_profile)
            
            # Elliott Wave Analizi
            df['wave_trend'] = self._calculate_elliott_waves(df)
            
            # Pivot Noktaları (Gelişmiş)
            df['pp'] = (df['high'].shift(1) + df['low'].shift(1) + df['close'].shift(1)) / 3
            df['r1'] = 2 * df['pp'] - df['low'].shift(1)
            df['s1'] = 2 * df['pp'] - df['high'].shift(1)
            df['r2'] = df['pp'] + (df['high'].shift(1) - df['low'].shift(1))
            df['s2'] = df['pp'] - (df['high'].shift(1) - df['low'].shift(1))
            
            # Hacim Ağırlıklı Bollinger Bantları
            vwma = df['vwap'].rolling(window=20).mean()
            vwsd = df['vwap'].rolling(window=20).std()
            df['vw_bb_upper'] = vwma + (2 * vwsd)
            df['vw_bb_lower'] = vwma - (2 * vwsd)
            
            # Market Profili
            df['market_profile'] = self._calculate_market_profile(df)
            
            # Momentum Göstergeleri
            # Chaikin Money Flow
            mf_multiplier = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
            df['cmf'] = (mf_multiplier * df['volume']).rolling(20).sum() / df['volume'].rolling(20).sum()
            
            # Awesome Oscillator
            median_price = (df['high'] + df['low']) / 2
            ao_fast = median_price.rolling(window=5).mean()
            ao_slow = median_price.rolling(window=34).mean()
            df['ao'] = ao_fast - ao_slow
            
            # DeMark Göstergeleri
            df['demark_9'] = self._calculate_demark_9(df)
            df['demark_13'] = self._calculate_demark_13(df)
            
            # Volatilite İndeksi
            df['volatility_index'] = df['close'].rolling(window=20).std() / df['close'].rolling(window=20).mean() * 100
            
            return df
            
        except Exception as e:
            print(f"Gösterge hesaplama hatası: {e}")
            return None

    def _calculate_elliott_waves(self, df):
        """Elliott Wave analizi"""
        try:
            # Basit Elliott Wave hesaplaması
            close = df['close'].values
            waves = np.zeros(len(close))
            
            for i in range(5, len(close)):
                # Dalga 1-5 tespiti
                price_changes = np.diff(close[i-5:i])
                wave_pattern = np.sign(price_changes)
                
                # İdeal Elliott Wave paterni: [1,-1,1,-1,1]
                if np.array_equal(wave_pattern, [1,-1,1,-1,1]):
                    waves[i] = 1  # Yükseliş dalgası
                elif np.array_equal(wave_pattern, [-1,1,-1,1,-1]):
                    waves[i] = -1  # Düşüş dalgası
                    
            return waves
            
        except Exception as e:
            print(f"Elliott Wave hesaplama hatası: {e}")
            return np.zeros(len(df))

    def _calculate_market_profile(self, df):
        """Market Profili hesapla"""
        try:
            # Fiyat aralıklarını belirle
            price_range = df['high'].max() - df['low'].min()
            tick_size = price_range / 30  # 30 fiyat seviyesi
            
            # Her zaman dilimi için fiyat dağılımını hesapla
            profile = np.zeros(len(df))
            
            for i in range(len(df)):
                if i < 30:
                    continue
                    
                # Son 30 mum için fiyat dağılımı
                recent_prices = df['close'].iloc[i-30:i]
                hist, _ = np.histogram(recent_prices, bins=30)
                
                # Value Area hesapla (%70 hacim)
                total_volume = hist.sum()
                value_area_volume = total_volume * 0.7
                
                cumsum = 0
                for j, volume in enumerate(sorted(hist, reverse=True)):
                    cumsum += volume
                    if cumsum >= value_area_volume:
                        profile[i] = j
                        break
                        
            return profile
            
        except Exception as e:
            print(f"Market Profile hesaplama hatası: {e}")
            return np.zeros(len(df))

    def _calculate_demark_9(self, df):
        """DeMark 9 göstergesi"""
        try:
            setup = np.zeros(len(df))
            count = 0
            
            for i in range(4, len(df)):
                if df['close'].iloc[i] < df['close'].iloc[i-4]:
                    count += 1
                    if count == 9:
                        setup[i] = 1
                        count = 0
                else:
                    count = 0
                    
            return setup
            
        except Exception as e:
            print(f"DeMark 9 hesaplama hatası: {e}")
            return np.zeros(len(df))

    def _calculate_demark_13(self, df):
        """DeMark 13 göstergesi"""
        try:
            setup = np.zeros(len(df))
            count = 0
            
            for i in range(13, len(df)):
                if df['close'].iloc[i] > df['close'].iloc[i-13]:
                    count += 1
                    if count == 13:
                        setup[i] = 1
                        count = 0
                else:
                    count = 0
                    
            return setup
            
        except Exception as e:
            print(f"DeMark 13 hesaplama hatası: {e}")
            return np.zeros(len(df))

    def get_ai_prediction(self, df):
        """AI tahmin modeli"""
        try:
            # Feature hazırlama
            features = self._prepare_ai_features(df)
            
            # Model tahminini al
            prediction = self.model.predict(features)
            
            # Tahmin güvenilirliği
            confidence = self._calculate_prediction_confidence(prediction, features)
            
            # Normalize edilmiş tahmin (-1 ile 1 arası)
            normalized_prediction = (prediction[0] - 0.5) * 2
            
            return normalized_prediction * confidence
            
        except Exception as e:
            print(f"AI tahmin hatası: {e}")
            return None
            
    def _prepare_ai_features(self, df):
        """AI için özellikleri hazırla"""
        try:
            features = pd.DataFrame()
            
            # Teknik göstergeler
            features['rsi'] = ta.rsi(df['close'], length=14)
            features['macd'] = ta.macd(df['close'], fast=12, slow=26, signal=9)['MACD_12_26_9']
            features['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
            
            # Fiyat değişimleri
            features['price_change'] = df['close'].pct_change()
            features['volume_change'] = df['volume'].pct_change()
            
            # Momentum göstergeleri
            features['mom'] = ta.mom(df['close'], length=10)
            features['cci'] = ta.cci(df['high'], df['low'], df['close'], length=14)
            
            # Trend göstergeleri
            features['adx'] = ta.adx(df['high'], df['low'], df['close'], length=14)['ADX_14']
            features['willr'] = ta.wr(df['high'], df['low'], df['close'], length=14)
            
            # Hareketli ortalamalar
            features['sma_20'] = ta.sma(df['close'], length=20)
            features['sma_50'] = ta.sma(df['close'], length=50)
            features['ema_20'] = ta.ema(df['close'], length=20)
            features['ema_50'] = ta.ema(df['close'], length=50)
            
            # Volatilite göstergeleri
            features['bb_width'] = self._calculate_bb_width(df)
            features['volatility'] = df['close'].rolling(window=20).std() / df['close'].rolling(window=20).mean()
            
            # Hacim bazlı göstergeler
            features['vwap'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()
            features['obv'] = ta.obv(df['close'], df['volume'])
            
            # NaN değerleri temizle
            features = features.fillna(0)
            
            # Son satırı al
            features = features.iloc[-1:].values
            
            return features
            
        except Exception as e:
            print(f"Feature hazırlama hatası: {e}")
            return None
            
    def _calculate_prediction_confidence(self, prediction, features):
        """Tahmin güvenilirliğini hesapla"""
        try:
            confidence = 0.5  # Baz güvenilirlik
            
            # Trend gücüne göre güvenilirlik
            adx = features[0][7]  # ADX değeri
            if adx > 25:  # Güçlü trend
                confidence += 0.2
            
            # Hacim değişimine göre güvenilirlik
            volume_change = features[0][4]  # Hacim değişimi
            if abs(volume_change) > 0.1:  # Önemli hacim değişimi
                confidence += 0.1
            
            # Volatiliteye göre güvenilirlik
            volatility = features[0][15]  # Volatilite
            if 0.01 < volatility < 0.05:  # İdeal volatilite aralığı
                confidence += 0.1
            
            # RSI ekstrem değerlere göre güvenilirlik
            rsi = features[0][0]  # RSI değeri
            if rsi < 30 or rsi > 70:  # Aşırı alım/satım
                confidence += 0.1
            
            return min(confidence, 1.0)  # Maksimum 1.0
            
        except Exception as e:
            print(f"Güvenilirlik hesaplama hatası: {e}")
            return 0.5
            
    def _calculate_bb_width(self, df):
        """Bollinger Bant genişliğini hesapla"""
        try:
            bb = ta.bbands(df['close'], length=20, std=2)
            upper = bb['BBU_20_2.0']
            lower = bb['BBL_20_2.0']
            middle = bb['BBM_20_2.0']
            
            return (upper - lower) / middle
            
        except Exception as e:
            print(f"BB genişlik hesaplama hatası: {e}")
            return pd.Series(0, index=df.index)

    def calculate_signals(self, df):
        """Sinyalleri hesapla"""
        try:
            if df is None or len(df) < 100:
                return None, None
                
            metrics = {}
            
            try:
                # RSI
                rsi = ta.rsi(df['close'], length=14)
                metrics['rsi'] = rsi.iloc[-1]
                
                # MACD
                macd = ta.macd(df['close'], fast=12, slow=26, signal=9)
                metrics['macd'] = macd['MACD_12_26_9'].iloc[-1]
                metrics['macd_signal'] = macd['SIGNAL_12_26_9'].iloc[-1]
                
                # Bollinger Bantları
                bb = ta.bbands(df['close'], length=20, std=2)
                metrics['bb_upper'] = bb['BBU_20_2.0'].iloc[-1]
                metrics['bb_lower'] = bb['BBL_20_2.0'].iloc[-1]
                
                # ATR
                atr = ta.atr(df['high'], df['low'], df['close'], length=14)
                metrics['atr'] = atr.iloc[-1]
                
                # Hacim Analizi
                metrics['volume_sma'] = df['volume'].rolling(window=20).mean().iloc[-1]
                metrics['volume_change'] = ((df['volume'].iloc[-1] / metrics['volume_sma']) - 1) * 100
                
                # Trend Gücü
                adx = ta.adx(df['high'], df['low'], df['close'], length=14)
                metrics['trend_strength'] = adx['ADX_14'].iloc[-1] / 100
                
            except Exception as e:
                print(f"Gösterge hesaplama hatası: {str(e)}")
                return None, None
                
            # Sinyal Hesaplama
            signal = 0
            
            # RSI Sinyali
            if metrics['rsi'] < 30:
                signal += 0.2
            elif metrics['rsi'] > 70:
                signal -= 0.2
                
            # MACD Sinyali
            if metrics['macd'] > metrics['macd_signal']:
                signal += 0.3
            else:
                signal -= 0.3
                
            # Bollinger Band Sinyali
            current_price = df['close'].iloc[-1]
            if current_price < metrics['bb_lower']:
                signal += 0.2
            elif current_price > metrics['bb_upper']:
                signal -= 0.2
                
            # Hacim Sinyali
            if metrics['volume_change'] > 50:  # Hacim artışı
                if signal > 0:
                    signal *= 1.2
                elif signal < 0:
                    signal *= 0.8
                    
            # Trend Gücü Etkisi
            signal *= (1 + metrics['trend_strength'])
            
            return signal, metrics
            
        except Exception as e:
            print(f"Sinyal hesaplama hatası: {str(e)}")
            return None, None

    def calculate_position_size(self, current_balance, entry_price):
        """Pozisyon büyüklüğünü hesapla"""
        try:
            # Kelly Criterion ile pozisyon büyüklüğünü hesapla
            kelly_size = current_balance * self.config.POSITION_SIZE * self.config.KELLY_FRACTION
            
            # Minimum pozisyon büyüklüğünü kontrol et
            position_size = max(kelly_size, 10)  # Minimum 10 USDT
            
            # Maksimum açık pozisyon sayısını dikkate al
            max_position_size = current_balance / self.config.MAX_OPEN_POSITIONS
            position_size = min(position_size, max_position_size)
            
            # USDT cinsinden pozisyon büyüklüğünü coin miktarına çevir
            quantity = position_size / entry_price
            
            return quantity
            
        except Exception as e:
            print(f"Pozisyon büyüklüğü hesaplama hatası: {e}")
            return None

    def calculate_stop_loss(self, entry_price, signal_type, atr=None):
        """Dinamik stop loss hesapla"""
        if atr:
            # ATR bazlı stop loss
            multiplier = 2  # ATR çarpanı
            if signal_type == 'LONG':
                return entry_price - (atr * multiplier)
            else:
                return entry_price + (atr * multiplier)
        else:
            # Yüzde bazlı stop loss
            if signal_type == 'LONG':
                return entry_price * (1 - self.config.MAX_POSITION_LOSS / 100)
            else:
                return entry_price * (1 + self.config.MAX_POSITION_LOSS / 100)

    def calculate_take_profit(self, entry_price, signal_type, atr=None):
        """Dinamik take profit hesapla"""
        if atr:
            # ATR bazlı take profit
            multiplier = 4  # ATR çarpanı (Risk:Reward = 1:2)
            if signal_type == 'LONG':
                return entry_price + (atr * multiplier)
            else:
                return entry_price - (atr * multiplier)
        else:
            # Yüzde bazlı take profit
            risk = entry_price * (self.config.MAX_POSITION_LOSS / 100)
            if signal_type == 'LONG':
                return entry_price + (risk * 2)
            else:
                return entry_price - (risk * 2)

    def analyze(self, df, symbol):
        """
        Teknik analiz yap ve trading sinyalleri üret
        """
        try:
            if df is None or len(df) < 50:  # Minimum veri kontrolü
                return None
                
            # RSI hesapla
            df['rsi'] = ta.rsi(df['close'], length=14)
            
            # MACD hesapla
            macd = ta.macd(df['close'], fast=12, slow=26, signal=9)
            df = pd.concat([df, macd], axis=1)
            
            # Bollinger Bands hesapla
            bb = ta.bbands(df['close'], length=20, std=2)
            df = pd.concat([df, bb], axis=1)
            
            # Hacim analizi
            df['volume_ma'] = ta.sma(df['volume'], length=20)
            df['volume_ratio'] = df['volume'] / df['volume_ma']
            
            # Volatilite hesapla
            df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
            df['volatility'] = (df['atr'] / df['close']) * 100
            
            # Trend gücü (ADX)
            adx = ta.adx(df['high'], df['low'], df['close'], length=14)
            df = pd.concat([df, adx], axis=1)
            
            # Sinyal gücünü hesapla
            signal_strength = self._calculate_signal_strength(df)
            
            # Giriş kriterlerini kontrol et
            entry_signal = self._check_entry_criteria(df, signal_strength)
            
            # Son fiyat değişimi
            price_change = ((df['close'].iloc[-1] - df['close'].iloc[-2]) / df['close'].iloc[-2]) * 100
            
            return {
                'symbol': symbol,
                'signal': entry_signal,
                'signal_strength': signal_strength,
                'current_price': df['close'].iloc[-1],
                'price_change': price_change,
                'rsi': df['rsi'].iloc[-1],
                'macd_hist': df['MACDh_12_26_9'].iloc[-1],
                'bb_width': (df['BBU_20_2.0'].iloc[-1] - df['BBL_20_2.0'].iloc[-1]) / df['BBM_20_2.0'].iloc[-1],
                'volume_ratio': df['volume_ratio'].iloc[-1],
                'volatility': df['volatility'].iloc[-1],
                'trend_strength': df['ADX_14'].iloc[-1]
            }
            
        except Exception as e:
            print(f"{symbol} analizi sırasında hata: {str(e)}")
            return None
            
    def _calculate_signal_strength(self, df):
        """
        Sinyal gücünü hesapla (1-5 arası)
        """
        strength = 0
        
        # RSI sinyali
        rsi = df['rsi'].iloc[-1]
        if rsi < 30:
            strength += 1
        elif rsi > 70:
            strength += 1
            
        # MACD sinyali
        if df['MACDh_12_26_9'].iloc[-1] > 0 and df['MACDh_12_26_9'].iloc[-2] <= 0:
            strength += 1
            
        # Bollinger Band sinyali
        bb_width = (df['BBU_20_2.0'].iloc[-1] - df['BBL_20_2.0'].iloc[-1]) / df['BBM_20_2.0'].iloc[-1]
        if bb_width > 0.05:
            strength += 1
            
        # Hacim sinyali
        if df['volume_ratio'].iloc[-1] > 2:
            strength += 1
            
        # Trend gücü sinyali
        if df['ADX_14'].iloc[-1] > 25:
            strength += 1
            
        return strength
        
    def _check_entry_criteria(self, df, signal_strength):
        """
        Giriş kriterlerini kontrol et
        """
        if signal_strength < 3:
            return False
            
        # Fiyat hareketi kontrolü
        price_change = abs((df['close'].iloc[-1] - df['close'].iloc[-2]) / df['close'].iloc[-2] * 100)
        if price_change < 1:
            return False
            
        # Hacim artışı kontrolü
        volume_change = ((df['volume'].iloc[-1] - df['volume'].iloc[-2]) / df['volume'].iloc[-2]) * 100
        if volume_change < 50:
            return False
            
        # Volatilite kontrolü
        volatility = df['volatility'].iloc[-1]
        if not (1 <= volatility <= 5):
            return False
            
        return True
