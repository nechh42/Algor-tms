import numpy as np
import pandas as pd
import pandas_ta as ta
from scipy import stats
from sklearn.ensemble import IsolationForest
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta

class AdvancedAnalysis:
    def __init__(self):
        self.setup_logging()
        self.anomaly_detector = IsolationForest(contamination=0.1)
        self.lstm_model = self._build_lstm_model()
        
    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(message)s',
            handlers=[logging.StreamHandler()]
        )
        
    def _build_lstm_model(self):
        """LSTM modeli oluştur"""
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(60, 5)),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        return model
        
    def calculate_fibonacci_levels(self, high, low):
        """Fibonacci seviyeleri hesapla"""
        diff = high - low
        levels = {
            'Extension 1.618': high + diff * 0.618,
            'Extension 1.0': high,
            'Retracement 0.618': high - diff * 0.618,
            'Retracement 0.5': high - diff * 0.5,
            'Retracement 0.382': high - diff * 0.382,
            'Extension 0.0': low
        }
        return levels
        
    def calculate_pivot_points(self, high, low, close):
        """Pivot noktaları hesapla"""
        pivot = (high + low + close) / 3
        r1 = 2 * pivot - low
        r2 = pivot + (high - low)
        r3 = high + 2 * (pivot - low)
        s1 = 2 * pivot - high
        s2 = pivot - (high - low)
        s3 = low - 2 * (high - pivot)
        
        return {
            'R3': r3, 'R2': r2, 'R1': r1,
            'Pivot': pivot,
            'S1': s1, 'S2': s2, 'S3': s3
        }
        
    def analyze_volume_profile(self, price_data, volume_data, num_bins=10):
        """Hacim profili analizi"""
        # Fiyat aralıklarını belirle
        price_bins = pd.qcut(price_data, num_bins)
        
        # Her aralık için hacim toplamını hesapla
        volume_profile = pd.DataFrame({
            'price': price_data,
            'volume': volume_data
        }).groupby(price_bins)['volume'].sum()
        
        # POC (Point of Control) hesapla
        poc_price = volume_profile.idxmax().left
        
        # Value Area hesapla (%70 hacim)
        total_volume = volume_profile.sum()
        volume_threshold = total_volume * 0.7
        
        sorted_profile = volume_profile.sort_values(ascending=False)
        cumsum_volume = sorted_profile.cumsum()
        value_area = sorted_profile[cumsum_volume <= volume_threshold]
        
        value_area_high = value_area.index[-1].right
        value_area_low = value_area.index[0].left
        
        return {
            'poc': poc_price,
            'value_area_high': value_area_high,
            'value_area_low': value_area_low,
            'volume_profile': volume_profile
        }
        
    def detect_elliott_waves(self, prices, window=20):
        """Elliott dalgalarını tespit et"""
        # Trend yönünü belirle
        sma = ta.sma(prices, length=window)
        trend = 'up' if prices[-1] > sma[-1] else 'down'
        
        # Zigzag noktaları bul
        highs = []
        lows = []
        
        for i in range(1, len(prices)-1):
            if prices[i] > prices[i-1] and prices[i] > prices[i+1]:
                highs.append((i, prices[i]))
            elif prices[i] < prices[i-1] and prices[i] < prices[i+1]:
                lows.append((i, prices[i]))
                
        # Dalga yapısını analiz et
        waves = []
        if trend == 'up':
            for i in range(min(5, len(highs)-1)):
                waves.append({
                    'wave': i+1,
                    'type': 'motive' if i % 2 == 0 else 'corrective',
                    'start': highs[i][1],
                    'end': highs[i+1][1]
                })
        else:
            for i in range(min(5, len(lows)-1)):
                waves.append({
                    'wave': i+1,
                    'type': 'motive' if i % 2 == 0 else 'corrective',
                    'start': lows[i][1],
                    'end': lows[i+1][1]
                })
                
        return waves
        
    def detect_anomalies(self, data):
        """Anormallikleri tespit et"""
        # Veriyi yeniden şekillendir
        reshaped_data = data.reshape(-1, 1)
        
        # Anomali tespiti yap
        predictions = self.anomaly_detector.fit_predict(reshaped_data)
        
        # Anormallikleri işaretle
        anomalies = np.where(predictions == -1)[0]
        
        return anomalies
        
    def predict_price(self, data):
        """LSTM ile fiyat tahmini yap"""
        # Veriyi normalize et
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data)
        
        # Veriyi LSTM formatına dönüştür
        X = []
        y = []
        for i in range(60, len(scaled_data)):
            X.append(scaled_data[i-60:i])
            y.append(scaled_data[i, 0])
        X, y = np.array(X), np.array(y)
        
        # Tahmin yap
        prediction = self.lstm_model.predict(X[-1:])
        
        # Tahmini orijinal ölçeğe dönüştür
        predicted_price = scaler.inverse_transform(prediction)[0, 0]
        
        return predicted_price
        
    def calculate_ema(self, data, period):
        """EMA (Exponential Moving Average) hesapla"""
        try:
            multiplier = 2 / (period + 1)
            ema = [data.iloc[0]]  # İlk değer SMA olarak başlar
            
            for i in range(1, len(data)):
                price = data.iloc[i]
                ema.append((price * multiplier) + (ema[-1] * (1 - multiplier)))
            
            return pd.Series(ema, index=data.index)
        except Exception as e:
            logging.error(f"EMA hesaplama hatası: {str(e)}")
            return None

    def calculate_market_regime(self, high, low, close, window=20):
        """Piyasa rejimini hesapla"""
        try:
            if len(close) < window:
                return None
                
            # Trend yönü
            sma = close.rolling(window=window).mean()
            trend = 'yatay'
            
            if close.iloc[-1] > sma.iloc[-1] * 1.05:
                trend = 'güçlü_yükseliş'
            elif close.iloc[-1] > sma.iloc[-1]:
                trend = 'yükseliş'
            elif close.iloc[-1] < sma.iloc[-1] * 0.95:
                trend = 'güçlü_düşüş'
            elif close.iloc[-1] < sma.iloc[-1]:
                trend = 'düşüş'
            
            # Volatilite
            returns = close.pct_change()
            volatility = returns.std() * np.sqrt(252)  # Yıllık volatilite
            
            # Momentum
            roc = ((close.iloc[-1] - close.iloc[-window]) / close.iloc[-window]) * 100
            
            return {
                'trend': trend,
                'volatility': volatility,
                'momentum': roc
            }
            
        except Exception as e:
            logging.error(f"Piyasa rejimi hesaplama hatası: {str(e)}")
            return None
        
    def calculate_macd(self, prices, fast_period=12, slow_period=26, signal_period=9):
        """MACD (Moving Average Convergence Divergence) hesapla"""
        try:
            # Fast EMA hesapla
            fast_ema = self.calculate_ema(prices, fast_period)
            if fast_ema is None:
                return None
            
            # Slow EMA hesapla
            slow_ema = self.calculate_ema(prices, slow_period)
            if slow_ema is None:
                return None
            
            # MACD Line = Fast EMA - Slow EMA
            macd_line = fast_ema - slow_ema
            
            # Signal Line = 9-day EMA of MACD Line
            signal_line = self.calculate_ema(macd_line, signal_period)
            if signal_line is None:
                return None
            
            # MACD Histogram = MACD Line - Signal Line
            macd_hist = macd_line - signal_line
            
            return {
                'MACD_12_26_9': macd_line,
                'MACDs_12_26_9': signal_line,
                'MACDh_12_26_9': macd_hist
            }
        except Exception as e:
            logging.error(f"MACD hesaplama hatası: {str(e)}")
            return None

    def calculate_bollinger_bands(self, close, window=20, num_std=2):
        """Bollinger Bantlarını hesapla"""
        try:
            if len(close) < window:
                return None
            
            # Orta bant (20 günlük SMA)
            middle_band = close.rolling(window=window).mean()
            
            # Standart sapma
            std = close.rolling(window=window).std()
            
            # Üst ve alt bantlar
            upper_band = middle_band + (std * num_std)
            lower_band = middle_band - (std * num_std)
            
            return {
                'BBU_20_2.0': upper_band,
                'BBM_20_2.0': middle_band,
                'BBL_20_2.0': lower_band
            }
            
        except Exception as e:
            logging.error(f"Bollinger Bands hesaplama hatası: {str(e)}")
            return None

    def get_indicator_color(self, indicator_type, value, prev_value=None):
        """Gösterge rengini belirle"""
        try:
            colors = {
                'bullish': '\033[92m',  # Yeşil
                'bearish': '\033[91m',  # Kırmızı
                'neutral': '\033[93m',  # Sarı
                'reset': '\033[0m'      # Reset
            }
            
            if indicator_type == 'macd':
                if value > 0 and prev_value is not None and value > prev_value:
                    return colors['bullish']
                elif value < 0 and prev_value is not None and value < prev_value:
                    return colors['bearish']
                return colors['neutral']
                
            elif indicator_type == 'rsi':
                if value > 70:
                    return colors['bearish']
                elif value < 30:
                    return colors['bullish']
                return colors['neutral']
                
            elif indicator_type == 'bb':
                if value > 0.8:
                    return colors['bearish']
                elif value < 0.2:
                    return colors['bullish']
                return colors['neutral']
                
            return colors['reset']
            
        except Exception as e:
            logging.error(f"Renk belirleme hatası: {str(e)}")
            return '\033[0m'

    def log_indicators(self, df, symbol):
        """Göstergeleri logla"""
        try:
            current = df.iloc[-1]
            prev = df.iloc[-2] if len(df) > 1 else None
            
            # MACD
            macd_color = self.get_indicator_color('macd', current['macd'], prev['macd'] if prev is not None else None)
            macd_signal_color = self.get_indicator_color('macd', current['macd_signal'], prev['macd_signal'] if prev is not None else None)
            
            # RSI
            rsi_color = self.get_indicator_color('rsi', current['rsi'])
            
            # Bollinger Bands
            bb_position = (current['close'] - current['bb_lower']) / (current['bb_upper'] - current['bb_lower'])
            bb_color = self.get_indicator_color('bb', bb_position)
            
            # Log mesajlarını oluştur
            log_messages = [
                f"\n{symbol} Gösterge Değerleri:",
                f"MACD: {macd_color}{current['macd']:.8f}\033[0m",
                f"MACD Signal: {macd_signal_color}{current['macd_signal']:.8f}\033[0m",
                f"RSI: {rsi_color}{current['rsi']:.2f}\033[0m",
                f"Bollinger Bands:",
                f"  Üst: {bb_color}{current['bb_upper']:.2f}\033[0m",
                f"  Orta: {current['bb_middle']:.2f}",
                f"  Alt: {bb_color}{current['bb_lower']:.2f}\033[0m",
                f"Fiyat Pozisyonu: {bb_color}{bb_position:.2%}\033[0m"
            ]
            
            # Logları yazdır
            logging.info('\n'.join(log_messages))
            
        except Exception as e:
            logging.error(f"Gösterge loglama hatası: {str(e)}")

    def calculate_technical_indicators(self, df):
        """Teknik göstergeleri hesapla"""
        try:
            if df is None or len(df) < 2:
                return None
                
            # DataFrame'i kopyala
            df = df.copy()
            
            # MACD - kendi fonksiyonumuzu kullan
            macd_data = self.calculate_macd(df['close'])
            if macd_data is not None:
                df['macd'] = macd_data['MACD_12_26_9']
                df['macd_signal'] = macd_data['MACDs_12_26_9']
                df['macd_hist'] = macd_data['MACDh_12_26_9']
                logging.info(f"MACD hesaplandı: {df['macd'].iloc[-1]:.8f}")
            
            # RSI
            df['rsi'] = ta.rsi(df['close'], length=14)
            if 'rsi' in df:
                logging.info(f"RSI hesaplandı: {df['rsi'].iloc[-1]:.2f}")
            
            # Bollinger Bantları - kendi fonksiyonumuzu kullan
            bbands = self.calculate_bollinger_bands(df['close'])
            if bbands is not None:
                df['bb_upper'] = bbands['BBU_20_2.0']
                df['bb_middle'] = bbands['BBM_20_2.0']
                df['bb_lower'] = bbands['BBL_20_2.0']
                logging.info("Bollinger Bands hesaplandı")
            
            # Stochastic RSI
            stoch = ta.stochrsi(df['close'])
            if stoch is not None:
                df['stoch_k'] = stoch['STOCHRSIk_14_14_3_3']
                df['stoch_d'] = stoch['STOCHRSId_14_14_3_3']
                logging.info(f"Stochastic RSI hesaplandı: K={df['stoch_k'].iloc[-1]:.2f}, D={df['stoch_d'].iloc[-1]:.2f}")
            
            # NaN değerleri temizle
            df = df.ffill().bfill()
            
            return df
            
        except Exception as e:
            logging.error(f"Teknik gösterge hesaplama hatası: {str(e)}")
            return None

    def analyze_market_structure(self, df):
        """Piyasa yapısı analizi"""
        try:
            # Son kapanış
            current_price = df['close'].iloc[-1]
            
            # Hacim profili
            volume_profile = self.analyze_volume_profile(df['close'], df['volume'])
            
            # Fibonacci seviyeleri (son yüksek ve düşük noktalardan)
            high = df['high'].max()
            low = df['low'].min()
            fib_levels = self.calculate_fibonacci_levels(high, low)
            
            # Pivot noktaları
            pivots = self.calculate_pivot_points(
                df['high'].iloc[-1],
                df['low'].iloc[-1],
                df['close'].iloc[-1]
            )
            
            # Destek ve direnç seviyeleri
            support_resistance = {
                'volume_poc': volume_profile['poc'],
                'value_area_low': volume_profile['value_area_low'],
                'value_area_high': volume_profile['value_area_high'],
                'fib_levels': fib_levels,
                'pivot_points': pivots
            }
            
            # Fiyatın konumu
            price_position = {
                'above_poc': current_price > volume_profile['poc'],
                'in_value_area': volume_profile['value_area_low'] <= current_price <= volume_profile['value_area_high'],
                'near_fib': min(abs(current_price - v) for k, v in fib_levels.items() if isinstance(v, (int, float))),
                'near_pivot': min(abs(current_price - v) for k, v in pivots.items() if isinstance(v, (int, float)))
            }
            
            return {
                'support_resistance': support_resistance,
                'price_position': price_position,
                'current_price': current_price
            }
            
        except Exception as e:
            logging.error(f"Piyasa yapısı analizi hatası: {str(e)}")
            return None

    def fibonacci_levels(self, high: float, low: float) -> Dict[str, float]:
        """
        Fibonacci seviyelerini hesapla
        """
        try:
            diff = high - low
            levels = {
                'level_0': low,  # 0%
                'level_236': low + 0.236 * diff,  # 23.6%
                'level_382': low + 0.382 * diff,  # 38.2%
                'level_500': low + 0.500 * diff,  # 50%
                'level_618': low + 0.618 * diff,  # 61.8%
                'level_786': low + 0.786 * diff,  # 78.6%
                'level_1000': high  # 100%
            }
            return levels
            
        except Exception as e:
            logging.error(f"Fibonacci hesaplama hatası: {str(e)}")
            return {}
            
    def elliott_wave_analysis(self, prices: pd.Series) -> Dict:
        """
        Elliott dalga analizi
        """
        try:
            # Trend yönünü belirle
            trend = 'up' if prices.iloc[-1] > prices.iloc[0] else 'down'
            
            # Dalgaları belirle
            waves = self._identify_waves(prices)
            
            # Dalga oranlarını hesapla
            ratios = self._calculate_wave_ratios(waves)
            
            return {
                'trend': trend,
                'waves': waves,
                'ratios': ratios
            }
            
        except Exception as e:
            logging.error(f"Elliott dalga analizi hatası: {str(e)}")
            return {}
            
    def volume_profile(self, prices: pd.Series, volumes: pd.Series,
                      num_levels: int = 10) -> Dict:
        """
        Hacim profili analizi
        """
        try:
            # Fiyat aralıklarını belirle
            price_range = np.linspace(prices.min(), prices.max(), num_levels)
            
            # Her aralıktaki hacmi hesapla
            volume_by_price = {}
            for i in range(len(price_range)-1):
                mask = (prices >= price_range[i]) & (prices < price_range[i+1])
                volume_by_price[f"level_{i}"] = {
                    'price_low': price_range[i],
                    'price_high': price_range[i+1],
                    'volume': volumes[mask].sum()
                }
                
            # POC (Point of Control) hesapla
            poc_level = max(volume_by_price.items(),
                          key=lambda x: x[1]['volume'])[0]
            
            return {
                'volume_profile': volume_by_price,
                'poc': volume_by_price[poc_level]
            }
            
        except Exception as e:
            logging.error(f"Hacim profili analizi hatası: {str(e)}")
            return {}
            
    def pivot_points(self, high: float, low: float, close: float) -> Dict[str, float]:
        """
        Pivot noktalarını hesapla
        """
        try:
            pivot = (high + low + close) / 3
            
            r1 = 2 * pivot - low  # Resistance 1
            r2 = pivot + (high - low)  # Resistance 2
            r3 = high + 2 * (pivot - low)  # Resistance 3
            
            s1 = 2 * pivot - high  # Support 1
            s2 = pivot - (high - low)  # Support 2
            s3 = low - 2 * (high - pivot)  # Support 3
            
            return {
                'pivot': pivot,
                'r1': r1, 'r2': r2, 'r3': r3,
                's1': s1, 's2': s2, 's3': s3
            }
            
        except Exception as e:
            logging.error(f"Pivot noktaları hesaplama hatası: {str(e)}")
            return {}
            
    def monte_carlo_simulation(self, prices: pd.Series, n_simulations: int = 1000,
                             n_days: int = 30) -> Dict:
        """
        Monte Carlo simülasyonu
        """
        try:
            # Günlük getiriler
            returns = np.log(1 + prices.pct_change())
            
            # Simülasyon parametreleri
            mu = returns.mean()
            sigma = returns.std()
            
            # Simülasyonları gerçekleştir
            simulations = np.zeros((n_simulations, n_days))
            last_price = prices.iloc[-1]
            
            for i in range(n_simulations):
                prices_sim = [last_price]
                for d in range(n_days):
                    price = prices_sim[-1] * np.exp(np.random.normal(mu, sigma))
                    prices_sim.append(price)
                simulations[i] = prices_sim[1:]
                
            # Sonuçları analiz et
            final_prices = simulations[:, -1]
            conf_intervals = {
                '95': np.percentile(final_prices, [2.5, 97.5]),
                '99': np.percentile(final_prices, [0.5, 99.5])
            }
            
            return {
                'expected_price': np.mean(final_prices),
                'confidence_intervals': conf_intervals,
                'max_price': np.max(final_prices),
                'min_price': np.min(final_prices),
                'simulations': simulations.tolist()
            }
            
        except Exception as e:
            logging.error(f"Monte Carlo simülasyonu hatası: {str(e)}")
            return {}
            
    def stress_test(self, portfolio: Dict, scenarios: List[Dict]) -> Dict:
        """
        Stres testi
        """
        try:
            results = {}
            
            for scenario in scenarios:
                # Senaryo parametreleri
                price_change = scenario.get('price_change', 0)
                volatility_change = scenario.get('volatility_change', 0)
                volume_change = scenario.get('volume_change', 0)
                
                # Portföy değerini hesapla
                portfolio_value = sum(
                    pos['size'] * pos['entry_price'] * (1 + price_change)
                    for pos in portfolio.values()
                )
                
                # Risk metriklerini hesapla
                var = self._calculate_var(portfolio_value, volatility_change)
                expected_shortfall = self._calculate_expected_shortfall(
                    portfolio_value, volatility_change
                )
                
                results[scenario['name']] = {
                    'portfolio_value': portfolio_value,
                    'value_at_risk': var,
                    'expected_shortfall': expected_shortfall,
                    'price_impact': price_change * 100,
                    'volatility_impact': volatility_change * 100
                }
                
            return results
            
        except Exception as e:
            logging.error(f"Stres testi hatası: {str(e)}")
            return {}
            
    def setup_grid_strategy(self, current_price: float, grid_levels: int = 10,
                          grid_spacing: float = 0.02) -> Dict:
        """
        Grid strateji kurulumu
        """
        try:
            grid = {}
            
            # Alt ve üst limitleri belirle
            upper_limit = current_price * (1 + grid_spacing * grid_levels/2)
            lower_limit = current_price * (1 - grid_spacing * grid_levels/2)
            
            # Grid seviyelerini oluştur
            price_levels = np.linspace(lower_limit, upper_limit, grid_levels)
            
            for i, price in enumerate(price_levels):
                grid[f"level_{i}"] = {
                    'price': price,
                    'type': 'buy' if price < current_price else 'sell',
                    'size': self._calculate_grid_size(price, current_price)
                }
                
            return {
                'grid_levels': grid,
                'current_price': current_price,
                'upper_limit': upper_limit,
                'lower_limit': lower_limit
            }
            
        except Exception as e:
            logging.error(f"Grid strateji kurulum hatası: {str(e)}")
            return {}
            
    def optimize_grid_parameters(self, price_history: pd.Series,
                               initial_params: Dict) -> Dict:
        """
        Grid parametrelerini optimize et
        """
        try:
            results = {}
            
            # Test edilecek parametre kombinasyonları
            grid_levels_range = range(5, 21, 5)
            grid_spacing_range = np.arange(0.01, 0.06, 0.01)
            
            for levels in grid_levels_range:
                for spacing in grid_spacing_range:
                    # Grid stratejisini test et
                    performance = self._backtest_grid_strategy(
                        price_history, levels, spacing
                    )
                    
                    results[f"levels_{levels}_spacing_{spacing:.2f}"] = {
                        'profit': performance['profit'],
                        'num_trades': performance['num_trades'],
                        'max_drawdown': performance['max_drawdown'],
                        'sharpe_ratio': performance['sharpe_ratio']
                    }
                    
            # En iyi parametreleri bul
            best_config = max(results.items(), key=lambda x: x[1]['sharpe_ratio'])
            
            return {
                'all_results': results,
                'best_config': best_config[0],
                'best_performance': best_config[1]
            }
            
        except Exception as e:
            logging.error(f"Grid parametre optimizasyonu hatası: {str(e)}")
            return {}
            
    def _identify_waves(self, prices: pd.Series) -> List[Dict]:
        """Elliott dalgalarını belirle"""
        try:
            waves = []
            # Trend değişim noktalarını bul
            peaks = self._find_peaks(prices)
            troughs = self._find_troughs(prices)
            
            # Dalgaları oluştur
            points = sorted(peaks + troughs, key=lambda x: x[0])
            
            for i in range(len(points)-1):
                wave = {
                    'start_idx': points[i][0],
                    'end_idx': points[i+1][0],
                    'start_price': points[i][1],
                    'end_price': points[i+1][1],
                    'direction': 'up' if points[i+1][1] > points[i][1] else 'down'
                }
                waves.append(wave)
                
            return waves
            
        except Exception as e:
            logging.error(f"Dalga belirleme hatası: {str(e)}")
            return []
            
    def _calculate_wave_ratios(self, waves: List[Dict]) -> Dict:
        """Dalga oranlarını hesapla"""
        try:
            ratios = {}
            
            for i in range(len(waves)-1):
                current_wave = abs(waves[i]['end_price'] - waves[i]['start_price'])
                next_wave = abs(waves[i+1]['end_price'] - waves[i+1]['start_price'])
                
                ratios[f"wave_{i}_to_{i+1}"] = next_wave / current_wave
                
            return ratios
            
        except Exception as e:
            logging.error(f"Dalga oranları hesaplama hatası: {str(e)}")
            return {}
            
    def _find_peaks(self, prices: pd.Series) -> List[Tuple[int, float]]:
        """Tepe noktalarını bul"""
        peaks = []
        for i in range(1, len(prices)-1):
            if prices[i] > prices[i-1] and prices[i] > prices[i+1]:
                peaks.append((i, prices[i]))
        return peaks
        
    def _find_troughs(self, prices: pd.Series) -> List[Tuple[int, float]]:
        """Dip noktalarını bul"""
        troughs = []
        for i in range(1, len(prices)-1):
            if prices[i] < prices[i-1] and prices[i] < prices[i+1]:
                troughs.append((i, prices[i]))
        return troughs
        
    def _calculate_grid_size(self, grid_price: float,
                           current_price: float) -> float:
        """Grid seviyesi için işlem büyüklüğünü hesapla"""
        try:
            # Fiyat farkına göre büyüklüğü ayarla
            price_diff = abs(grid_price - current_price) / current_price
            
            # Temel büyüklük
            base_size = 100  # USDT
            
            # Fiyat farkı arttıkça büyüklüğü azalt
            size = base_size * (1 - price_diff)
            
            return max(size, base_size * 0.1)  # Minimum %10
            
        except Exception as e:
            logging.error(f"Grid büyüklüğü hesaplama hatası: {str(e)}")
            return 0
            
    def _backtest_grid_strategy(self, price_history: pd.Series,
                              grid_levels: int, grid_spacing: float) -> Dict:
        """Grid stratejisini test et"""
        try:
            trades = []
            position = 0
            balance = 10000  # Başlangıç bakiyesi
            
            # Her fiyat için grid seviyelerini kontrol et
            for i in range(1, len(price_history)):
                current_price = price_history[i]
                prev_price = price_history[i-1]
                
                # Grid kurulumu
                grid = self.setup_grid_strategy(
                    prev_price, grid_levels, grid_spacing
                )
                
                # İşlem sinyallerini kontrol et
                for level in grid['grid_levels'].values():
                    if level['type'] == 'buy' and prev_price > level['price'] >= current_price:
                        # Alış sinyali
                        position += level['size']
                        balance -= level['size'] * current_price
                        trades.append({
                            'type': 'buy',
                            'price': current_price,
                            'size': level['size']
                        })
                    elif level['type'] == 'sell' and prev_price < level['price'] <= current_price:
                        # Satış sinyali
                        position -= level['size']
                        balance += level['size'] * current_price
                        trades.append({
                            'type': 'sell',
                            'price': current_price,
                            'size': level['size']
                        })
                        
            # Performans metrikleri
            final_balance = balance + position * price_history[-1]
            profit = final_balance - 10000
            
            returns = pd.Series([t['price'] for t in trades]).pct_change().dropna()
            sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std() if len(returns) > 1 else 0
            
            max_balance = 10000
            max_drawdown = 0
            for trade in trades:
                current_balance = balance + position * trade['price']
                max_balance = max(max_balance, current_balance)
                drawdown = (max_balance - current_balance) / max_balance
                max_drawdown = max(max_drawdown, drawdown)
                
            return {
                'profit': profit,
                'num_trades': len(trades),
                'max_drawdown': max_drawdown,
                'sharpe_ratio': sharpe_ratio
            }
            
        except Exception as e:
            logging.error(f"Grid strateji testi hatası: {str(e)}")
            return {
                'profit': 0,
                'num_trades': 0,
                'max_drawdown': 0,
                'sharpe_ratio': 0
            }
            
    def _calculate_var(self, portfolio_value: float,
                      volatility_change: float) -> float:
        """Value at Risk hesapla"""
        try:
            # %99 güven aralığı için VaR
            z_score = stats.norm.ppf(0.99)
            volatility = 0.02 * (1 + volatility_change)  # Baz volatilite %2
            var = portfolio_value * volatility * z_score
            return var
            
        except Exception as e:
            logging.error(f"VaR hesaplama hatası: {str(e)}")
            return 0
            
    def _calculate_expected_shortfall(self, portfolio_value: float,
                                    volatility_change: float) -> float:
        """Expected Shortfall (CVaR) hesapla"""
        try:
            # %99 güven aralığı için ES
            z_score = stats.norm.ppf(0.99)
            volatility = 0.02 * (1 + volatility_change)
            pdf = stats.norm.pdf(z_score)
            es = portfolio_value * volatility * (pdf / (1 - 0.99))
            return es
            
        except Exception as e:
            logging.error(f"Expected Shortfall hesaplama hatası: {str(e)}")
            return 0
