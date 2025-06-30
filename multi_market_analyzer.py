from binance.client import Client
import pandas as pd
import numpy as np
import ta
import config
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib
import os
from datetime import datetime

class MultiMarketAnalyzer:
    def __init__(self):
        self.client = Client(config.API_KEY, config.API_SECRET)
        self.setup_logging()
        self.scaler = StandardScaler()
        self.model = self.load_or_create_model()
        self.opportunity_threshold = 0.75  # Fırsat skoru eşiği
        
    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(message)s',
            handlers=[logging.StreamHandler()]
        )
        
    def load_or_create_model(self):
        """AI modelini yükle veya oluştur"""
        model_path = 'market_model.joblib'
        if os.path.exists(model_path):
            return joblib.load(model_path)
        return RandomForestClassifier(n_estimators=100, random_state=42)
        
    def get_all_futures_symbols(self):
        """Tüm USDT futures sembollerini al"""
        try:
            exchange_info = self.client.futures_exchange_info()
            symbols = [s['symbol'] for s in exchange_info['symbols'] 
                      if s['symbol'].endswith('USDT') and s['status'] == 'TRADING']
            return symbols
        except Exception as e:
            logging.error(f"Sembol listesi alınamadı: {str(e)}")
            return []
            
    def get_historical_data(self, symbol, interval='15m', limit=100):
        """Geçmiş fiyat verilerini al"""
        try:
            klines = self.client.futures_klines(
                symbol=symbol,
                interval=interval,
                limit=limit
            )
            
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])
            
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
            
        except Exception as e:
            logging.error(f"{symbol} için veri alınamadı: {str(e)}")
            return None
            
    def calculate_indicators(self, df):
        """Teknik göstergeleri hesapla"""
        try:
            # Temel göstergeler
            df['RSI'] = ta.momentum.RSIIndicator(df['close']).rsi()
            
            macd = ta.trend.MACD(df['close'])
            df['MACD'] = macd.macd()
            df['MACD_Signal'] = macd.macd_signal()
            
            bollinger = ta.volatility.BollingerBands(df['close'])
            df['BB_upper'] = bollinger.bollinger_hband()
            df['BB_lower'] = bollinger.bollinger_lband()
            df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['close']
            
            # Volatilite göstergeleri
            df['ATR'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
            
            # Momentum göstergeleri
            df['MFI'] = ta.volume.MFIIndicator(df['high'], df['low'], df['close'], df['volume']).money_flow_index()
            
            # Trend göstergeleri
            df['ADX'] = ta.trend.ADXIndicator(df['high'], df['low'], df['close']).adx()
            
            return df
            
        except Exception as e:
            logging.error(f"Gösterge hesaplamada hata: {str(e)}")
            return None
            
    def calculate_opportunity_score(self, df):
        """Kural tabanlı fırsat skoru hesapla"""
        try:
            # Son veriyi al
            current = df.iloc[-1]
            
            # Temel skor hesaplama
            score = 0.0
            
            # RSI bazlı skor (Aşırı satım: 30'un altı, Aşırı alım: 70'in üstü)
            if current['RSI'] < 30:
                score += 0.3  # Aşırı satım: alım fırsatı
            elif current['RSI'] > 70:
                score += 0.1  # Aşırı alım: riskli
            else:
                score += 0.2  # Normal bölge
                
            # MACD bazlı skor
            if current['MACD'] > current['MACD_Signal']:
                score += 0.2  # Yükseliş sinyali
            
            # Bollinger Bant bazlı skor
            bb_position = (current['close'] - current['BB_lower']) / (current['BB_upper'] - current['BB_lower'])
            if bb_position < 0.2:  # Fiyat alt banda yakın
                score += 0.2
            elif bb_position > 0.8:  # Fiyat üst banda yakın
                score += 0.1
                
            # ADX bazlı skor (Trend gücü)
            if current['ADX'] > 25:
                score += 0.2  # Güçlü trend
                
            # Volatilite skoru
            volatility = current['BB_width']
            if volatility > df['BB_width'].mean():
                score += 0.1  # Yüksek volatilite
                
            # Son 3 mum analizi
            last_3_candles = df.iloc[-3:]
            price_change = (current['close'] - last_3_candles['close'].iloc[0]) / last_3_candles['close'].iloc[0]
            if abs(price_change) > 0.02:  # %2'den fazla hareket
                score += 0.1
                
            return min(score, 1.0)  # Maksimum 1.0
            
        except Exception as e:
            logging.error(f"Fırsat skoru hesaplanamadı: {str(e)}")
            return 0
            
    def analyze_all_markets(self):
        """Tüm marketleri analiz et"""
        opportunities = []
        symbols = self.get_all_futures_symbols()
        
        logging.info(f"\n=== {len(symbols)} market analiz ediliyor ===\n")
        
        for symbol in symbols:
            try:
                df = self.get_historical_data(symbol)
                if df is None:
                    continue
                    
                df = self.calculate_indicators(df)
                if df is None:
                    continue
                    
                current = df.iloc[-1]
                opportunity_score = self.calculate_opportunity_score(df)
                
                # Fırsat bilgilerini kaydet
                if opportunity_score > self.opportunity_threshold:
                    opportunity = {
                        'symbol': symbol,
                        'price': current['close'],
                        'score': opportunity_score,
                        'rsi': current['RSI'],
                        'macd': current['MACD'],
                        'bb_width': current['BB_width'],
                        'mfi': current['MFI'],
                        'adx': current['ADX']
                    }
                    opportunities.append(opportunity)
                    
            except Exception as e:
                logging.error(f"{symbol} analiz edilirken hata: {str(e)}")
                continue
                
        # Fırsatları skora göre sırala
        opportunities.sort(key=lambda x: x['score'], reverse=True)
        self.print_opportunities(opportunities)
        return opportunities
        
    def print_opportunities(self, opportunities):
        """Tespit edilen fırsatları yazdır"""
        logging.info("\n=== Tespit Edilen Fırsatlar ===")
        
        if not opportunities:
            logging.info("Şu anda uygun fırsat bulunamadı.")
            return
            
        for opp in opportunities:
            logging.info(f"\nSembol: {opp['symbol']}")
            logging.info(f"Fiyat: {opp['price']:.8f} USDT")
            logging.info(f"Fırsat Skoru: {opp['score']:.2%}")
            logging.info(f"RSI: {opp['rsi']:.2f}")
            logging.info(f"MACD: {opp['macd']:.8f}")
            logging.info(f"Bollinger Genişliği: {opp['bb_width']:.4f}")
            logging.info(f"MFI: {opp['mfi']:.2f}")
            logging.info(f"ADX: {opp['adx']:.2f}")
            
    def update_ai_model(self, training_data):
        """AI modelini güncelle"""
        try:
            X = training_data[['RSI', 'MACD', 'MACD_Signal', 'BB_width', 'ATR', 'MFI', 'ADX']]
            y = training_data['target']  # Başarılı işlemler 1, başarısızlar 0
            
            X_scaled = self.scaler.fit_transform(X)
            self.model.fit(X_scaled, y)
            
            # Modeli kaydet
            joblib.dump(self.model, 'market_model.joblib')
            logging.info("AI model güncellendi ve kaydedildi.")
            
        except Exception as e:
            logging.error(f"Model güncellenirken hata: {str(e)}")

if __name__ == "__main__":
    analyzer = MultiMarketAnalyzer()
    analyzer.analyze_all_markets()
