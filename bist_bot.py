import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import time
from ta.trend import SMAIndicator, MACD
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands
import os

class BistBot:
    def __init__(self):
        self.symbols = []
        self.data = {}
        self.results_dir = 'sonuclar'
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)
        
    def get_bist_symbols(self):
        """BIST sembollerini al"""
        # Örnek hisseler (daha sonra tüm BIST hisseleri eklenebilir)
        self.symbols = [
            'THYAO.IS', 'GARAN.IS', 'ASELS.IS', 'SISE.IS', 'KCHOL.IS',
            'AKBNK.IS', 'EREGL.IS', 'BIMAS.IS', 'TUPRS.IS', 'YKBNK.IS'
        ]
        
    def fetch_data(self, symbol, period='1y'):
        """Yahoo Finance'dan veri çek"""
        try:
            stock = yf.Ticker(symbol)
            df = stock.history(period=period)
            return df
        except Exception as e:
            print(f"Veri çekerken hata: {symbol} - {e}")
            return None
            
    def calculate_signals(self, df):
        """Teknik analiz sinyalleri hesapla"""
        if df is None or len(df) < 50:
            return None
            
        # Temel göstergeler
        sma20 = SMAIndicator(close=df['Close'], window=20)
        sma50 = SMAIndicator(close=df['Close'], window=50)
        sma200 = SMAIndicator(close=df['Close'], window=200)
        df['SMA20'] = sma20.sma_indicator()
        df['SMA50'] = sma50.sma_indicator()
        df['SMA200'] = sma200.sma_indicator()
        
        # Bollinger Bands
        bb = BollingerBands(close=df['Close'])
        df['BB_upper'] = bb.bollinger_hband()
        df['BB_middle'] = bb.bollinger_mavg()
        df['BB_lower'] = bb.bollinger_lband()
        
        # RSI
        rsi = RSIIndicator(close=df['Close'])
        df['RSI'] = rsi.rsi()
        
        # MACD
        macd = MACD(close=df['Close'])
        df['MACD'] = macd.macd()
        df['Signal'] = macd.macd_signal()
        df['MACD_Hist'] = macd.macd_diff()
        
        # Momentum (fiyat değişimi olarak hesaplayalım)
        df['Momentum'] = df['Close'].diff(periods=10)
        
        # Stochastic Oscillator
        stoch = StochasticOscillator(high=df['High'], low=df['Low'], close=df['Close'])
        df['Stoch_K'] = stoch.stoch()
        df['Stoch_D'] = stoch.stoch_signal()
        
        # Alım sinyalleri (geliştirilmiş)
        df['Buy_Signal'] = (
            (df['SMA20'] > df['SMA50']) &  # Altın kesişim
            (df['Close'] > df['SMA200']) &  # Uzun vadeli yükseliş trendi
            (df['RSI'] < 70) &  # Aşırı alım bölgesinde değil
            (df['MACD'] > df['Signal']) &  # MACD sinyali
            (df['Close'] > df['BB_middle']) &  # Orta bandın üstünde
            (df['Stoch_K'] > df['Stoch_D']) &  # Stochastic sinyal
            (df['Momentum'] > 0)  # Pozitif momentum
        )
        
        # Satış sinyalleri (geliştirilmiş)
        df['Sell_Signal'] = (
            (df['SMA20'] < df['SMA50']) &  # Ölüm kesişimi
            (df['Close'] < df['SMA200']) &  # Uzun vadeli düşüş trendi
            (df['RSI'] > 30) &  # Aşırı satım bölgesinde değil
            (df['MACD'] < df['Signal']) &  # MACD sinyali
            (df['Close'] < df['BB_middle']) &  # Orta bandın altında
            (df['Stoch_K'] < df['Stoch_D']) &  # Stochastic sinyal
            (df['Momentum'] < 0)  # Negatif momentum
        )
        
        return df
        
    def save_analysis(self, symbol, df):
        """Analiz sonuçlarını kaydet"""
        if df is not None and len(df) > 0:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{self.results_dir}/{symbol}_{timestamp}_analiz.csv"
            df.tail(30).to_csv(filename)  # Son 30 günün verisini kaydet
            
    def analyze_all_stocks(self):
        """Tüm hisseleri analiz et"""
        self.get_bist_symbols()
        
        analysis_results = []
        
        for symbol in self.symbols:
            print(f"\nAnaliz ediliyor: {symbol}")
            df = self.fetch_data(symbol)
            if df is not None:
                df = self.calculate_signals(df)
                if df is not None:
                    last_row = df.iloc[-1]
                    
                    # Sonuçları kaydet
                    self.save_analysis(symbol, df)
                    
                    # Analiz özeti
                    result = {
                        'Sembol': symbol,
                        'Son_Fiyat': last_row['Close'],
                        'RSI': last_row['RSI'],
                        'MACD': last_row['MACD'],
                        'Stoch_K': last_row['Stoch_K'],
                        'Momentum': last_row['Momentum'],
                        'SMA20': last_row['SMA20'],
                        'SMA50': last_row['SMA50']
                    }
                    
                    print(f"Son Kapanış: {result['Son_Fiyat']:.2f}")
                    print(f"RSI: {result['RSI']:.2f}")
                    print(f"MACD: {result['MACD']:.2f}")
                    print(f"Stochastic K: {result['Stoch_K']:.2f}")
                    print(f"Momentum: {result['Momentum']:.2f}")
                    
                    if last_row['Buy_Signal']:
                        print(">>> GÜÇLÜ ALIM SİNYALİ!")
                        result['Sinyal'] = 'ALIM'
                    elif last_row['Sell_Signal']:
                        print(">>> GÜÇLÜ SATIŞ SİNYALİ!")
                        result['Sinyal'] = 'SATIŞ'
                    else:
                        print(">>> BEKLE VE İZLE")
                        result['Sinyal'] = 'BEKLE'
                    
                    analysis_results.append(result)
                        
            time.sleep(1)  # API limitlerini aşmamak için bekle
        
        # Tüm sonuçları tek bir dosyaya kaydet
        if analysis_results:
            results_df = pd.DataFrame(analysis_results)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            results_df.to_csv(f"{self.results_dir}/toplu_analiz_{timestamp}.csv", index=False)
            print(f"\nTüm analizler {self.results_dir} klasörüne kaydedildi.")

if __name__ == "__main__":
    bot = BistBot()
    bot.analyze_all_stocks()
