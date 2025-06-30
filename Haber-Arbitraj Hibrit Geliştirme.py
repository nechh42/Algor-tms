import ccxt
import time
import logging
import pandas as pd
import ta
import requests
from transformers import pipeline

class NewsArbHybrid:
    def __init__(self, binance_api_key, binance_api_secret, kucoin_api_key, kucoin_api_secret, 
                 newsapi_key, capital=100, fee_rate=0.001, max_risk=0.01):
        self.binance = ccxt.binance({
            'apiKey': binance_api_key,
            'secret': binance_api_secret,
            'enableRateLimit': True
        })
        self.kucoin = ccxt.kucoin({
            'apiKey': kucoin_api_key,
            'secret': kucoin_api_secret,
            'enableRateLimit': True
        })
        self.newsapi_key = newsapi_key  # NewsAPI anahtarı
        self.capital = capital  # $100
        self.fee_rate = fee_rate  # %0.1 işlem ücreti
        self.max_risk = max_risk  # Maksimum risk: %1
        self.volatility_threshold = 0.5  # ATR eşiği
        self.sentiment_threshold = 0.5  # Sentiment tetikleyici
        self.sentiment_analyzer = pipeline("sentiment-analysis")
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

    def fetch_news(self):
        """NewsAPI’den BTC ile ilgili haberleri çeker."""
        try:
            url = f"https://newsapi.org/v2/everything?q=bitcoin&apiKey={self.newsapi_key}"
            response = requests.get(url)
            articles = response.json().get('articles', [])
            return [article['title'] + " " + article['description'] for article in articles if article['description']]
        except Exception as e:
            logging.error(f"Haber alınırken hata: {e}")
            return []

    def get_sentiment(self):
        """Haberlerden sentiment skoru hesaplar."""
        news = self.fetch_news()
        if not news:
            return 0
        scores = [self.sentiment_analyzer(text)[0]['score'] 
                  if self.sentiment_analyzer(text)[0]['label'] == 'POSITIVE' 
                  else -self.sentiment_analyzer(text)[0]['score'] for text in news]
        return sum(scores) / len(scores) if scores else 0

    def fetch_prices(self, symbol='BTCUSDT'):
        """Binance ve KuCoin’dan fiyatları çeker."""
        try:
            binance_ticker = self.binance.fetch_ticker(symbol)
            kucoin_ticker = self.kucoin.fetch_ticker(symbol)
            return {
                'binance': {'bid': binance_ticker['bid'], 'ask': binance_ticker['ask']},
                'kucoin': {'bid': kucoin_ticker['bid'], 'ask': kucoin_ticker['ask']}
            }
        except Exception as e:
            logging.error(f"Fiyat alınırken hata: {e}")
            return None

    def get_volatility(self, symbol='BTC/USDT', timeframe='1m', limit=100):
        """Son 100 dakikalık ATR’yi hesaplar."""
        try:
            klines = self.binance.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
            return df['atr'].iloc[-1]
        except Exception as e:
            logging.error(f"Volatilite hesaplanırken hata: {e}")
            return None

    def find_arbitrage_opportunity(self, symbol='BTC/USDT'):
        """Arbitraj fırsatlarını tespit eder."""
        sentiment = self.get_sentiment()
        if abs(sentiment) < self.sentiment_threshold:
            logging.info("Düşük sentiment skoru, arbitraj taraması durduruldu.")
            return None

        prices = self.fetch_prices(symbol)
        if not prices:
            return None

        atr = self.get_volatility(symbol)
        if not atr:
            return None

        # Risk yönetimi: Volatiliteye göre pozisyon büyüklüğü
        risk_factor = 0.02 if atr > self.volatility_threshold else 0.005  # Agresif/konservatif
        amount = (self.capital * min(self.max_risk, risk_factor)) / min(prices['binance']['ask'], prices['kucoin']['ask'])

        max_bid = max(prices['binance']['bid'], prices['kucoin']['bid'])
        min_ask =:min(prices['binance']['ask'], prices['kucoin']['ask'])
        buy_exchange = 'binance' if prices['binance']['ask'] == min_ask else 'kucoin'
        sell_exchange = 'binance' if prices['binance']['bid'] == max_bid else 'kucoin'

        profit = max_bid - min_ask
        fees = (min_ask * self.fee_rate) + (max_bid * self.fee_rate)
        net_profit = profit - fees

        if net_profit > 0:
            return {
                'buy_exchange': buy_exchange,
                'sell_exchange': sell_exchange,
                'buy_price': min_ask,
                'sell_price': max_bid,
                'amount': amount,
                'net_profit': net_profit * amount,
                'atr': atr,
                'sentiment': sentiment
            }
        return None

    def execute_trade(self, opportunity, symbol='BTC/USDT'):
        """Gerçek limit emri yerleştirir."""
        if not opportunity:
            return False

        try:
            buy_client = self.binance if opportunity['buy_exchange'] == 'binance' else self.kucoin
            sell_client = self.binance if opportunity['sell_exchange'] == 'binance' else self.kucoin

            # Alım emri
            buy_order = buy_client.create_limit_buy_order(symbol, opportunity['amount'], opportunity['buy_price'])
            logging.info(f"Alım emri: {opportunity['buy_exchange']}’te {opportunity['amount']:.6f} BTC @ {opportunity['buy_price']}")

            # Satım emri
            sell_order = sell_client.create_limit_sell_order(symbol, opportunity['amount'], opportunity['sell_price'])
            logging.info(f"Satım emri: {opportunity['sell_exchange']}’te {opportunity['amount']:.6f} BTC @ {opportunity['sell_price']}")
            logging.info(f"Tahmini kâr: ${opportunity['net_profit']:.2f}, Sentiment: {opportunity['sentiment']:.2f}")
            return True
        except Exception as e:
            logging.error(f"Emir yürütülürken hata: {e}")
            return False

    def run(self, symbol='BTC/USDT', interval=5):
        """Botu çalıştırır."""
        while True:
            try:
                opportunity = self.find_arbitrage_opportunity(symbol)
                if opportunity:
                    self.execute_trade(opportunity, symbol)
                else:
                    logging.info("Arbitraj fırsatı bulunamadı.")
                time.sleep(interval if opportunity else interval * 2)  # Volatilite düşükse daha az tarama
            except Exception as e:
                logging.error(f"Hata oluştu: {str(e)}")
                time.sleep(interval)

if __name__ == "__main__":
    try:
        # Binance ve KuCoin API anahtarlarınızı, NewsAPI anahtarınızı buraya girin
        bot = NewsArbHybrid(
            binance_api_key='your_api_key',
            binance_api_secret='your_api_secret',
            kucoin_api_key='your_kucoin_api_key',
            kucoin_api_secret='your_kucoin_api_secret',
            newsapi_key='your_newsapi_key'
        )
        bot.run()
    except Exception as e:
        logging.error(f"Başlangıç hatası: {str(e)}")