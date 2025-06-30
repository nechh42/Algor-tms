import ccxt
import time
import logging
import pandas as pd
import ta
from transformers import pipeline

class NewsArbHybrid:
    def __init__(self, binance_api_key, binance_api_secret, capital=100, fee_rate=0.001, max_risk=0.01):
        self.binance = ccxt.binance({
            'apiKey': '',
            'secret': '',
            'enableRateLimit': True
        })
        self.capital = capital  # $100
        self.fee_rate = fee_rate  # %0.1 işlem ücreti
        self.max_risk = max_risk  # Maksimum risk: %1
        self.sentiment_threshold = 0.5  # Sentiment eşiği
        self.sentiment_analyzer = pipeline("sentiment-analysis")
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

    def fetch_x_posts(self):
        """X’ten BTC post’ları (senin topladıkların)."""
        # Buraya senin topladığın 5 post’u ekle
        mock_posts = [
            "Bitcoin to the moon! #BTC",
            "BTC is pumping hard! #BTC",
            "Bearish on BTC, sell now #BTC",
            "Elon tweeted about BTC, bullish! #BTC",
            "BTC crash incoming #BTC"
        ]
        return mock_posts

    def get_sentiment(self):
        """X post’lardan sentiment skoru hesaplar."""
        posts = self.fetch_x_posts()
        if not posts:
            logging.info("Post bulunamadı, sentiment 0.")
            return 0
        scores = [self.sentiment_analyzer(post)[0]['score'] 
                  if self.sentiment_analyzer(post)[0]['label'] == 'POSITIVE' 
                  else -self.sentiment_analyzer(post)[0]['score'] for post in posts]
        return sum(scores) / len(scores)

    def fetch_prices(self, symbol='BTC/USDT'):
        """Binance’ten fiyatları çeker."""
        try:
            ticker = self.binance.fetch_ticker(symbol)
            return {'bid': ticker['bid'], 'ask': ticker['ask']}
        except Exception as e:
            logging.error(f"Fiyat alınırken hata: {e}")
            return None

    def find_arbitrage_opportunity(self, symbol='BTC/USDT'):
        """Arbitraj fırsatlarını tespit eder."""
        sentiment = self.get_sentiment()
        if abs(sentiment) < self.sentiment_threshold:
            logging.info(f"Düşük sentiment skoru ({sentiment:.2f}), tarama durduruldu.")
            return None

        prices = self.fetch_prices(symbol)
        if not prices:
            return None

        amount = (self.capital * self.max_risk) / prices['ask']  # %1 risk
        bid, ask = prices['bid'], prices['ask']
        profit = bid - ask
        fees = (ask * self.fee_rate) + (bid * self.fee_rate)
        net_profit = profit - fees

        if net_profit > 0:
            return {
                'buy_price': ask,
                'sell_price': bid,
                'amount': amount,
                'net_profit': net_profit * amount,
                'sentiment': sentiment
            }
        return None

    def execute_trade(self, opportunity, symbol='BTC/USDT'):
        """Test için simüle emir (canlı için gerçek emir)."""
        if not opportunity:
            return False
        logging.info(f"Alım: {opportunity['amount']:.6f} BTC @ {opportunity['buy_price']}")
        logging.info(f"Satım: {opportunity['amount']:.6f} BTC @ {opportunity['sell_price']}")
        logging.info(f"Tahmini kâr: ${opportunity['net_profit']:.2f}, Sentiment: {opportunity['sentiment']:.2f}")
        # Canlı için: self.binance.create_limit_buy_order() ve sell_order
        return True

    def run(self, symbol='BTC/USDT', interval=10):
        """Botu çalıştırır."""
        while True:
            opportunity = self.find_arbitrage_opportunity(symbol)
            if opportunity:
                self.execute_trade(opportunity, symbol)
            else:
                logging.info("Fırsat bulunamadı.")
            time.sleep(interval)

if __name__ == "__main__":
    try:
        bot = NewsArbHybrid(
            binance_api_key='your_api_key',  # Binance API anahtarınızı buraya girin
            binance_api_secret='your_api_secret'
        )
        bot.run()
    except Exception as e:
        logging.error(f"Hata: {str(e)}")