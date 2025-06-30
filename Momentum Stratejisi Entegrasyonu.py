import pandas as pd
import numpy as np
import ta
from binance.client import Client

class GridMomentumFusion:
    def __init__(self, grid_size=500, num_grids=10, capital_per_grid=1000, api_key=None, api_secret=None):
        self.grid_size = grid_size
        self.num_grids = num_grids
        self.capital_per_grid = capital_per_grid
        self.api_key = api_key or 'your_api_key'
        self.api_secret = api_secret or 'your_api_secret'
        self.client = Client(self.api_key, self.api_secret)

    def get_market_data(self, symbol="BTCUSDT", timeframe="1h", limit=100):
        """Binance’ten gerçek zamanlı veri alır."""
        try:
            klines = self.client.get_historical_klines(symbol, timeframe, f"{limit} hours ago UTC")
            columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time',
                       'quote_asset_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignore']
            data = pd.DataFrame(klines, columns=columns)
            data = data[['open', 'high', 'low', 'close', 'volume']].astype(float)
            data['sar'] = ta.trend.PSARIndicator(data['high'], data['low'], data['close']).psar()
            data['rsi'] = ta.momentum.RSIIndicator(data['close']).rsi()
            data['atr'] = ta.volatility.AverageTrueRange(data['high'], data['low'], data['close']).average_true_range()
            return data.dropna()
        except Exception as e:
            print(f"Veri alınırken hata: {e}")
            return None

    def is_trending(self, data):
        """Piyasanın trendde olup olmadığını kontrol eder."""
        latest_data = data.iloc[-1]
        if latest_data['close'] > latest_data['sar'] and latest_data['rsi'] > 70:
            return "up"  # Yükseliş trendi
        elif latest_data['close'] < latest_data['sar'] and latest_data['rsi'] < 30:
            return "down"  # Düşüş trendi
        return None  # Range-bound piyasa

    def calculate_grid_levels(self, current_price, atr):
        """Dinamik grid seviyelerini hesaplar."""
        grid_interval = atr * self.grid_size
        grid_levels = []
        for i in range(-self.num_grids // 2, self.num_grids // 2 + 1):
            level = current_price + (i * grid_interval)
            grid_levels.append(level)
        return grid_levels

    def place_orders(self, current_price, atr):
        """Alım ve satım emirlerini oluşturur."""
        grid_levels = self.calculate_grid_levels(current_price, atr)
        orders = []
        for level in grid_levels:
            if level < current_price:
                orders.append({"type": "buy", "price": level, "amount": self.capital_per_grid / level})
            elif level > current_price:
                orders.append({"type": "sell", "price": level, "amount": self.capital_per_grid / level})
        return orders

    def generate_signal(self, symbol="BTCUSDT"):
        """Strateji sinyali üretir."""
        data = self.get_market_data(symbol)
        if data is None:
            return "Hata: Veri alınamadı."
        
        current_price = data['close'].iloc[-1]
        atr = data['atr'].iloc[-1]
        trend = self.is_trending(data)

        if trend:
            return f"Trend algılandı ({trend}), grid trading durduruldu."
        else:
            orders = self.place_orders(current_price, atr)
            return {"status": "Grid trading aktif", "orders": orders}

if __name__ == "__main__":
    try:
        # API anahtarlarınızı buraya girin
        strategy = GridMomentumFusion(grid_size=2, num_grids=10, capital_per_grid=1000)
        result = strategy.generate_signal()
        print("GridMomentumFusion çalışıyor...")
        print(result)
    except Exception as e:
        print(f"HATA OLUŞTU: {str(e)}")