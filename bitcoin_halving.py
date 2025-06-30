import pandas as pd
import pandas_ta as ta
from binance.client import Client
from bitcoin_halving import bitcoin_halving
import config

class BitcoinHalvingPro:
    def __init__(self, client, symbol="BTCUSDT", timeframe="1d"):
        self.client = client
        self.symbol = symbol
        self.timeframe = timeframe
        self.position = None

    def get_data(self, limit=1000):
        klines = self.client.futures_klines(symbol=self.symbol, interval=self.timeframe, limit=limit)
        df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time',
                                           'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignore'])
        df['close'] = pd.to_numeric(df['close'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df

    def calculate_indicators(self, df):
        df['rsi'] = ta.rsi(df['close'], length=14)
        df['macd'] = ta.macd(df['close'])['MACD_12_26_9']
        df['macd_signal'] = ta.macd(df['close'])['MACDs_12_26_9']
        halving_info = bitcoin_halving(900000)  # Örnek: Mevcut blok yüksekliği
        df['is_halving_soon'] = df['timestamp'].apply(
            lambda x: abs((x - pd.to_datetime(halving_info['estimated_date'])).days) < 60
        )
        return df

    def generate_signal(self, df):
        last = df.iloc[-1]
        if last['is_halving_soon'] and last['rsi'] < 50 and last['macd'] > last['macd_signal']:
            return "BUY"
        elif last['rsi'] > 70 and last['macd'] < last['macd_signal']:
            return "SELL"
        return "HOLD"

    def execute_trade(self, signal, price, quantity):
        if signal == "BUY" and not self.position:
            stop_loss = price * (1 - 0.05)  # %5 stop-loss
            take_profit = price * (1 + 0.10)  # %10 take-profit
            order = self.client.futures_create_order(
                symbol=self.symbol, side="BUY", type="MARKET", quantity=quantity
            )
            self.client.futures_create_order(
                symbol=self.symbol, side="SELL", type="STOP_MARKET", stopPrice=stop_loss, closePosition=True
            )
            self.client.futures_create_order(
                symbol=self.symbol, side="SELL", type="TAKE_PROFIT_MARKET", stopPrice=take_profit, closePosition=True
            )
            self.position = {"side": "BUY", "price": price, "quantity": quantity}
            print(f"BUY: {quantity} {self.symbol} @ {price}, SL: {stop_loss}, TP: {take_profit}")
        elif signal == "SELL" and self.position:
            self.client.futures_create_order(
                symbol=self.symbol, side="SELL", type="MARKET", quantity=self.position['quantity'], reduceOnly=True
            )
            self.position = None
            print(f"SELL: {quantity} {self.symbol} @ {price}")

    def run(self):
        df = self.get_data()
        df = self.calculate_indicators(df)
        signal = self.generate_signal(df)
        if signal != "HOLD":
            price = float(df['close'].iloc[-1])
            quantity = 0.001  # Örnek: 100$ için 0.001 BTC
            self.execute_trade(signal, price, quantity)