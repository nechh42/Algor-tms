# database.py - Veritabanı işlemleri
import sqlite3

class Database:
    def __init__(self):
        self.conn = sqlite3.connect('trading_bot.db')
        self.create_tables()
        
    def create_tables(self):
        cursor = self.conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY,
                symbol TEXT,
                side TEXT,
                entry_price REAL,
                exit_price REAL,
                quantity REAL,
                timestamp DATETIME,
                profit_loss REAL
            )
        ''')
        self.conn.commit()
        
    def save_trade(self, trade_data):
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO trades (symbol, side, entry_price, quantity, timestamp)
            VALUES (?, ?, ?, ?, ?)
        ''', (trade_data['symbol'], trade_data['side'], 
              trade_data['price'], trade_data['quantity'],
              trade_data['timestamp']))
        self.conn.commit()
