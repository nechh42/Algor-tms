# core/execution.py (Güncellenmiş)
import time
import logging
from config import settings
# market.py ve execution.py üst kısımlarına ekleyin:
import config.settings as settings

class QuantumTrader:
    def __init__(self, initial_balance=100.0):
        self.balance = initial_balance
        self.position = None
        self.entry_price = 0.0
        self.position_size = 0.0
        self.trade_count = 0
        self.commission_rate = 0.001  # %0.1 komisyon
        self.last_trade_time = 0
        self.trade_history = []
        
        # Gerçekçi spread simülasyonu
        self.spread_multiplier = 1.0002  # %0.02 spread
        
        # Statistikler
        self.wins = 0
        self.losses = 0

    def execute_trade(self, signal, current_price):
        if self.position:
            return 0  # Zaten pozisyon var
            
        if time.time() - self.last_trade_time < 60:
            return 0  # Sık işlem koruması
            
        try:
            # Spread uygula (gerçekçi fiyat)
            if signal == 'long':
                entry_price = current_price * self.spread_multiplier
            else:
                entry_price = current_price / self.spread_multiplier
                
            # Pozisyon büyüklüğü (%1 risk)
            risk_amount = self.balance * settings.RISK_PER_TRADE
            # %1 stop
            self.position_size = risk_amount / (current_price * 0.01)
            
            # Komisyon hesapla
            commission = self.position_size * entry_price * self.commission_rate
            if commission > risk_amount * 0.3:
                logging.warning("Komisyon risk limitini aştı, işlem iptal")
                return 0
                
            # Pozisyonu aç
            self.position = signal
            self.entry_price = entry_price
            self.balance -= commission
            self.last_trade_time = time.time()
            
            logging.info(f"{signal.upper()} pozisyon açıldı | "
                         f"Giriş: ${entry_price:.2f} | "
                         f"Miktar: {self.position_size:.6f}")
            
            # Pozisyonu kapat (test amaçlı hemen)
            profit = self.close_position(current_price)
            return profit
            
        except Exception as e:
            logging.error(f"İşlem hatası: {str(e)}")
            return 0

    def close_position(self, current_price):
        if not self.position:
            return 0
            
        try:
            # Spread uygula
            if self.position == 'long':
                exit_price = current_price / self.spread_multiplier
                profit = (exit_price - self.entry_price) * self.position_size
            else:
                exit_price = current_price * self.spread_multiplier
                profit = (self.entry_price - exit_price) * self.position_size
                
            # Komisyon
            commission = self.position_size * exit_price * self.commission_rate
            net_profit = profit - commission
            
            # Bakiyeyi güncelle
            self.balance += net_profit
            self.trade_count += 1
            
            # İstatistik
            if net_profit > 0:
                self.wins += 1
            else:
                self.losses += 1
                
            # Kayıt
            trade_record = {
                'time': time.strftime("%Y-%m-%d %H:%M:%S"),
                'symbol': settings.SYMBOL,
                'direction': self.position,
                'entry': self.entry_price,
                'exit': exit_price,
                'profit': net_profit,
                'balance': self.balance
            }
            self.trade_history.append(trade_record)
            
            # Pozisyonu sıfırla
            self.position = None
            self.position_size = 0.0
            
            return net_profit
            
        except Exception as e:
            logging.error(f"Kapatma hatası: {str(e)}")
            return 0

    def get_performance(self):
        """Performans raporu"""
        win_rate = self.wins / self.trade_count * 100 if self.trade_count > 0 else 0
        return {
            'balance': self.balance,
            'trades': self.trade_count,
            'win_rate': win_rate,
            'wins': self.wins,
            'losses': self.losses
        }

def get_trade_manager():
    global _trade_manager
    if '_trade_manager' not in globals():
        _trade_manager = QuantumTrader(initial_balance=100.0)
    return _trade_manager