"""
Pozisyon ve risk yönetimi
"""
import ccxt
from config.trading_config import RISK_MANAGEMENT, TRADE_EXECUTION

class PositionManager:
    def __init__(self):
        self.exchange = None
        self.open_positions = {}
        self.daily_pnl = 0
        self.min_position_size = 10  # Minimum 10 USDT
        self.max_position_size = 1000  # Maksimum 1000 USDT
        
    def calculate_position_size(self, current_price, signal_strength):
        """
        İşlem büyüklüğünü hesapla
        """
        try:
            # Bakiyeyi kontrol et
            balance = self.get_balance()
            if balance < self.min_position_size:
                print(f"Yetersiz bakiye: {balance} USDT")
                return 0
                
            # Risk bazlı pozisyon büyüklüğü
            risk_amount = balance * (RISK_MANAGEMENT['position_size_percent'] / 100)
            
            # Sinyal gücüne göre pozisyon büyüklüğünü ayarla
            position_size = risk_amount * (signal_strength / 5)  # 5 maksimum sinyal gücü
            
            # Minimum ve maksimum limitleri kontrol et
            position_size = max(self.min_position_size, min(position_size, self.max_position_size))
            
            # Lot büyüklüğünü ayarla
            quantity = position_size / current_price
            
            return round(quantity, 6)  # 6 decimal hassasiyet
            
        except Exception as e:
            print(f"Pozisyon büyüklüğü hesaplama hatası: {str(e)}")
            return 0
            
    def open_position(self, symbol, current_price, signal_strength):
        """
        Yeni pozisyon aç
        """
        try:
            # Günlük işlem limitini kontrol et
            if len(self.open_positions) >= TRADE_EXECUTION['max_trades_per_day']:
                print("Günlük maksimum işlem limitine ulaşıldı")
                return False
                
            # Günlük zarar limitini kontrol et
            if self.daily_pnl <= -(RISK_MANAGEMENT['max_daily_loss_percent'] / 100):
                print("Günlük zarar limitine ulaşıldı")
                return False
                
            # Pozisyon büyüklüğünü hesapla
            quantity = self.calculate_position_size(current_price, signal_strength)
            if quantity <= 0:
                return False
                
            # Stop loss ve take profit hesapla
            stop_loss = self.calculate_stop_loss(current_price, 'long')
            take_profit = self.calculate_take_profit(current_price, 'long')
            
            print(f"\nYeni pozisyon açılıyor:")
            print(f"Symbol: {symbol}")
            print(f"Miktar: {quantity}")
            print(f"Giriş Fiyatı: {current_price}")
            print(f"Stop Loss: {stop_loss}")
            print(f"Take Profit: {take_profit}")
            
            # Pozisyonu kaydet
            self.open_positions[symbol] = {
                'quantity': quantity,
                'entry_price': current_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'type': 'long'
            }
            
            return True
            
        except Exception as e:
            print(f"Pozisyon açma hatası: {str(e)}")
            return False
            
    def close_position(self, symbol, current_price):
        """
        Pozisyonu kapat
        """
        try:
            if symbol not in self.open_positions:
                return False
                
            position = self.open_positions[symbol]
            pnl = (current_price - position['entry_price']) * position['quantity']
            
            print(f"\nPozisyon kapatılıyor:")
            print(f"Symbol: {symbol}")
            print(f"PNL: {pnl:.2f} USDT")
            
            # Günlük PNL'i güncelle
            self.daily_pnl += pnl
            
            # Pozisyonu sil
            del self.open_positions[symbol]
            
            return True
            
        except Exception as e:
            print(f"Pozisyon kapatma hatası: {str(e)}")
            return False
            
    def check_open_positions(self):
        """
        Açık pozisyonları kontrol et
        """
        for symbol, position in list(self.open_positions.items()):
            try:
                current_price = self.get_current_price(symbol)
                
                # Stop loss kontrolü
                if current_price <= position['stop_loss']:
                    print(f"\n{symbol} için stop loss tetiklendi")
                    self.close_position(symbol, current_price)
                    continue
                    
                # Take profit kontrolü
                if current_price >= position['take_profit']:
                    print(f"\n{symbol} için take profit tetiklendi")
                    self.close_position(symbol, current_price)
                    continue
                    
                # Trailing stop kontrolü
                if current_price >= position['entry_price'] * (1 + RISK_MANAGEMENT['trailing_stop_activation']/100):
                    new_stop_loss = current_price * (1 - RISK_MANAGEMENT['trailing_stop_distance']/100)
                    if new_stop_loss > position['stop_loss']:
                        position['stop_loss'] = new_stop_loss
                        print(f"\n{symbol} için trailing stop güncellendi: {new_stop_loss}")
                        
            except Exception as e:
                print(f"{symbol} pozisyon kontrolü hatası: {str(e)}")
                
    def get_balance(self):
        """
        USDT bakiyesini getir
        """
        try:
            balance = self.exchange.fetch_balance()
            return balance['USDT']['free']
        except:
            return 0
            
    def get_current_price(self, symbol):
        """
        Güncel fiyatı getir
        """
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            return ticker['last']
        except:
            return 0
            
    def calculate_stop_loss(self, entry_price, position_type):
        """
        Stop loss seviyesini hesapla
        """
        risk = entry_price * (RISK_MANAGEMENT['stop_loss_percent'] / 100)
        if position_type == 'long':
            return entry_price - risk
        else:
            return entry_price + risk
            
    def calculate_take_profit(self, entry_price, position_type):
        """
        Take profit seviyesini hesapla
        """
        risk = entry_price * (RISK_MANAGEMENT['stop_loss_percent'] / 100)
        if position_type == 'long':
            return entry_price + (risk * 2)
        else:
            return entry_price - (risk * 2)
