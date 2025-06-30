import logging
from decimal import Decimal, ROUND_DOWN

class CapitalManager:
    def __init__(self, client):
        self.client = client
        self.logger = logging.getLogger('CapitalManager')
        self.min_position_size = 5  # Minimum pozisyon büyüklüğü (USDT)
        self.max_position_size = 100  # Maximum pozisyon büyüklüğü (USDT)
        self.position_size_percent = 0.12  # Her coin için sermayenin %12'si

    def get_total_balance(self):
        """Toplam USDT bakiyesini al"""
        try:
            balances = self.client.futures_account_balance()
            for balance in balances:
                if balance['asset'] == 'USDT':
                    return float(balance['balance'])
            return 0
        except Exception as e:
            self.logger.error(f"Error getting balance: {e}")
            return 0

    def calculate_position_sizes(self, num_pairs):
        """Her coin için optimum pozisyon büyüklüğünü hesapla"""
        try:
            total_balance = self.get_total_balance()
            self.logger.info(f"\nTotal Balance: {total_balance:.2f} USDT")

            # Her coin için pozisyon büyüklüğü
            position_size = (total_balance * self.position_size_percent)
            
            # Pozisyon büyüklüğünü sınırlar içinde tut
            position_size = max(self.min_position_size, 
                              min(self.max_position_size, position_size))
            
            # Ondalık hassasiyeti ayarla
            position_size = Decimal(str(position_size)).quantize(
                Decimal('0.1'), rounding=ROUND_DOWN)
            
            self.logger.info(f"Position Size per Pair: {position_size} USDT")
            return float(position_size)

        except Exception as e:
            self.logger.error(f"Error calculating position sizes: {e}")
            return self.min_position_size

    def adjust_leverage(self, total_exposure):
        """Kaldıracı optimize et"""
        try:
            total_balance = self.get_total_balance()
            if total_balance == 0:
                return 10  # Varsayılan kaldıraç
                
            # Risk yönetimi için kaldıraç hesapla
            current_leverage = total_exposure / total_balance
            
            # Kaldıracı 1-20 arasında tut
            optimal_leverage = min(20, max(1, round(current_leverage)))
            
            self.logger.info(f"Optimal Leverage: {optimal_leverage}x")
            return optimal_leverage
            
        except Exception as e:
            self.logger.error(f"Error adjusting leverage: {e}")
            return 10  # Hata durumunda varsayılan kaldıraç
