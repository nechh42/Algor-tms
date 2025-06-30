
# risk_manager.py - Risk yönetimi
class RiskManager:
    def __init__(self, config):
        self.config = config
        
    def calculate_position_size(self, balance, current_price):
        max_position = balance * self.config.RISK_PER_TRADE
        return max_position / current_price
        
    def calculate_stop_loss(self, entry_price, side):
        if side == 'buy':
            return entry_price * (1 - self.config.STOP_LOSS_PERCENT / 100)
        return entry_price * (1 + self.config.STOP_LOSS_PERCENT / 100)
