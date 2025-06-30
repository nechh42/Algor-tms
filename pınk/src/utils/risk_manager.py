class TradingBotError(Exception):
    pass

class InsufficientFundsError(TradingBotError):
    pass

class APIError(TradingBotError):
    pass

class RiskManager:
    def __init__(self, max_positions, risk_per_trade):
        self.max_positions = max_positions
        self.risk_per_trade = risk_per_trade

    def calculate_position_size(self, account_balance, trade_risk):
        return account_balance * self.risk_per_trade / trade_risk

    def check_risk_limits(self, current_positions):
        if len(current_positions) >= self.max_positions:
            raise Exception("Maximum position limit reached.")
