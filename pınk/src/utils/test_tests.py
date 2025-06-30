import pytest
from src.utils.risk_manager import RiskManager


def test_calculate_position_size():
    rm = RiskManager(max_positions=5, risk_per_trade=0.02)
    position_size = rm.calculate_position_size(account_balance=1000, trade_risk=50)
    assert position_size == 40.0


def test_check_risk_limits():
    rm = RiskManager(max_positions=5, risk_per_trade=0.02)
    current_positions = [1, 2, 3, 4, 5]
    with pytest.raises(Exception, match="Maximum position limit reached."):
        rm.check_risk_limits(current_positions)
