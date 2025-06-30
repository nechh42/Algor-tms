import logging
import numpy as np
from datetime import datetime

class RiskManager:
    def __init__(self, initial_capital=1000.0, max_risk_per_trade=0.02):
        self.logger = logging.getLogger(__name__)
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.max_risk_per_trade = max_risk_per_trade
        self.positions = {}
        self.trade_history = []
        
    def calculate_position_size(self, symbol, price, volatility):
        """Dinamik pozisyon boyutu hesaplama"""
        # Volatiliteye bağlı risk ayarlaması
        risk_adjustment = 1.0 - (volatility * 0.5)  # Yüksek volatilitede daha küçük pozisyon
        max_risk_amount = self.current_capital * self.max_risk_per_trade * risk_adjustment
        
        # Kademeli kar alma seviyeleri
        take_profit_levels = [
            {'level': 1.02, 'portion': 0.3},  # %2 karda %30 sat
            {'level': 1.05, 'portion': 0.3},  # %5 karda %30 sat
            {'level': 1.08, 'portion': 0.4}   # %8 karda %40 sat
        ]
        
        position_size = max_risk_amount / price
        return {
            'size': position_size,
            'take_profit_levels': take_profit_levels,
            'stop_loss': price * 0.98  # %2 stop loss
        }
    
    def update_portfolio_risk(self, market_state):
        """Portföy risk durumunu güncelle"""
        total_risk = 0
        for symbol, position in self.positions.items():
            # Pozisyon riski hesaplama
            position_risk = position['size'] * position['current_price'] / self.current_capital
            total_risk += position_risk
            
            # Maksimum kayıp kontrolü
            if position['unrealized_pnl'] < -(self.max_risk_per_trade * self.current_capital):
                self.logger.warning(f"{symbol} için maksimum kayıp limitine ulaşıldı. Pozisyon kapatılmalı.")
                return {'action': 'close_position', 'symbol': symbol}
        
        # Portföy çeşitlendirme önerileri
        if len(self.positions) < 3 and total_risk < 0.5:
            return {'action': 'diversify', 'max_new_position_size': (0.5 - total_risk) * self.current_capital}
        
        return {'action': 'hold'}
    
    def calculate_risk_metrics(self):
        """Risk metriklerini hesapla"""
        if not self.trade_history:
            return {}
            
        profits = [trade['profit'] for trade in self.trade_history]
        
        return {
            'sharpe_ratio': self._calculate_sharpe_ratio(profits),
            'max_drawdown': self._calculate_max_drawdown(profits),
            'win_rate': len([p for p in profits if p > 0]) / len(profits),
            'risk_reward_ratio': self._calculate_risk_reward_ratio(profits)
        }
    
    def _calculate_sharpe_ratio(self, profits, risk_free_rate=0.02):
        if not profits:
            return 0
        returns = np.array(profits) / self.initial_capital
        excess_returns = returns - (risk_free_rate / 252)  # Günlük risk-free rate
        return np.mean(excess_returns) / np.std(excess_returns) if np.std(excess_returns) != 0 else 0
    
    def _calculate_max_drawdown(self, profits):
        cumulative = np.cumsum(profits)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = running_max - cumulative
        return np.max(drawdown) / self.initial_capital if len(drawdown) > 0 else 0
    
    def _calculate_risk_reward_ratio(self, profits):
        if not profits:
            return 0
        gains = [p for p in profits if p > 0]
        losses = [abs(p) for p in profits if p < 0]
        avg_gain = np.mean(gains) if gains else 0
        avg_loss = np.mean(losses) if losses else 0
        return avg_gain / avg_loss if avg_loss != 0 else 0
