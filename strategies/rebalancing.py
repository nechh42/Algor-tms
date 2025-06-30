from src.utils.logger import setup_logger
import pandas as pd
from typing import Dict, List
import numpy as np

logger = setup_logger(__name__)

class RebalancingStrategy:
    def __init__(self, client, portfolio_weights: Dict[str, float], rebalance_threshold=0.1):
        self.client = client
        self.target_weights = portfolio_weights
        self.rebalance_threshold = rebalance_threshold
        logger.info(f"Rebalancing strategy initialized with weights: {portfolio_weights}")
    
    def get_current_portfolio(self) -> Dict[str, float]:
        """Mevcut portfolio durumunu al"""
        try:
            portfolio = {}
            total_value = 0
            
            # Tüm varlıkların değerini hesapla
            for asset in self.target_weights.keys():
                balance = float(self.client.get_asset_balance(asset=asset)['free'])
                if asset != 'USDT':
                    price = float(self.client.get_symbol_ticker(symbol=f"{asset}USDT")['price'])
                    value = balance * price
                else:
                    value = balance
                
                portfolio[asset] = value
                total_value += value
            
            # Ağırlıkları hesapla
            for asset in portfolio:
                portfolio[asset] = portfolio[asset] / total_value
            
            return portfolio
            
        except Exception as e:
            logger.error(f"Error getting current portfolio: {e}")
            raise
    
    def calculate_rebalance_trades(self) -> List[Dict]:
        """Rebalancing için gerekli işlemleri hesapla"""
        try:
            current_portfolio = self.get_current_portfolio()
            trades = []
            
            for asset, target_weight in self.target_weights.items():
                current_weight = current_portfolio.get(asset, 0)
                weight_diff = target_weight - current_weight
                
                # Rebalance threshold'u aşıldı mı kontrol et
                if abs(weight_diff) > self.rebalance_threshold:
                    trades.append({
                        'asset': asset,
                        'side': 'BUY' if weight_diff > 0 else 'SELL',
                        'weight_difference': weight_diff
                    })
            
            return trades
            
        except Exception as e:
            logger.error(f"Error calculating rebalance trades: {e}")
            raise
    
    def execute_rebalancing(self):
        """Rebalancing işlemlerini gerçekleştir"""
        try:
            trades = self.calculate_rebalance_trades()
            orders = []
            
            for trade in trades:
                # İşlem miktarını hesapla
                portfolio_value = self.get_portfolio_value()
                trade_value = abs(trade['weight_difference']) * portfolio_value
                
                if trade['asset'] != 'USDT':
                    price = float(self.client.get_symbol_ticker(symbol=f"{trade['asset']}USDT")['price'])
                    quantity = trade_value / price
                else:
                    quantity = trade_value
                
                # İşlemi gerçekleştir
                order = self.client.create_order(
                    symbol=f"{trade['asset']}USDT",
                    side=trade['side'],
                    type='MARKET',
                    quantity=quantity
                )
                orders.append(order)
            
            logger.info(f"Rebalancing executed: {orders}")
            return orders
            
        except Exception as e:
            logger.error(f"Error executing rebalancing: {e}")
            raise 