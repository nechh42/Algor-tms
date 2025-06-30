from src.utils.logger import setup_logger
import numpy as np

logger = setup_logger(__name__)

class GridStrategy:
    def __init__(self, client, symbol, upper_price, lower_price, n_grids=10, investment_amount=1000):
        # Parametre validasyonu
        if upper_price <= lower_price:
            raise ValueError("Upper price must be greater than lower price")
        if n_grids < 2:
            raise ValueError("Number of grids must be at least 2")
            
        self.client = client
        self.symbol = symbol
        self.upper_price = upper_price
        self.lower_price = lower_price
        self.n_grids = n_grids
        self.investment_amount = investment_amount
        self.grid_levels = self.calculate_grid_levels()
        logger.info(f"Grid strategy initialized with {n_grids} levels")
    
    def calculate_grid_levels(self):
        """Grid seviyelerini hesapla"""
        return np.linspace(self.lower_price, self.upper_price, self.n_grids)
    
    def calculate_quantity(self, price):
        """Her grid seviyesi için işlem miktarını hesapla"""
        grid_investment = self.investment_amount / (self.n_grids - 1)
        return round(grid_investment / price, 8)  # 8 decimal precision
    
    def place_grid_orders(self):
        """Grid orderları yerleştir"""
        orders = []
        for i in range(len(self.grid_levels) - 1):
            # Buy order at lower level
            buy_order = {
                'symbol': self.symbol,
                'side': 'BUY',
                'type': 'LIMIT',
                'price': self.grid_levels[i],
                'quantity': self.calculate_quantity(self.grid_levels[i])
            }
            orders.append(buy_order)
            
            # Sell order at upper level
            sell_order = {
                'symbol': self.symbol,
                'side': 'SELL',
                'type': 'LIMIT',
                'price': self.grid_levels[i+1],
                'quantity': self.calculate_quantity(self.grid_levels[i+1])
            }
            orders.append(sell_order)
        
        return orders
    
    def monitor_and_rebalance(self):
        """Grid durumunu izle ve gerekirse rebalance yap"""
        # TODO: Implement grid monitoring and rebalancing logic
        pass 