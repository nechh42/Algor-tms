import numpy as np
from datetime import datetime
import time

class RiskManager:
    def __init__(self, config):
        self.config = config
        self.open_positions = {}  # symbol -> pozisyon detayları
        self.position_history = []  # Geçmiş pozisyonlar
        self.daily_pnl = 0
        self.last_position_time = {}  # symbol -> son pozisyon zamanı
        self.trades_today = {}
        self.last_reset = datetime.now().date()

    def reset_daily_stats(self):
        """Günlük istatistikleri sıfırla"""
        current_date = datetime.now().date()
        if current_date > self.last_reset:
            self.daily_pnl = 0
            self.trades_today = {}
            self.last_reset = current_date

    def run_monte_carlo(self, returns, initial_balance):
        """Monte Carlo simülasyonu çalıştır"""
        try:
            # Geçmiş getirilerden ortalama ve std sapma hesapla
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            
            # Simülasyon matrisi oluştur
            days = 30  # 30 günlük simülasyon
            sims = self.config.SIMULATION_RUNS
            
            # Normal dağılımdan rastgele getiriler üret
            sim_returns = np.random.normal(mean_return, std_return, (sims, days))
            
            # Her simülasyon için equity eğrisi hesapla
            equity_curves = initial_balance * (1 + sim_returns).cumprod(axis=1)
            
            # Metrikleri hesapla
            final_values = equity_curves[:, -1]
            drawdowns = np.array([self._calculate_drawdown(curve) for curve in equity_curves])
            
            # Value at Risk (VaR)
            var = np.percentile(final_values, (1 - self.config.CONFIDENCE_LEVEL) * 100)
            
            # Expected Shortfall (ES)
            es = np.mean(final_values[final_values <= var])
            
            # Sharpe Ratio
            excess_returns = mean_return - self.config.RISK_FREE_RATE / 252  # Günlük
            sharpe = np.sqrt(252) * excess_returns / std_return
            
            # Sortino Ratio
            downside_returns = returns[returns < 0]
            sortino = np.sqrt(252) * excess_returns / np.std(downside_returns) if len(downside_returns) > 0 else np.inf
            
            return {
                'expected_return': np.mean(final_values) / initial_balance - 1,
                'var': (initial_balance - var) / initial_balance,
                'expected_shortfall': (initial_balance - es) / initial_balance,
                'max_drawdown': np.max(drawdowns),
                'sharpe_ratio': sharpe,
                'sortino_ratio': sortino,
                'win_rate': len(returns[returns > 0]) / len(returns)
            }
            
        except Exception as e:
            print(f"Monte Carlo simülasyon hatası: {e}")
            return None
    
    def _calculate_drawdown(self, equity_curve):
        """Drawdown hesapla"""
        peaks = np.maximum.accumulate(equity_curve)
        drawdowns = (peaks - equity_curve) / peaks
        return np.max(drawdowns)
    
    def can_open_position(self, symbol, balance, risk_metrics=None):
        """Yeni pozisyon açılabilir mi kontrol et"""
        try:
            # Mevcut pozisyon sayısı kontrolü
            if len(self.open_positions) >= self.config.MAX_OPEN_POSITIONS:
                return False, "Maksimum açık pozisyon sayısına ulaşıldı"
            
            # Coin başına pozisyon limiti kontrolü
            if symbol in self.open_positions:
                if len([p for p in self.open_positions[symbol] if p['active']]) >= self.config.MAX_POSITIONS_PER_COIN:
                    return False, f"{symbol} için maksimum pozisyon sayısına ulaşıldı"
            
            # Pozisyonlar arası minimum süre kontrolü
            if symbol in self.last_position_time:
                time_since_last = time.time() - self.last_position_time[symbol]
                if time_since_last < self.config.MIN_POSITION_INTERVAL:
                    return False, f"{symbol} için minimum bekleme süresi dolmadı"
            
            # Risk metrikleri kontrolü
            if risk_metrics:
                if risk_metrics['sharpe_ratio'] < self.config.MIN_SHARPE_RATIO:
                    return False, "Sharpe oranı çok düşük"
                if risk_metrics['sortino_ratio'] < self.config.MIN_SORTINO_RATIO:
                    return False, "Sortino oranı çok düşük"
                if risk_metrics['var'] > self.config.MAX_VAR_THRESHOLD:
                    return False, "VaR çok yüksek"
                if risk_metrics['win_rate'] < self.config.MIN_WIN_RATE:
                    return False, "Kazanma oranı çok düşük"
            
            # Günlük kayıp limiti kontrolü
            self.reset_daily_stats()
            if self.daily_pnl <= -balance * (self.config.MAX_DAILY_LOSS / 100):
                return False, "Günlük kayıp limitine ulaşıldı"
            
            return True, None
            
        except Exception as e:
            print(f"Pozisyon kontrolü hatası: {e}")
            return False, "Kontrol hatası"
    
    def calculate_position_size(self, balance, risk_metrics=None):
        """Pozisyon büyüklüğünü hesapla"""
        try:
            base_size = balance * (self.config.POSITION_SIZE / 100)
            
            # Risk bazlı ayarlama
            if risk_metrics:
                # Sharpe oranına göre ayarla
                sharpe_factor = min(max(risk_metrics['sharpe_ratio'] / 2, 0.5), 1.5)
                
                # VaR'a göre ayarla
                var_factor = min(max(1 - risk_metrics['var'] / self.config.MAX_VAR_THRESHOLD, 0.5), 1.5)
                
                # Win rate'e göre ayarla
                win_factor = min(max(risk_metrics['win_rate'] / self.config.MIN_WIN_RATE, 0.5), 1.5)
                
                # Faktörleri birleştir
                risk_factor = np.mean([sharpe_factor, var_factor, win_factor])
                
                # Pozisyon büyüklüğünü ayarla
                base_size *= risk_factor
            
            # Minimum ve maksimum limitleri uygula
            position_size = max(min(base_size, balance * 0.75), self.config.MIN_POSITION_SIZE)
            
            return position_size
            
        except Exception as e:
            print(f"Pozisyon büyüklüğü hesaplama hatası: {e}")
            return self.config.MIN_POSITION_SIZE

    def calculate_position_size_kelly(self, balance: float, risk_per_trade: float) -> float:
        """Kelly Criterion bazlı pozisyon büyüklüğü hesapla"""
        # Basit Kelly formülü: K = W - [(1-W)/R]
        # W = kazanma oranı (0.5 varsayılan)
        # R = Risk:Reward oranı (2 varsayılan)
        win_rate = 0.5  # Başlangıç için 0.5 kullan
        risk_reward = 2  # 1:2 risk:reward
        
        kelly_fraction = win_rate - ((1 - win_rate) / risk_reward)
        kelly_fraction = max(0.1, min(kelly_fraction, self.config.KELLY_FRACTION))  # 0.1 ile KELLY_FRACTION arası sınırla
        
        position_size = balance * kelly_fraction * (risk_per_trade / 100)
        max_position_size = balance * self.config.POSITION_SIZE
        
        return min(position_size, max_position_size)

    def update_stats(self, pnl: float):
        """İstatistikleri güncelle"""
        self.daily_pnl += pnl

    def add_trade(self, trade_info: dict):
        """Yeni trade ekle"""
        self.trades_today[trade_info['symbol']] = trade_info

    def remove_trade(self, symbol: str):
        """Trade'i kaldır"""
        if symbol in self.trades_today:
            del self.trades_today[symbol]

    def update_position_status(self, symbol: str, current_price: float):
        """Pozisyon durumunu güncelle ve trailing stop kontrol et"""
        if symbol not in self.trades_today:
            return None
            
        trade = self.trades_today[symbol]
        entry_price = trade['entry_price']
        side = trade['side']
        
        # Kar/Zarar hesapla
        if side == 'buy':
            pnl_percent = ((current_price - entry_price) / entry_price) * 100
            highest_price = trade.get('highest_price', entry_price)
            
            # En yüksek fiyatı güncelle
            if current_price > highest_price:
                trade['highest_price'] = current_price
                
            # Trailing stop kontrolü
            if pnl_percent >= self.config.TRAILING_ACTIVATION:
                trailing_stop_price = current_price * (1 - self.config.TRAILING_STOP / 100)
                if trailing_stop_price > trade.get('trailing_stop_price', 0):
                    trade['trailing_stop_price'] = trailing_stop_price
                    print(f"Trailing stop güncellendi: {symbol} -> {trailing_stop_price:.2f}")
                
                # Trailing stop tetiklendi mi?
                if trade.get('trailing_stop_price') and current_price < trade['trailing_stop_price']:
                    return 'close'
                    
        else:  # side == 'sell'
            pnl_percent = ((entry_price - current_price) / entry_price) * 100
            lowest_price = trade.get('lowest_price', entry_price)
            
            # En düşük fiyatı güncelle
            if current_price < lowest_price:
                trade['lowest_price'] = current_price
                
            # Trailing stop kontrolü
            if pnl_percent >= self.config.TRAILING_ACTIVATION:
                trailing_stop_price = current_price * (1 + self.config.TRAILING_STOP / 100)
                if trailing_stop_price < trade.get('trailing_stop_price', float('inf')):
                    trade['trailing_stop_price'] = trailing_stop_price
                    print(f"Trailing stop güncellendi: {symbol} -> {trailing_stop_price:.2f}")
                
                # Trailing stop tetiklendi mi?
                if trade.get('trailing_stop_price') and current_price > trade['trailing_stop_price']:
                    return 'close'
        
        # Normal TP/SL kontrolü
        if pnl_percent >= self.config.TAKE_PROFIT:
            return 'close'
        elif pnl_percent <= -self.config.STOP_LOSS:
            return 'close'
            
        # Pozisyon durumunu güncelle
        trade['current_pnl'] = pnl_percent
        self.trades_today[symbol] = trade
        
        return None

    def position_size_adjustment(self, symbol: str, volatility: float, trend_strength: float) -> float:
        """Pozisyon büyüklüğünü dinamik olarak ayarla"""
        base_size = self.config.POSITION_SIZE
        
        # Volatiliteye göre ayarla (yüksek volatilite = düşük pozisyon)
        if volatility > 50:
            base_size *= 0.8
        elif volatility > 30:
            base_size *= 0.9
            
        # Trend gücüne göre ayarla (güçlü trend = büyük pozisyon)
        if trend_strength > 50:
            base_size *= 1.2
        elif trend_strength > 30:
            base_size *= 1.1
            
        # Minimum ve maksimum sınırlar
        base_size = max(0.1, min(base_size, 0.5))  # %10-%50 arası
        
        return base_size
