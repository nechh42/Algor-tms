import logging
from datetime import datetime, timedelta

class PerformanceAnalyzer:
    def __init__(self):
        self.logger = logging.getLogger('PerformanceAnalyzer')
        self.start_time = datetime.now()
        self.trades = []
        self.min_success_rate = 55  # Minimum başarı oranı %
        self.min_profit_factor = 1.5  # Minimum profit faktörü
        self.analysis_interval = 4  # 4 saatlik analiz
        
    def add_trade(self, symbol, entry_price, exit_price, position_size, side):
        """Trade sonuçlarını ekle"""
        profit_loss = (exit_price - entry_price) * position_size
        if side == 'SELL':  # Short pozisyonlar için tersine çevir
            profit_loss = -profit_loss
            
        self.trades.append({
            'symbol': symbol,
            'profit_loss': profit_loss,
            'timestamp': datetime.now()
        })
        
    def analyze_performance(self):
        """4 saatlik performans analizi"""
        if not self.trades:
            return False, "Henüz yeterli trade verisi yok"
            
        # Son 4 saatlik tradeleri filtrele
        recent_trades = [t for t in self.trades 
                        if t['timestamp'] > datetime.now() - timedelta(hours=self.analysis_interval)]
        
        if not recent_trades:
            return False, "Son 4 saatte trade yok"
            
        # Temel metrikler
        winning_trades = [t for t in recent_trades if t['profit_loss'] > 0]
        total_profit = sum(t['profit_loss'] for t in winning_trades)
        total_loss = abs(sum(t['profit_loss'] for t in recent_trades if t['profit_loss'] < 0))
        
        # Başarı oranı
        win_rate = (len(winning_trades) / len(recent_trades)) * 100
        
        # Profit faktörü
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        
        # Sonuçları logla
        self.logger.info("\n=== 4 Saatlik Performans Analizi ===")
        self.logger.info(f"Toplam Trade: {len(recent_trades)}")
        self.logger.info(f"Kazanan Trade: {len(winning_trades)}")
        self.logger.info(f"Başarı Oranı: {win_rate:.1f}%")
        self.logger.info(f"Toplam Kar: {total_profit:.2f} USDT")
        self.logger.info(f"Toplam Zarar: {total_loss:.2f} USDT")
        self.logger.info(f"Profit Faktörü: {profit_factor:.2f}")
        
        # Her coinin performansını analiz et
        by_symbol = {}
        for trade in recent_trades:
            symbol = trade['symbol']
            if symbol not in by_symbol:
                by_symbol[symbol] = []
            by_symbol[symbol].append(trade)
            
        self.logger.info("\nCoin Bazlı Performans:")
        for symbol, trades in by_symbol.items():
            wins = len([t for t in trades if t['profit_loss'] > 0])
            total = len(trades)
            win_rate = (wins / total) * 100
            profit = sum(t['profit_loss'] for t in trades)
            self.logger.info(f"{symbol}: Başarı: {win_rate:.1f}%, P/L: {profit:.2f} USDT")
        
        # Gerçek trading'e geçiş için değerlendirme
        ready_for_live = (win_rate >= self.min_success_rate and 
                         profit_factor >= self.min_profit_factor and
                         len(recent_trades) >= 10)  # En az 10 trade
        
        if ready_for_live:
            message = "🚀 Performans kriterleri karşılandı! Gerçek trading'e geçilebilir."
        else:
            message = "⚠️ Test modunda devam edilmeli. Kriterler: Başarı >55%, P-Faktör >1.5"
            
        self.logger.info(message)
        return ready_for_live, message
