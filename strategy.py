
# strategy.py - Trading stratejisi
class TradingStrategy:
    def analyze_opportunity(self, df):
        score = 0
        last_row = df.iloc[-1]
        
        # RSI Stratejisi
        if last_row['rsi'] < 30:
            score += 1
        elif last_row['rsi'] > 70:
            score -= 1
            
        # MACD Stratejisi
        if last_row['macd'] > last_row['macd_signal']:
            score += 1
        else:
            score -= 1
            
        # Bollinger Bands Stratejisi
        if last_row['close'] < last_row['bb_lower']:
            score += 1
        elif last_row['close'] > last_row['bb_upper']:
            score -= 1
            
        # Fibonacci Stratejisi
        current_price = last_row['close']
        if current_price <= last_row['fib_236']:
            score += 1
        elif current_price >= last_row['fib_618']:
            score -= 1
            
        return score
    