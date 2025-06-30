export class TechnicalIndicators {
  calculateRSI(prices: number[], period: number = 14): number[] {
    const rsi: number[] = [];
    let gains: number[] = [];
    let losses: number[] = [];

    // Calculate price changes
    for (let i = 1; i < prices.length; i++) {
      const change = prices[i] - prices[i - 1];
      gains.push(change > 0 ? change : 0);
      losses.push(change < 0 ? Math.abs(change) : 0);
    }

    // Calculate RSI
    for (let i = period; i < prices.length; i++) {
      const avgGain = gains.slice(i - period, i).reduce((a, b) => a + b) / period;
      const avgLoss = losses.slice(i - period, i).reduce((a, b) => a + b) / period;
      const rs = avgGain / avgLoss;
      rsi.push(100 - (100 / (1 + rs)));
    }

    return rsi;
  }

  calculateMACD(prices: number[], fastPeriod: number = 12, slowPeriod: number = 26, signalPeriod: number = 9) {
    const ema12 = this.calculateEMA(prices, fastPeriod);
    const ema26 = this.calculateEMA(prices, slowPeriod);
    const macdLine = ema12.map((value, index) => value - ema26[index]);
    const signalLine = this.calculateEMA(macdLine, signalPeriod);

    return {
      macdLine,
      signalLine,
      histogram: macdLine.map((value, index) => value - signalLine[index])
    };
  }

  private calculateEMA(prices: number[], period: number): number[] {
    const ema: number[] = [];
    const multiplier = 2 / (period + 1);

    // Initialize EMA with SMA
    const sma = prices.slice(0, period).reduce((a, b) => a + b) / period;
    ema.push(sma);

    for (let i = period; i < prices.length; i++) {
      ema.push((prices[i] - ema[ema.length - 1]) * multiplier + ema[ema.length - 1]);
    }

    return ema;
  }
}