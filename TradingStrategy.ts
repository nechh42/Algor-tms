import { TechnicalIndicators } from '../indicators/TechnicalIndicators.js';
import { TradeSignal, MACDResult } from '../types/trading.js';
import { ProcessedCandle } from '../types/binance.js';
import { Logger } from '../utils/logger.js';

export class TradingStrategy {
  private indicators: TechnicalIndicators;

  constructor() {
    this.indicators = new TechnicalIndicators();
  }

  async analyzeMarket(data: ProcessedCandle[]): Promise<TradeSignal[]> {
    const prices = data.map(candle => candle.close);
    const rsi = this.indicators.calculateRSI(prices);
    const macd = this.indicators.calculateMACD(prices);

    const signals = this.generateSignals(rsi, macd);
    return this.applyRiskManagement(signals, data);
  }

  private generateSignals(rsi: number[], macd: MACDResult): TradeSignal[] {
    const signals: TradeSignal[] = [];
    const lastRSI = rsi[rsi.length - 1];
    const lastMACD = macd.histogram[macd.histogram.length - 1];

    if (lastRSI < 30 && lastMACD > 0) {
      signals.push({ type: 'BUY', strength: 'STRONG' });
    } else if (lastRSI > 70 && lastMACD < 0) {
      signals.push({ type: 'SELL', strength: 'STRONG' });
    }

    return signals;
  }

  private applyRiskManagement(signals: TradeSignal[], data: ProcessedCandle[]): TradeSignal[] {
    return signals.map(signal => ({
      ...signal,
      stopLoss: this.calculateStopLoss(signal, data),
      takeProfit: this.calculateTakeProfit(signal, data)
    }));
  }

  private calculateStopLoss(signal: TradeSignal, data: ProcessedCandle[]): number {
    const lastPrice = data[data.length - 1].close;
    return signal.type === 'BUY' 
      ? lastPrice * 0.98  // 2% stop loss for long positions
      : lastPrice * 1.02; // 2% stop loss for short positions
  }

  private calculateTakeProfit(signal: TradeSignal, data: ProcessedCandle[]): number {
    const lastPrice = data[data.length - 1].close;
    return signal.type === 'BUY'
      ? lastPrice * 1.04  // 4% take profit for long positions
      : lastPrice * 0.96; // 4% take profit for short positions
  }
}