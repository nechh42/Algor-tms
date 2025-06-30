import { DataCollector } from '../services/DataCollector.js';
import { TradingStrategy } from '../strategies/TradingStrategy.js';
import { BinanceConfig } from '../types/binance.js';
import { TradeSignal } from '../types/trading.js';
import { Logger } from '../utils/logger.js';

export class TradingBot {
  private dataCollector: DataCollector;
  private strategy: TradingStrategy;
  private isRunning: boolean = false;

  constructor(config: BinanceConfig) {
    this.dataCollector = new DataCollector(config);
    this.strategy = new TradingStrategy();
  }

  async start(symbol: string, interval: string = '1h') {
    this.isRunning = true;
    Logger.info(`Starting trading bot for ${symbol} with ${interval} interval`);
    
    while (this.isRunning) {
      try {
        const historicalData = await this.dataCollector.getHistoricalData(symbol, interval, 100);
        const signals = await this.strategy.analyzeMarket(historicalData);
        await this.executeTrades(signals, symbol);
        await this.sleep(60000); // Wait for 1 minute before next iteration
      } catch (error) {
        Logger.error('Error in trading loop:', error);
        await this.sleep(60000); // Wait before retrying
      }
    }
  }

  stop() {
    Logger.info('Stopping trading bot');
    this.isRunning = false;
  }

  private async executeTrades(signals: TradeSignal[], symbol: string) {
    for (const signal of signals) {
      try {
        Logger.info(`Executing ${signal.type} signal for ${symbol}`, {
          stopLoss: signal.stopLoss,
          takeProfit: signal.takeProfit
        });
        // Implement actual trade execution logic here
      } catch (error) {
        Logger.error('Error executing trade:', error);
      }
    }
  }

  private sleep(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
}