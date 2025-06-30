from src.api.binance_client import BinanceClient
from src.utils.logger import logger
from src.utils.risk_manager import RiskManager, TradingBotError, InsufficientFundsError, APIError
import os
from dotenv import load_dotenv

load_dotenv()

# Bot konfigürasyonu
api_key = os.getenv('BINANCE_API_KEY')
api_secret = os.getenv('BINANCE_API_SECRET')
max_positions = 5
risk_per_trade = 0.02

# Binance Client ve Risk Manager oluşturma
client = BinanceClient(api_key, api_secret)
risk_manager = RiskManager(max_positions, risk_per_trade)

# Ana işlem döngüsü
async def main():
    try:
        # Canlı işlemler burada yapılacak
        order = await client.execute_trade('UXLINK/USDT', 'buy', 10)
        logger.info(f"Order executed: {order}")
        # Piyasa analizi yapılıyor
        # analysis = await client.analyze_market('UXLINK/USDT')
        # logger.info(f"Market analysis: {analysis}")
    except InsufficientFundsError as e:
        logger.error(f"Insufficient funds: {e}")
    except APIError as e:
        logger.error(f"API error: {e}")
    except TradingBotError as e:
        logger.error(f"Trading bot error: {e}")
    except Exception as e:
        logger.error(f"An error occurred: {e}")

if __name__ == '__main__':
    import asyncio
    asyncio.run(main())
