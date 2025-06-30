import logging
from typing import Any, Dict, Optional
from functools import wraps
import time

# Özel Exception Sınıfları
class BinanceAPIError(Exception):
    """Binance API ile ilgili hatalar için özel exception"""
    pass

class ValidationError(Exception):
    """Veri doğrulama hataları için özel exception"""
    pass

class RiskLimitError(Exception):
    """Risk limiti aşıldığında fırlatılan exception"""
    pass

# Decorator'lar
def retry_on_error(max_retries: int = 3, delay: int = 5):
    """Hata durumunda belirli sayıda tekrar deneyen decorator"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    logging.warning(f"{func.__name__} hatası (deneme {attempt + 1}/{max_retries}): {str(e)}")
                    time.sleep(delay)
            return None
        return wrapper
    return decorator

def validate_params(func):
    """Fonksiyon parametrelerini doğrulayan decorator"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        # İlk parametre self ise atla
        params = list(args[1:]) if len(args) > 0 and hasattr(args[0], '__class__') else list(args)
        
        # None değer kontrolü
        for i, param in enumerate(params):
            if param is None:
                raise ValidationError(f"{func.__name__} fonksiyonunda parametre {i + 1} None olamaz")
        
        return func(*args, **kwargs)
    return wrapper

# Yardımcı Fonksiyonlar
def format_number(number: float, decimals: int = 8) -> str:
    """Float sayıları belirli ondalık basamakla formatla"""
    return f"{number:.{decimals}f}"

def validate_symbol(symbol: str) -> bool:
    """Sembol formatını doğrula"""
    if not isinstance(symbol, str):
        return False
    if '/' not in symbol:
        return False
    base, quote = symbol.split('/')
    return bool(base and quote)

def calculate_percentage_change(old_value: float, new_value: float) -> float:
    """Yüzde değişimi hesapla"""
    if old_value == 0:
        return 0
    return ((new_value - old_value) / old_value) * 100

def safe_divide(a: float, b: float, default: float = 0.0) -> float:
    """Güvenli bölme işlemi"""
    try:
        return a / b if b != 0 else default
    except:
        return default

def validate_api_response(response: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """API yanıtını doğrula"""
    if not isinstance(response, dict):
        raise ValidationError("API yanıtı dictionary değil")
    
    required_fields = ['symbol', 'price', 'quantity']
    for field in required_fields:
        if field not in response:
            raise ValidationError(f"API yanıtında {field} alanı eksik")
    
    return response

# Loglama Yardımcıları
def setup_logger(name: str, log_file: str, level=logging.INFO) -> logging.Logger:
    """Özel logger oluştur"""
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    handler = logging.FileHandler(log_file, encoding='utf-8')
    handler.setFormatter(formatter)
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    
    return logger

def log_trade(logger: logging.Logger, trade_info: Dict[str, Any]):
    """İşlem bilgilerini logla"""
    logger.info(
        "İşlem Gerçekleşti:\n"
        f"Sembol: {trade_info.get('symbol')}\n"
        f"Yön: {trade_info.get('side')}\n"
        f"Fiyat: {trade_info.get('price')}\n"
        f"Miktar: {trade_info.get('quantity')}\n"
        f"Toplam: {trade_info.get('total')}"
    )
