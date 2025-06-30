import logging

# Logger yapılandırması
# Bu, hata ve bilgi kayıtları için kullanılacak.
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)
