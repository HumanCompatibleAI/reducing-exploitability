import logging
import os

logger = logging
logger.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
