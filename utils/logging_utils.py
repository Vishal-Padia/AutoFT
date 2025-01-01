import logging
import os

# Create logs directory if it doesn't exist
if not os.path.exists("logs"):
    os.makedirs("logs")

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/autoft.log"),  # Save logs to a file
        logging.StreamHandler(),  # Print logs to the console
    ],
)

# Create a logger
logger = logging.getLogger("AutoFT")
