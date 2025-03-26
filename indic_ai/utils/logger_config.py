import logging
import os
from logging.handlers import TimedRotatingFileHandler

# Create a logs directory if it doesn't exist
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# Define log file path
LOG_FILE_PATH = os.path.join(LOG_DIR, "central_log.txt")

# Configure Timed Rotating Log Handler (resets every 2 days)
log_handler = TimedRotatingFileHandler(
    LOG_FILE_PATH, when="D", interval=2, backupCount=0, encoding="utf-8"
)
log_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(module)s | %(message)s"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Capture all log levels
    handlers=[
        log_handler,       # Log to rotating file
        logging.StreamHandler()  # Log to console
    ],
    force= True
)

# Create logger instance
logger = logging.getLogger("crewai_logger")

# Optional: Completely disable logs from specific libraries
third_party_loggers = [
    "httpcore",
    "httpx",
    "posthog",
    "chromadb",
    "uvicorn",
    "uvicorn.access",
]

for logger_name in third_party_loggers:
    logging.getLogger(logger_name).setLevel(logging.WARNING)  # To suppress excessive API logs

logging.getLogger("litellm").disabled = True

