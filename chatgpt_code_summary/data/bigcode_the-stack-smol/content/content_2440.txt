import os
from pathlib import Path

DB_NAME = "chatapp.db"
PROJECT_PATH = Path(__file__).parents[1]
DB_PATH = os.path.join(PROJECT_PATH, "resource", DB_NAME)

PORT_MIN = 1024
PORT_MAX = 65535

DEBUG = os.getenv("CHAT_APP_DEBUG", False)

if DEBUG:
    TIMEOUT = 30
else:
    TIMEOUT = 0.5
