import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BASE_DIR.parent

# Camera source (0=default webcam). Can also be rtsp/http URL
CAMERA_SOURCE = os.getenv("CAMERA_SOURCE", "0")

# Alert thresholds
CROWD_COUNT_THRESHOLD = int(os.getenv("CROWD_COUNT_THRESHOLD", 8))
LOITERING_SECONDS = int(os.getenv("LOITERING_SECONDS", 120))
ABANDONED_BAG_SECONDS = int(os.getenv("ABANDONED_BAG_SECONDS", 30))
STILL_MOVEMENT_PX_RADIUS = int(os.getenv("STILL_MOVEMENT_PX_RADIUS", 50))

# DB
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{BASE_DIR / 'surveillance.db'}")

# Custom trained YOLO model (default now points to backend/best.pt). Override via env if needed.
_DEFAULT_MODEL_PATH = BASE_DIR / 'best.pt'
YOLO_MODEL_NAME = os.getenv("YOLO_MODEL_NAME", str(_DEFAULT_MODEL_PATH))

# Explicit custom class names for the trained model (index aligned)
CUSTOM_CLASS_NAMES = [
    'formal beard',
    'formal hair',
    'formal id card',
    'formal shoes',
    'formal tuck in',
    'in',
    'informal beard',
    'informal hair',
    'informal id card',
    'informal shoes',
    'informal tuck in',
    'wrong bag',
    'school bag',
    'person',
    'knife'
]

# Performance
INFERENCE_FRAME_SKIP = int(os.getenv("INFERENCE_FRAME_SKIP", 1))  # process every Nth frame
TARGET_FPS = int(os.getenv("TARGET_FPS", 15))

# Websocket broadcast queue size
ALERT_QUEUE_MAX = int(os.getenv("ALERT_QUEUE_MAX", 100))

# If set, runs without actual camera (generates dummy frames)
HEADLESS_MODE = os.getenv("HEADLESS_MODE", "false").lower() in {"1", "true", "yes"}

# Debug / development mode
DEBUG = os.getenv("DEBUG", "false").lower() in {"1", "true", "yes"}
