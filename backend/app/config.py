import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# Camera source (0=default webcam). Can also be rtsp/http URL
CAMERA_SOURCE = os.getenv("CAMERA_SOURCE", "0")

# Alert thresholds
CROWD_COUNT_THRESHOLD = int(os.getenv("CROWD_COUNT_THRESHOLD", 8))
LOITERING_SECONDS = int(os.getenv("LOITERING_SECONDS", 120))
ABANDONED_BAG_SECONDS = int(os.getenv("ABANDONED_BAG_SECONDS", 30))
STILL_MOVEMENT_PX_RADIUS = int(os.getenv("STILL_MOVEMENT_PX_RADIUS", 50))

# DB
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{BASE_DIR / 'surveillance.db'}")

# Model
YOLO_MODEL_NAME = os.getenv("YOLO_MODEL_NAME", "yolov8n.pt")

# Performance
INFERENCE_FRAME_SKIP = int(os.getenv("INFERENCE_FRAME_SKIP", 1))  # process every Nth frame
TARGET_FPS = int(os.getenv("TARGET_FPS", 15))

# Websocket broadcast queue size
ALERT_QUEUE_MAX = int(os.getenv("ALERT_QUEUE_MAX", 100))

# If set, runs without actual camera (generates dummy frames)
HEADLESS_MODE = os.getenv("HEADLESS_MODE", "false").lower() in {"1", "true", "yes"}

# Debug / development mode
DEBUG = os.getenv("DEBUG", "false").lower() in {"1", "true", "yes"}
