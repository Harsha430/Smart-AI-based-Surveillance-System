# Smart AI Surveillance System

Pretrained YOLOv8 + Deep SORT + rule-based threat analysis (no custom training) with FastAPI backend and React (Vite + Tailwind + shadcn-style primitives + Framer Motion) dashboard.

## Features
- Object detection (people, bags, vehicles) using pretrained YOLOv8 (n/s) model
- Multi-object tracking via Deep SORT (fallback lightweight centroid tracker if Deep SORT unavailable)
- Rule engine alerts:
  - Abandoned Bag (>30s unattended)
  - Crowd Surge (people count threshold)
  - Loitering (person idle >120s)
- Real-time MJPEG video streaming endpoint `/video_feed`
- WebSocket instant alerts `/ws/alerts`
- Persistent alert log (SQLite) + REST endpoints `/alerts`, `/recent_alerts`
- React dashboard: live feed + animated alerts feed + metrics cards

## Backend Quick Start
```bash
cd backend
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# CMD: .venv\Scripts\activate.bat
pip install -r requirements.txt
# Headless (no camera / skip model if not present)
set HEADLESS_MODE=true
set DEBUG=true
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```
Open: http://localhost:8000/docs

### Test Inject Alert (DEBUG)
```bash
curl -X POST http://localhost:8000/debug/alert -H "Content-Type: application/json" ^
  -d "{\"type\":\"Test Alert\",\"description\":\"Manual injection\",\"severity\":\"low\"}"
```

## Frontend Quick Start
```bash
cd frontend
npm install
npm run dev
```
Dashboard: http://localhost:5173

## Environment Variables (Backend)
| Name | Default | Description |
|------|---------|-------------|
| CAMERA_SOURCE | 0 | Webcam index or RTSP/HTTP URL |
| YOLO_MODEL_NAME | yolov8n.pt | Pretrained model file (auto-download on first use) |
| HEADLESS_MODE | false | Generate synthetic frames & skip model init |
| DEBUG | false | Enable /debug/alert route |
| CROWD_COUNT_THRESHOLD | 8 | People count for crowd alert |
| LOITERING_SECONDS | 120 | Idle time for loitering |
| ABANDONED_BAG_SECONDS | 30 | Unattended bag threshold |

## Architecture Notes
- Video thread performs capture + inference + tracking, updates latest encoded JPEG in memory (low latency)
- Rule engine stateful per track; updated each frame; emits alerts into queue
- Consumer thread persists alerts & broadcasts over WebSocket
- Frontend consumes WebSocket stream and animates entries (Framer Motion + flash animation)

## Roadmap / Enhancements
- Multi-camera support (pool processors + camera registry)
- Authentication & RBAC
- Alert severity tuning & suppression windows
- Export / archive alerts (CSV, JSON)
- Docker compose packaging
- GPU inference toggle (Torch CUDA)

## License
MIT

