# Smart AI Surveillance System

Custom trained YOLOv8 (best.pt) + Deep SORT + rule-based threat analysis with FastAPI backend and React (Vite + Tailwind + shadcn-style primitives + Framer Motion) dashboard.

## Features
- Object detection using your custom YOLO model (best.pt) with 15 classes
- Multi-object tracking via Deep SORT (fallback lightweight centroid tracker if Deep SORT unavailable)
- Rule engine alerts:
  - Abandoned Bag (>30s unattended)  (includes custom labels: school bag, wrong bag)
  - Crowd Surge (people count threshold)
  - Loitering (person idle >120s)
- Real-time MJPEG video streaming endpoint `/video_feed`
- WebSocket instant alerts `/ws/alerts`
- Persistent alert log (SQLite) + REST endpoints `/alerts`, `/recent_alerts`
- React dashboard: live feed + animated alerts feed + metrics cards
- Standalone inference script for webcam / image / video batch (`backend/run_inference.py`)

## Custom Model
Place your trained weights file `best.pt` at the project root (already referenced by default). The code does NOT auto-download any pretrained weights; it only loads the specified `best.pt` (override via `YOLO_MODEL_NAME` env var if needed).

Class index mapping used (CUSTOM_CLASS_NAMES):
```
0: formal beard
1: formal hair
2: formal id card
3: formal shoes
4: formal tuck in
5: in
6: informal beard
7: informal hair
8: informal id card
9: informal shoes
10: informal tuck in
11: wrong bag
12: school bag
13: person
14: knife
```

## Backend Quick Start
```bash
cd backend
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# CMD: .venv\Scripts\activate.bat
pip install -r requirements.txt
set DEBUG=true
# (Optional) set CAMERA_SOURCE index or RTSP URL
# set CAMERA_SOURCE=0
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```
Open: http://localhost:8000/docs

### Test Inject Alert (DEBUG)
```bash
curl -X POST http://localhost:8000/debug/alert -H "Content-Type: application/json" ^
  -d "{\"type\":\"Test Alert\",\"description\":\"Manual injection\",\"severity\":\"low\"}"
```

## Standalone Inference Script
Use this for ad-hoc inference (webcam / images / videos) and automatic saving of annotated outputs.
```bash
cd backend
python run_inference.py --source 0                 # webcam
python run_inference.py --source ../samples/img.jpg
python run_inference.py --source ../samples/images_folder
python run_inference.py --source ../samples/video.mp4
```
Options (see `--help`):
- `--model PATH` (defaults to ../best.pt)
- `--conf 0.25` confidence threshold
- `--imgsz 640` inference size
- `--show` display annotated window
- `--save-txt` export YOLO txt labels
- `--output runs/outputs` base dir (timestamp subfolder created)

Results:
- Images saved with annotations
- Videos saved as `annotated.mp4`
- Optional `labels/` with normalized txt predictions

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
| YOLO_MODEL_NAME | ../best.pt | Path to custom trained model weights |
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
- Custom class names injected after model load to guarantee correct labeling

## Roadmap / Enhancements
- Multi-camera support (pool processors + camera registry)
- Authentication & RBAC
- Alert severity tuning & suppression windows
- Export / archive alerts (CSV, JSON)
- Docker compose packaging
- GPU inference toggle (Torch CUDA)

## License
MIT
