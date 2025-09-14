import os
import cv2
import time
import threading
import traceback
import queue
import json
from typing import Optional, List, Tuple, Dict, Any

import numpy as np

from . import config
from .alert_rules import RuleEngine, PERSON_CLASS_NAME, BAG_CLASS_NAMES

# Attempt to import YOLO and Deep SORT
YOLO_AVAILABLE = True
DEEPSORT_AVAILABLE = True
try:
    from ultralytics import YOLO  # type: ignore
except Exception:  # pragma: no cover
    YOLO_AVAILABLE = False

try:
    from deep_sort_realtime.deepsort_tracker import DeepSort  # type: ignore
except Exception:  # pragma: no cover
    DEEPSORT_AVAILABLE = False

class SimpleTracker:
    """Fallback centroid tracker if Deep SORT not available."""
    def __init__(self, max_distance: float = 80.0, max_age: float = 1.0):
        self.next_id = 1
        self.tracks: Dict[int, Dict[str, Any]] = {}
        self.max_distance = max_distance
        self.max_age = max_age

    def update(self, detections: List[Tuple[int, int, int, int, float, str]]):
        # detections: list (x1,y1,x2,y2,conf,cls_name)
        now = time.time()
        # Mark tracks as unmatched initially
        for tid, tr in self.tracks.items():
            tr['matched'] = False
        for (x1,y1,x2,y2,conf,cls_name) in detections:
            cx = (x1 + x2)/2
            cy = (y1 + y2)/2
            chosen_id = None
            chosen_dist = 1e9
            for tid, tr in self.tracks.items():
                if tr['cls'] != cls_name:
                    continue
                dist = ((cx - tr['cx'])**2 + (cy - tr['cy'])**2)**0.5
                if dist < self.max_distance and dist < chosen_dist:
                    chosen_dist = dist
                    chosen_id = tid
            if chosen_id is None:
                chosen_id = self.next_id
                self.next_id += 1
                self.tracks[chosen_id] = {'cls': cls_name, 'cx': cx, 'cy': cy, 'last_seen': now, 'matched': True,
                                          'box': (x1,y1,x2,y2)}
            else:
                tr = self.tracks[chosen_id]
                tr.update({'cx': cx, 'cy': cy, 'last_seen': now, 'matched': True, 'box': (x1,y1,x2,y2)})
        # Remove stale
        stale = [tid for tid,tr in self.tracks.items() if (now - tr['last_seen']) > self.max_age]
        for tid in stale:
            del self.tracks[tid]
        # Return track-like objects
        out = []
        for tid, tr in self.tracks.items():
            d = {
                'track_id': tid,
                'cls_name': tr['cls'],
                'bbox': tr['box'],
                'centroid': (tr['cx'], tr['cy'])
            }
            out.append(d)
        return out

class VideoProcessor:
    def __init__(self, alert_queue: 'queue.Queue[dict]'):
        self.alert_queue = alert_queue
        self.rule_engine = RuleEngine(self._emit_alert)
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.model = None
        self.class_names: Dict[int, str] = {}
        self.tracker = None
        self.frame_lock = threading.Lock()
        self.latest_frame_bytes: Optional[bytes] = None
        self.frame_index = 0
        self.last_inference_time = 0.0
        self.camera_source = 0 if config.CAMERA_SOURCE == '0' else config.CAMERA_SOURCE
        self.cap: Optional[cv2.VideoCapture] = None
        self.headless = config.HEADLESS_MODE or not YOLO_AVAILABLE
        self._init_components()

    def _init_components(self):
        if YOLO_AVAILABLE:
            try:
                self.model = YOLO(config.YOLO_MODEL_NAME)
                # model.names is dict id->name
                self.class_names = self.model.names
            except Exception:
                traceback.print_exc()
                self.headless = True
        else:
            self.headless = True
        if DEEPSORT_AVAILABLE and not self.headless:
            try:
                self.tracker = DeepSort(max_age=30)
            except Exception:
                traceback.print_exc()
                self.tracker = None
        if self.tracker is None:
            self.tracker = SimpleTracker()

    def start(self):
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
        if self.cap:
            self.cap.release()

    def _emit_alert(self, **kwargs):
        try:
            self.alert_queue.put_nowait(kwargs)
        except queue.Full:
            pass

    def _get_frame(self) -> Optional[np.ndarray]:
        if self.headless:
            # produce synthetic frame
            img = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(img, 'HEADLESS MODE', (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 2)
            cv2.putText(img, time.strftime('%H:%M:%S'), (50, 260), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
            return img
        if self.cap is None:
            self.cap = cv2.VideoCapture(self.camera_source)
            if not self.cap.isOpened():
                self.headless = True
                return self._get_frame()
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame

    def _process_frame(self, frame: np.ndarray):
        detections = []  # list (x1,y1,x2,y2,conf,cls_name)
        if not self.headless and self.model is not None:
            try:
                results = self.model(frame, verbose=False)
                for r in results:
                    if not hasattr(r, 'boxes'):
                        continue
                    for box in r.boxes:
                        cls_id = int(box.cls[0]) if box.cls is not None else -1
                        conf = float(box.conf[0]) if box.conf is not None else 0.0
                        cls_name = self.class_names.get(cls_id, str(cls_id))
                        if cls_name not in {PERSON_CLASS_NAME, *BAG_CLASS_NAMES, 'car', 'truck', 'bus', 'motorbike', 'bicycle'}:
                            continue
                        xyxy = box.xyxy[0].tolist()  # x1,y1,x2,y2
                        x1,y1,x2,y2 = map(int, xyxy)
                        detections.append((x1,y1,x2,y2,conf,cls_name))
            except Exception:
                traceback.print_exc()
        # Track
        tracks_out = []
        if isinstance(self.tracker, SimpleTracker):
            tracks_out = self.tracker.update(detections)
        else:
            # Deep SORT expects list of [ [x1,y1,x2,y2], conf, class_name ]
            ds_dets = [ ((x1,y1,x2,y2), conf, cls_name) for (x1,y1,x2,y2,conf,cls_name) in detections ]
            try:
                deep_tracks = self.tracker.update_tracks(ds_dets, frame=frame)
                for t in deep_tracks:
                    if not t.is_confirmed():
                        continue
                    track_id = t.track_id
                    ltrb = t.to_ltrb()  # left, top, right, bottom
                    x1,y1,x2,y2 = map(int, ltrb)
                    cls_name = t.get_class() or self._infer_class_from_detection((x1,y1,x2,y2), detections)
                    cx = (x1+x2)/2
                    cy = (y1+y2)/2
                    tracks_out.append({
                        'track_id': track_id,
                        'cls_name': cls_name,
                        'bbox': (x1,y1,x2,y2),
                        'centroid': (cx, cy)
                    })
            except Exception:
                traceback.print_exc()
        # Update rule engine
        persons = []
        bags = []
        for tr in tracks_out:
            track_id = str(tr['track_id'])
            cls_name = tr['cls_name']
            (x1,y1,x2,y2) = tr['bbox']
            cx, cy = tr['centroid']
            self.rule_engine.update_track(track_id, cls_name, cx, cy)
            if cls_name == PERSON_CLASS_NAME:
                persons.append((track_id, cx, cy))
            if cls_name in BAG_CLASS_NAMES:
                bags.append((track_id, cx, cy))
        # Associate bags with nearest person within radius
        for bag_id, bcx, bcy in bags:
            for pid, pcx, pcy in persons:
                dist = ((bcx-pcx)**2 + (bcy-pcy)**2)**0.5
                if dist < 150:  # px threshold
                    self.rule_engine.mark_bag_near_person(bag_id)
                    break
        self.rule_engine.evaluate_all()
        # Draw overlays
        for tr in tracks_out:
            (x1,y1,x2,y2) = tr['bbox']
            cls_name = tr['cls_name']
            track_id = tr['track_id']
            color = (0,255,0)
            if cls_name in BAG_CLASS_NAMES:
                color = (0,200,255)
            elif cls_name == PERSON_CLASS_NAME:
                color = (255,0,0)
            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
            label = f"{cls_name}_{track_id}"
            cv2.putText(frame, label, (x1, max(15,y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        # Add status line
        cv2.putText(frame, f"Tracks: {len(tracks_out)}", (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        if self.headless:
            cv2.putText(frame, f"HEADLESS (model unavailable)", (10,45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,200,255), 2)
        return frame

    def _infer_class_from_detection(self, bbox, detections):
        x1,y1,x2,y2 = bbox
        for (dx1,dy1,dx2,dy2,conf,cls_name) in detections:
            iou = self._iou((x1,y1,x2,y2), (dx1,dy1,dx2,dy2))
            if iou > 0.5:
                return cls_name
        return 'object'

    @staticmethod
    def _iou(a,b):
        ax1,ay1,ax2,ay2 = a
        bx1,by1,bx2,by2 = b
        inter_x1 = max(ax1,bx1)
        inter_y1 = max(ay1,by1)
        inter_x2 = min(ax2,bx2)
        inter_y2 = min(ay2,by2)
        if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
            return 0.0
        inter = (inter_x2-inter_x1)*(inter_y2-inter_y1)
        area_a = (ax2-ax1)*(ay2-ay1)
        area_b = (bx2-bx1)*(by2-by1)
        return inter / float(area_a + area_b - inter + 1e-6)

    def _loop(self):
        target_delay = 1.0 / max(1, config.TARGET_FPS)
        while self.running:
            start = time.time()
            frame = self._get_frame()
            if frame is None:
                time.sleep(0.1)
                continue
            self.frame_index += 1
            if self.frame_index % config.INFERENCE_FRAME_SKIP == 0:
                frame = self._process_frame(frame)
            # Encode JPEG
            try:
                ret, jpeg = cv2.imencode('.jpg', frame)
                if ret:
                    with self.frame_lock:
                        self.latest_frame_bytes = jpeg.tobytes()
            except Exception:
                traceback.print_exc()
            elapsed = time.time() - start
            sleep_time = target_delay - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def get_latest_frame(self) -> Optional[bytes]:
        with self.frame_lock:
            return self.latest_frame_bytes

