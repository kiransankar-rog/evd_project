"""
Emergency Vehicle Detection System — Real-Time Flask Application
==================================================================
Combines YOLOv50008 object detection, ORB feature matching against reference
images, HSV-based emergency-light detection, and (for uploaded videos)
FFT-based siren audio detection to identify ambulances, police vehicles,
and fire trucks in live camera feeds or uploaded traffic footage.

Run:
    python app.py
Then open http://127.0.0.1: in a browser.

Live detection uses your webcam (device 0). Video upload mode processes
a pre-recorded traffic clip end-to-end, exactly like the Phase-2 report.
"""

import os
import time
import threading
import cv2
import numpy as np
from flask import Flask, render_template, request, Response, jsonify
from ultralytics import YOLO

from detector import EmergencyVehicleDetector

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
STATIC_FOLDER = os.path.join(BASE_DIR, "static")
REF_FOLDER = os.path.join(BASE_DIR, "reference_images")
MODEL_PATH = os.path.join(BASE_DIR, "models", "yolov8n.pt")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# ---------------------------------------------------------------------
# Load model + detector once at startup (shared across requests)
# ---------------------------------------------------------------------
print("[INIT] Loading YOLOv8 model ...")
yolo_model = YOLO(MODEL_PATH)
print("[INIT] Building emergency-vehicle detector (ORB + HSV) ...")
detector = EmergencyVehicleDetector(yolo_model=yolo_model, reference_dir=REF_FOLDER)
print("[INIT] Ready.")

# ---------------------------------------------------------------------
# Shared state for the live camera stream
# ---------------------------------------------------------------------
class LiveState:
    def __init__(self):
        self.lock = threading.Lock()
        self.last_result = "No Emergency Vehicle Detected"
        self.last_class = None
        self.last_confidence = 0
        self.running = False
        self.frame_count = 0
        self.fps = 0.0


live_state = LiveState()


def generate_live_frames(camera_index=0):
    """MJPEG generator: reads webcam frames, runs detection, yields JPEG bytes."""
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        # Fall back message frame if no camera is available in this environment
        blank = np.zeros((360, 640, 3), dtype=np.uint8)
        cv2.putText(blank, "No camera detected on this device", (30, 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        ok, buf = cv2.imencode(".jpg", blank)
        frame_bytes = buf.tobytes()
        while True:
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")
            time.sleep(1)

    live_state.running = True
    prev_time = time.time()

    while live_state.running:
        ret, frame = cap.read()
        if not ret:
            break

        annotated, label, cls_name, conf = detector.process_frame(frame)

        now = time.time()
        fps = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now

        with live_state.lock:
            live_state.last_result = label
            live_state.last_class = cls_name
            live_state.last_confidence = conf
            live_state.frame_count += 1
            live_state.fps = fps

        cv2.putText(annotated, f"FPS: {fps:4.1f}", (10, annotated.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        ok, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            continue
        frame_bytes = buf.tobytes()
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")

    cap.release()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/live")
def live():
    """Live webcam detection page."""
    return render_template("live.html")


@app.route("/video_feed")
def video_feed():
    return Response(generate_live_frames(0), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/live_status")
def live_status():
    with live_state.lock:
        return jsonify({
            "result": live_state.last_result,
            "class": live_state.last_class,
            "confidence": live_state.last_confidence,
            "fps": round(live_state.fps, 1),
            "frames_processed": live_state.frame_count,
        })


@app.route("/stop_live", methods=["POST"])
def stop_live():
    live_state.running = False
    return jsonify({"stopped": True})


@app.route("/upload", methods=["GET", "POST"])
def upload():
    """Batch/offline detection on an uploaded traffic video (with audio siren check)."""
    result = "No Emergency Vehicle Detected"
    accuracy = 0
    counts = {"Ambulance": 0, "Police": 0, "Firetruck": 0}

    if request.method == "POST":
        file = request.files.get("file")
        if file and file.filename:
            video_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(video_path)

            summary = detector.process_video(
                video_path,
                output_image_path=os.path.join(STATIC_FOLDER, "output.jpg"),
                frame_limit=120,
                time_limit_sec=15,
            )
            result = summary["result"]
            accuracy = summary["accuracy"]
            counts = summary["counts"]

    return render_template("upload.html", result=result, accuracy=accuracy, counts=counts)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
