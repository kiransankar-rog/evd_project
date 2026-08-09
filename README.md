# Emergency Vehicle Detection — Real-Time System

A working implementation of the system described in your Phase-2 project
report: YOLOv8 vehicle detection + ORB reference-image matching + HSV
emergency-light detection + FFT siren-audio verification, wrapped in a
Flask web app with two modes:

- **Live Camera Detection** — real-time webcam stream, frame-by-frame
  detection, boxes + labels drawn live, status panel with FPS/confidence.
- **Upload Video** — full offline pipeline on a traffic clip, including
  audio-based siren detection, matching the Phase-2 report's workflow.

## 1. Setup

```bash
cd evd_project
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

The first run will auto-download `yolov8n.pt` (~6 MB) from Ultralytics'
GitHub releases if it isn't already in `models/`. It's already bundled
here, so nothing to do.

## 2. Replace the placeholder reference images (important)

`reference_images/ambulance.jpg`, `police.jpg`, and `firetruck.jpg` are
**synthetic placeholders** (generated shapes/text) so the app runs out of
the box without crashing. They are NOT real vehicle photos and will not
give you meaningful match accuracy.

For real detection quality, replace each file with a clear, front/side-on
photo of the relevant vehicle type (same filename, any resolution):

- `reference_images/ambulance.jpg`
- `reference_images/police.jpg`
- `reference_images/firetruck.jpg`

Use your own licensed/consented images — a picture of a real local
ambulance or police unit gives far better ORB keypoint matches than a
stock photo shot from an odd angle.

## 3. Run

```bash
python app.py
```

Open **http://127.0.0.1:5000**

- Click **Start Live Feed** to use your webcam (device index 0). If no
  camera is available, the stream shows a placeholder frame instead of
  crashing.
- Click **Upload Video** to run the full pipeline (with audio siren
  check) on a pre-recorded `.mp4`/`.avi` traffic clip.

## 4. How detection works

| Stage | Technique | File |
|---|---|---|
| Vehicle detection | YOLOv8n (COCO: car/bus/truck) | `detector.py` |
| Emergency-type matching | ORB + Brute-Force Hamming matcher vs. reference images | `detector.py` |
| Light detection | HSV thresholding for red/blue flashing patches | `detector.py` |
| Siren detection (video mode only) | FFT band-energy ratio in 500–1800 Hz range | `detector.py` |
| Fusion | Weighted combination of ORB score + light score (+ siren boost) vs. per-class threshold | `detector.py` |

Detected emergency vehicles are boxed in **red** with the class name and
confidence; normal vehicles are boxed in **green**.

## 5. Tuning

In `detector.py`:

- `MATCH_THRESHOLDS` — raise/lower per-class ORB match sensitivity.
- `CLASS_CONSTRAINTS` — which YOLO classes are allowed to match which
  emergency type (prevents a car being scored as a fire truck, etc.).
- `conf_threshold` (constructor arg) — YOLO detection confidence cutoff.
- Siren band `(500, 1800)` Hz and `band_ratio > 0.12` threshold in
  `detect_siren_in_audio` — adjust based on your test clips.

## 6. Known limitations

- ORB matching is lightweight and fast but less robust than a trained
  CNN classifier — accuracy depends heavily on reference image quality
  and lighting/angle similarity to the input footage.
- Siren detection is a simple FFT heuristic, not a trained audio model;
  it can be fooled by other tonal noises (car horns, alarms).
- Live webcam mode does not perform audio siren analysis (no synced
  audio stream from most webcams) — visual detection only.
- Designed for demo/project use, not production traffic-safety
  deployment.

## 7. Project structure

```
evd_project/
├── app.py                 # Flask routes: live stream, upload, status API
├── detector.py             # YOLOv8 + ORB + HSV + FFT detection engine
├── models/yolov8n.pt        # Pretrained YOLOv8 nano weights
├── reference_images/        # ambulance.jpg, police.jpg, firetruck.jpg
├── static/style.css         # Dashboard styling
├── static/output.jpg        # Generated after each video upload
├── templates/
│   ├── index.html           # Landing page
│   ├── live.html            # Live webcam view
│   └── upload.html          # Upload + results view
├── uploads/                 # Uploaded traffic videos land here
└── requirements.txt
```
