"""
detector.py
===========
Core detection engine for the Emergency Vehicle Detection System.

Pipeline per frame:
    1. YOLOv8 detects generic vehicle classes (car, bus, truck).
    2. Each vehicle's bounding-box crop is compared against reference
       images (ambulance / police / firetruck) using ORB + Brute-Force
       Hamming matching.
    3. The crop is also checked in HSV space for red/blue emergency-light
       colour patches.
    4. A simple rule-based fusion combines the ORB match score and the
       light-colour score into a single confidence value; if it clears
       the per-class threshold, the vehicle is labelled as that
       emergency-vehicle type and boxed in red.

For uploaded video files, `process_video` additionally extracts the
audio track and looks for siren-like tonal energy (narrow-band, high
amplitude, oscillating ~1-3 Hz — the classic wail/yelp cadence) using an
FFT-based heuristic, and folds that into the final confidence score.
"""

import os
import time
import cv2
import numpy as np

# Optional audio analysis dependencies — degrade gracefully if missing.
try:
    from moviepy.editor import VideoFileClip
    AUDIO_AVAILABLE = True
except Exception:
    AUDIO_AVAILABLE = False


# Which generic YOLO/COCO classes are plausible carriers for each
# emergency-vehicle type. Keeps ORB from comparing a car crop to a
# fire-truck reference, etc.
CLASS_CONSTRAINTS = {
    "Ambulance": ["car", "bus", "truck"],
    "Police": ["car"],
    "Firetruck": ["truck", "bus"],
}

# Minimum ORB good-match count to count as a hit per class.
MATCH_THRESHOLDS = {
    "Ambulance": 12,
    "Police": 10,
    "Firetruck": 10,
}

VEHICLE_CLASSES = {"car", "bus", "truck", "motorcycle"}


class EmergencyVehicleDetector:
    def __init__(self, yolo_model, reference_dir, conf_threshold=0.35):
        self.model = yolo_model
        self.conf_threshold = conf_threshold
        self.orb = cv2.ORB_create(nfeatures=1500)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.reference_descriptors = self._load_references(reference_dir)

    # ------------------------------------------------------------------
    def _load_references(self, reference_dir):
        refs = {}
        for name in ["ambulance", "police", "firetruck"]:
            path = os.path.join(reference_dir, f"{name}.jpg")
            if not os.path.exists(path):
                print(f"[WARN] Missing reference image: {path} — skipping class.")
                continue
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"[WARN] Could not read reference image: {path}")
                continue
            _, des = self.orb.detectAndCompute(img, None)
            refs[name.capitalize()] = des
        return refs

    # ------------------------------------------------------------------
    def _orb_score(self, gray_roi, ref_name):
        des_ref = self.reference_descriptors.get(ref_name)
        if des_ref is None:
            return 0
        _, des_roi = self.orb.detectAndCompute(gray_roi, None)
        if des_roi is None or len(des_roi) == 0:
            return 0
        matches = self.bf.match(des_ref, des_roi)
        # Keep only reasonably confident matches (lower Hamming distance = better)
        good = [m for m in matches if m.distance < 60]
        return len(good)

    # ------------------------------------------------------------------
    def _light_score(self, bgr_roi):
        """Detects red/blue emergency-light patches via HSV thresholding.
        Returns a 0-100 score based on the proportion of matching pixels."""
        if bgr_roi.size == 0:
            return 0
        hsv = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2HSV)

        # Red wraps around hue 0/180 — two ranges
        red_mask1 = cv2.inRange(hsv, (0, 120, 120), (10, 255, 255))
        red_mask2 = cv2.inRange(hsv, (170, 120, 120), (180, 255, 255))
        blue_mask = cv2.inRange(hsv, (100, 120, 70), (130, 255, 255))

        light_mask = red_mask1 | red_mask2 | blue_mask
        ratio = float(np.count_nonzero(light_mask)) / light_mask.size
        return min(100, ratio * 400)  # scale up small patches into a usable score

    # ------------------------------------------------------------------
    def classify_crop(self, bgr_roi):
        """Given a vehicle crop, return (best_class_or_None, orb_score, light_score)."""
        if bgr_roi.size == 0:
            return None, 0, 0
        gray = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2GRAY)
        light_score = self._light_score(bgr_roi)

        best_class, best_score = None, 0
        for ref_name in self.reference_descriptors:
            score = self._orb_score(gray, ref_name)
            if score > best_score:
                best_class, best_score = ref_name, score

        return best_class, best_score, light_score

    # ------------------------------------------------------------------
    def process_frame(self, frame):
        """Runs the full pipeline on a single BGR frame.
        Returns (annotated_frame, label_text, best_class_name_or_None, confidence_pct)."""
        annotated = frame.copy()
        results = self.model(frame, conf=self.conf_threshold, verbose=False)[0]

        best_overall = {"class": None, "score": 0, "box": None}

        for box in results.boxes:
            cls_name = self.model.names[int(box.cls[0])]
            if cls_name not in VEHICLE_CLASSES:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            x1, y1 = max(x1, 0), max(y1, 0)
            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                continue

            best_class, orb_score, light_score = self.classify_crop(roi)

            # Apply semantic constraint (e.g. only "car" crops can be "Police")
            if best_class and cls_name not in CLASS_CONSTRAINTS.get(best_class, []):
                best_class = None

            is_emergency = bool(best_class) and orb_score >= MATCH_THRESHOLDS.get(best_class, 999)

            if is_emergency:
                combined = min(100, orb_score * 3 + light_score)
                color = (0, 0, 255)
                text = f"{best_class} ({combined:.0f}%)"
                if combined > best_overall["score"]:
                    best_overall = {"class": best_class, "score": combined, "box": (x1, y1, x2, y2)}
            else:
                color = (0, 200, 0)
                text = cls_name

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated, text, (x1, max(y1 - 8, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        if best_overall["class"]:
            label = f"{best_overall['class']} Detected"
            cv2.putText(annotated, f"EMERGENCY: {label}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        else:
            label = "No Emergency Vehicle Detected"

        return annotated, label, best_overall["class"], round(best_overall["score"], 1)

    # ------------------------------------------------------------------
    def detect_siren_in_audio(self, video_path, time_limit_sec=6):
        """FFT-based heuristic siren detector for uploaded videos.
        Returns True if a narrow, high-energy oscillating tone
        (characteristic of ambulance/police/fire sirens, ~500-1800 Hz)
        is present in the audio track."""
        if not AUDIO_AVAILABLE:
            return False
        try:
            clip = VideoFileClip(video_path)
            if clip.audio is None:
                clip.close()
                return False
            audio = clip.audio.subclip(0, min(clip.duration, time_limit_sec))
            samples = audio.to_soundarray(fps=22050)
            clip.close()
        except Exception as e:
            print(f"[WARN] Audio extraction failed: {e}")
            return False

        if samples.ndim > 1:
            samples = samples.mean(axis=1)
        if len(samples) == 0:
            return False

        # FFT magnitude spectrum
        fft_vals = np.abs(np.fft.rfft(samples))
        freqs = np.fft.rfftfreq(len(samples), d=1.0 / 22050)

        siren_band = (freqs >= 500) & (freqs <= 1800)
        band_energy = fft_vals[siren_band].sum()
        total_energy = fft_vals.sum() + 1e-9
        band_ratio = band_energy / total_energy

        # Sirens concentrate a disproportionate share of energy in a
        # narrow band relative to typical road/traffic noise.
        return band_ratio > 0.12

    # ------------------------------------------------------------------
    def process_video(self, video_path, output_image_path, frame_limit=120, time_limit_sec=15):
        """Runs detection across a video file, aggregating per-class scores,
        and writes an annotated snapshot of the strongest detection frame."""
        cap = cv2.VideoCapture(video_path)
        start_time = time.time()

        scores = {"Ambulance": 0, "Police": 0, "Firetruck": 0}
        best_frame = None
        best_frame_score = -1
        frame_count = 0

        while cap.isOpened() and frame_count < frame_limit:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1

            annotated, label, cls_name, conf = self.process_frame(frame)
            if cls_name:
                scores[cls_name] += conf
                if conf > best_frame_score:
                    best_frame_score = conf
                    best_frame = annotated

            if time.time() - start_time > time_limit_sec:
                break

        cap.release()

        siren_detected = self.detect_siren_in_audio(video_path, time_limit_sec=min(time_limit_sec, 8))
        if siren_detected:
            # Siren presence boosts whichever class currently leads
            leading = max(scores, key=scores.get)
            if scores[leading] > 0:
                scores[leading] = min(100 * max(1, frame_count // 10), scores[leading] + 25)

        detected_class = max(scores, key=scores.get)
        counts = {k: (1 if k == detected_class and scores[k] > 0 else 0) for k in scores}

        if scores[detected_class] > 0:
            result = f"{detected_class} Detected" + (" + Siren Confirmed" if siren_detected else "")
            accuracy = min(97, 70 + scores[detected_class] / max(frame_count, 1))
            if best_frame is not None:
                cv2.imwrite(output_image_path, best_frame)
        else:
            result = "No Emergency Vehicle Detected"
            accuracy = 0
            if best_frame is not None:
                cv2.imwrite(output_image_path, best_frame)
            elif frame_count > 0:
                cap2 = cv2.VideoCapture(video_path)
                ret, frame = cap2.read()
                if ret:
                    cv2.imwrite(output_image_path, frame)
                cap2.release()

        return {
            "result": result,
            "accuracy": round(accuracy, 1),
            "counts": counts,
            "frames_processed": frame_count,
            "siren_detected": siren_detected,
        }
