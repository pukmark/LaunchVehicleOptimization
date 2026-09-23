"""Extract Starlink broadcast telemetry with resolution-aware crops and OCR."""

import argparse
import re
import subprocess
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytesseract


# ========== USER SETTINGS ==========
VIDEO_URL = "https://x.com/i/broadcasts/1nxnRBXEladxO"  # X (Twitter) stream
OUTPUT_DIR = Path("falcon9_telemetry")
VIDEO_FILENAME = OUTPUT_DIR / "falcon9_starlink_2026.mp4"

# How many frames per second to sample (effective)
SAMPLE_FPS = 1.0  # samples per second

# Coordinates measured on a 3840x2160 frame, scaled to the input resolution.
# A different broadcast layout still needs manual tuning.
ROI_REFERENCE_SIZE = (3840, 2160)  # width, height
# Coordinates: [y1:y2, x1:x2]
# Start with something like lower-left box; adjust after preview (see function show_sample_frame)
TELEMETRY_ROI_PHASE1 = {
    # Bottom-left widget: SPEED only (km/h).
    "speed": {"yx1": (647*3, 60*3), "yx2": (671*3, 140*3)},   # (y1, x1), (y2, x2) -- tune via show_sample_frame
    # Dedicated altitude widget (km).
    "altitude": {"yx1": (647*3, 172*3), "yx2": (671*3, 254*3)},  # tune visually
    # Bottom-right widget: ACCELERATION only (g).
    "acceleration": {"yx1": (647*3, 1029*3), "yx2": (671*3, 1105*3)},  # tune visually
    # Bottom-center timer: T-/T+ hh:mm:ss. Sized to avoid the subtitle line below.
    "time":  {"yx1": (646*3, 580*3), "yx2": (672*3, 725*3)},  # tune visually
}

# Phase 2 ROIs (after START_SEP_SEC). Tune these visually.
TELEMETRY_ROI_PHASE2 = {
    "speed": {"yx1": (647*3, 60*3), "yx2": (671*3, 140*3)},   # speed widget unchanged
    "altitude": {"yx1": (647*3, 172*3), "yx2": (671*3, 254*3)},  # stage 1 altitude (if present)
    "velocity2": {"yx1": (647*3, 1029*3), "yx2": (671*3, 1105*3)},  # acceleration widget now shows stage 2 velocity
    "altitude2": {"yx1": (647*3, 1138*3), "yx2": (671*3, 1215*3)},    # stage 2 altitude widget
    "time":  {"yx1": (646*3, 580*3), "yx2": (672*3, 725*3)},       # time widget unchanged
}

# ROI preview styling
ROI_BORDER_COLOR = (0, 255, 0)   # BGR
ROI_BORDER_THICKNESS = 2
ROI_LABEL_COLOR = (0, 255, 0)    # BGR
PREVIEW_MAX_WIDTH = 1280
PREVIEW_MAX_HEIGHT = 800

# Basic validity ranges for parsed telemetry (used to filter out OCR noise).
VEL_KMH_MIN, VEL_KMH_MAX = 0, 30000   # 0 to ~8.3 km/s
ALT_KM_MIN, ALT_KM_MAX = 0, 400       # 0 to ~400 km
ACC_G_MIN, ACC_G_MAX = -10, 10          # reasonable g-range

# Restrict OCR to digits and some punctuation to improve accuracy
# General config for telemetry gauges (speed/altitude).
TESSERACT_CONFIG = r"--psm 7 -c tessedit_char_whitelist=0123456789,+-.km/hKM/gG"
OCR_TIMEOUT_SEC = 10
# Dedicated config for the center timer (cleaner whitelist: digits, colon, T, +, -).
TIME_TESSERACT_CONFIG = r"--psm 7 -c tessedit_char_whitelist=0123456789:Tt:+-"

# Optional: start processing at an offset into the video (seconds).
# Set to None to process from the beginning.
START_TIME_SEC = 10*60+4  # skip the prelaunch portion of this broadcast
# Optional: stop processing at an offset into the video (seconds).
# Set to None to process through the end.
END_TIME_SEC = START_TIME_SEC + 8*60+48
START_SEP_SEC = START_TIME_SEC + 2*60+33  # set to seconds when second stage telemetry begins
# If True, display each sampled frame with ROI overlays and cropped ROIs for debugging.
DEBUG_SHOW_SAMPLES = False
DEBUG_SHOW_DELAY_MS = 1  # delay passed to cv2.waitKey when showing debug windows

# ========== UTILITIES ==========

def run_cmd(cmd_list):
    print(">>", " ".join(cmd_list))
    subprocess.run(cmd_list, check=True)


def ensure_output_dir():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def download_video():
    """
    Download the video using yt-dlp (supports X/Twitter).
    """
    if VIDEO_FILENAME.exists():
        print(f"[INFO] Video already exists: {VIDEO_FILENAME}")
        return

    VIDEO_FILENAME.parent.mkdir(parents=True, exist_ok=True)
    # Use best mp4 format
    cmd = [
        "yt-dlp",
        "-f", "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/mp4",
        "--merge-output-format", "mp4",
        "-o", str(VIDEO_FILENAME),
        VIDEO_URL,
    ]
    run_cmd(cmd)
    print(f"[INFO] Downloaded video to {VIDEO_FILENAME}")


def scaled_rois(frame_shape, rois):
    """Scale reference crops, rejecting empty or out-of-frame rectangles."""
    height, width = frame_shape[:2]
    ref_width, ref_height = ROI_REFERENCE_SIZE
    if min(height, width, ref_width, ref_height) <= 0:
        raise ValueError("Frame and ROI reference dimensions must be positive.")
    scaled = {}
    for name, roi in rois.items():
        y1, x1 = roi["yx1"]
        y2, x2 = roi["yx2"]
        if not (0 <= x1 < x2 <= ref_width and 0 <= y1 < y2 <= ref_height):
            raise ValueError(f"ROI {name!r} is outside the reference frame: {roi}")
        x1, x2 = round(x1 * width / ref_width), round(x2 * width / ref_width)
        y1, y2 = round(y1 * height / ref_height), round(y2 * height / ref_height)
        if x1 >= x2 or y1 >= y2:
            raise ValueError(f"ROI {name!r} is empty at {width}x{height} resolution.")
        scaled[name] = {"yx1": (y1, x1), "yx2": (y2, x2)}
    return scaled


def rois_at_time(frame_shape, time_sec, separation_sec):
    rois = (TELEMETRY_ROI_PHASE2 if separation_sec is not None and time_sec >= separation_sec
            else TELEMETRY_ROI_PHASE1)
    return scaled_rois(frame_shape, rois)


def show_sample_frame(start_time_sec=None, *, video_path=None, separation_sec=START_SEP_SEC):
    """Preview scaled crop rectangles for the selected video time."""
    video_path = VIDEO_FILENAME if video_path is None else Path(video_path)
    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")
        if start_time_sec is not None:
            if not np.isfinite(start_time_sec) or start_time_sec < 0:
                raise ValueError("Preview start time must be finite and nonnegative.")
            cap.set(cv2.CAP_PROP_POS_MSEC, start_time_sec * 1000)
        ok, frame = cap.read()
        if not ok or frame is None:
            raise RuntimeError("Could not read a frame at the requested preview time.")
    finally:
        cap.release()

    h, w = frame.shape[:2]
    print(f"[INFO] Frame resolution: {w}x{h}")
    rois = rois_at_time(frame.shape, start_time_sec or 0, separation_sec)
    for name, roi in rois.items():
        y1, x1 = roi["yx1"]
        y2, x2 = roi["yx2"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), ROI_BORDER_COLOR, ROI_BORDER_THICKNESS)
        cv2.putText(frame, name, (x1, max(y1 - 10, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, ROI_LABEL_COLOR, 2)
    scale = min(PREVIEW_MAX_WIDTH / w, PREVIEW_MAX_HEIGHT / h, 1.0)
    if scale < 1:
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    try:
        cv2.imshow("Sample frame with ROIs", frame)
        print("[INFO] Press any key in the image window to close.")
        cv2.waitKey(0)
    finally:
        cv2.destroyAllWindows()


def preprocess_roi(roi_img, invert=False, use_otsu=False):
    """
    Basic preprocessing to help OCR:
    - convert to grayscale
    - resize
    - blur slightly
    - threshold
    - optional inversion (useful when text is light on dark)
    - optional Otsu binarization (sometimes better for clean, high-contrast text)
    """
    if roi_img is None or roi_img.size == 0:
        raise ValueError("Cannot OCR an empty crop; check the ROI coordinates.")
    gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
    # Upscale for better OCR resolution
    scale_factor = 2.0
    gray = cv2.resize(
        gray,
        None,
        fx=scale_factor,
        fy=scale_factor,
        interpolation=cv2.INTER_CUBIC,
    )
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    # Adaptive threshold
    if use_otsu:
        _, thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        thr = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,
            2,
        )
    if invert:
        thr = cv2.bitwise_not(thr)
    return thr


def ocr_telemetry(roi_img, config=TESSERACT_CONFIG, invert=False, use_otsu=False):
    """
    Run Tesseract on a telemetry ROI image and return raw text.
    """
    processed = preprocess_roi(roi_img, invert=invert, use_otsu=use_otsu)
    try:
        text = pytesseract.image_to_string(processed, config=config, timeout=OCR_TIMEOUT_SEC)
    except RuntimeError as exc:
        if "timeout" not in str(exc).lower():
            raise
        print("[WARN] OCR timed out; trying another preprocessing variant.")
        return ""
    # Clean up some typical artifacts
    text = text.replace("\n", " ").replace("\r", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def ocr_time_robust(roi_img):
    """Prefer a valid clock reading over longer OCR noise."""
    attempts = []
    for use_otsu in (True, False):
        text = ocr_telemetry(roi_img, config=TIME_TESSERACT_CONFIG,
                             invert=True, use_otsu=use_otsu)
        attempts.append(text)
        if parse_telemetry_text(text, origin="time")["time_str"] is not None:
            return text
    return attempts[0]


def _candidate_value(text, origin):
    parsed = sanitize_parsed(parse_telemetry_text(text, origin=origin))
    key = {"speed": "velocity", "velocity2": "velocity",
           "altitude": "altitude", "altitude2": "altitude",
           "acceleration": "accel_g"}.get(origin, "velocity")
    return parsed[key]


def _best_numeric_candidate(candidates, origin="speed", families=None):
    """Vote across preprocessing families, not repeated inversions of one mask.

    Ties keep the first valid candidate (Otsu in the extraction pipeline).
    """
    if families is None:
        families = range(len(candidates))
    votes = {}
    valid = []
    for text, family in zip(candidates, families):
        value = _candidate_value(text, origin)
        if value is not None:
            valid.append((text, value))
            votes.setdefault(value, set()).add(family)
    if not valid:
        return ""
    return max(valid, key=lambda item: len(votes[item[1]]))[0]


def sanitize_parsed(parsed):
    """Drop values that fall outside reasonable ranges."""
    cleaned = dict(parsed)
    v = cleaned.get("velocity")
    if v is not None and not (VEL_KMH_MIN <= v <= VEL_KMH_MAX):
        cleaned["velocity"] = None
    a = cleaned.get("altitude")
    if a is not None and not (ALT_KM_MIN <= a <= ALT_KM_MAX):
        cleaned["altitude"] = None
    g = cleaned.get("accel_g")
    if g is not None and not (ACC_G_MIN <= g <= ACC_G_MAX):
        cleaned["accel_g"] = None
    return cleaned


def filter_series_outliers(s, window=5, thresh=5.0):
    """Hampel filter with deviations measured against each window's median.

    Only full windows of finite readings are filtered, preserving short runs
    and endpoints instead of classifying the ends of a ramp as OCR spikes.
    """
    if window < 3 or window % 2 == 0 or thresh <= 0:
        raise ValueError("Use an odd window >= 3 and a positive outlier threshold.")
    s = pd.to_numeric(s, errors="coerce").astype(float)
    s = s.replace([np.inf, -np.inf], np.nan)
    rolling = s.rolling(window, center=True, min_periods=window)
    med = rolling.median()
    mad = rolling.apply(lambda values: np.median(np.abs(values - np.median(values))), raw=True)
    tolerance = thresh * 1.4826 * mad + 1e-6
    return s.mask((s - med).abs() > tolerance)


def ocr_gauge_robust(roi_img, origin="speed"):
    """Choose valid, agreeing gauge readings, with extra passes on disagreement."""
    candidates = []
    for invert in (True, False):
        for use_otsu in (True, False):
            candidates.append(ocr_telemetry(roi_img, config=TESSERACT_CONFIG,
                                             invert=invert, use_otsu=use_otsu))
            if len(candidates) == 2:
                values = [_candidate_value(text, origin) for text in candidates]
                if values[0] is not None and values[0] == values[1]:
                    return candidates[0]
    return _best_numeric_candidate(candidates, origin, families=["otsu", "adaptive"] * 2)


def show_debug(frame, roi_images, roi_texts, parsed_data, frame_idx, rois):
    """Display the current sampled frame with ROI boxes, cropped ROIs, and extracted text."""
    frame_vis = frame.copy()
    for name, roi in rois.items():
        (y1, x1) = roi["yx1"]
        (y2, x2) = roi["yx2"]
        cv2.rectangle(frame_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame_vis, name, (x1, max(y1 - 10, 0)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
        )
    cv2.putText(
        frame_vis, f"Frame {frame_idx}", (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2
    )

    # Overlay parsed telemetry summary on the frame.
    overlay_lines = []
    time_primary = parsed_data.get("time", {}).get("time_str")
    if time_primary:
        overlay_lines.append(f"Time: {time_primary}")
    if parsed_data.get("speed", {}).get("velocity") is not None:
        overlay_lines.append(
            f"Speed (km/h): {parsed_data.get('speed', {}).get('velocity')}"
        )
    if parsed_data.get("altitude", {}).get("altitude") is not None:
        overlay_lines.append(
            f"Altitude (km): {parsed_data.get('altitude', {}).get('altitude')}"
        )
    if parsed_data.get("acceleration", {}).get("accel_g") is not None:
        overlay_lines.append(
            f"Accel (g): {parsed_data.get('acceleration', {}).get('accel_g')}"
        )
    y_text = 70
    for line in overlay_lines:
        cv2.putText(
            frame_vis, line, (20, y_text),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2
        )
        y_text += 25

    # Resize preview to fit screen if needed.
    h, w = frame_vis.shape[:2]
    scale = min(PREVIEW_MAX_WIDTH / w, PREVIEW_MAX_HEIGHT / h, 1.0)
    display_frame = frame_vis
    if scale < 1.0:
        new_w, new_h = int(w * scale), int(h * scale)
        display_frame = cv2.resize(frame_vis, (new_w, new_h), interpolation=cv2.INTER_AREA)
        print(f"[DEBUG] Preview resized to {new_w}x{new_h} for display.")

    cv2.imshow("Sampled frame", display_frame)
    for name, img in roi_images.items():
        cv2.imshow(f"ROI - {name}", img)

    # Also show raw OCR text for each ROI in its own window (white image with black text).
    for name, text in roi_texts.items():
        canvas = 255 * np.ones((120, 400, 3), dtype=np.uint8)
        cv2.putText(
            canvas, text[:200], (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1
        )
        cv2.imshow(f"OCR - {name}", canvas)

    key = cv2.waitKey(DEBUG_SHOW_DELAY_MS) & 0xFF
    # Press 'q' to stop debugging early.
    return key == ord('q')


# ========== PARSING TELEMETRY TEXT ==========

TIME_REGEX = re.compile(
    r"(?:(T)\s*([+-])\s*)?(\d{1,2})\s*:\s*([0-5]\d)\s*:\s*([0-5]\d)",
    re.IGNORECASE,
)
NUMBER_PATTERN = r"[+-]?(?:\d{1,3}(?:[ ,]\d{3})+|\d+)(?:\.\d+)?"
GAUGE_UNITS = {"velocity": r"(?:km\s*/\s*h)?", "altitude": r"(?:km)?", "accel_g": r"g?"}


def parse_telemetry_text(text, origin=None):
    """Parse a whole gauge value or HH:MM:SS timer; reject ambiguous noise.

    Gauge values accept signs, arbitrary decimal precision and thousands
    separators. Stage 2 uses the same units as the corresponding stage 1 ROI.
    """
    result = {"time_str": None, "velocity": None, "altitude": None, "accel_g": None}
    text = text.strip().replace("\u2212", "-")
    if origin in (None, "", "time"):
        match = TIME_REGEX.fullmatch(text)
        if match:
            prefix = f"T{match[2]}" if match[1] else ""
            result["time_str"] = f"{prefix}{int(match[3]):02d}:{match[4]}:{match[5]}"
            return result
    fields = {"speed": ("velocity",), "velocity2": ("velocity",),
              "altitude": ("altitude",), "altitude2": ("altitude",),
              "acceleration": ("accel_g",)}
    keys = fields.get(origin, ()) if origin else GAUGE_UNITS
    for key in keys:
        match = re.fullmatch(rf"({NUMBER_PATTERN})\s*{GAUGE_UNITS[key]}", text, re.IGNORECASE)
        if match:
            result[key] = float(match[1].replace(",", "").replace(" ", ""))
    return result


# ========== MAIN EXTRACTION LOOP ==========

METRIC_SOURCES = {
    "speed_kmh": ("speed", "velocity"),
    "altitude_km": ("altitude", "altitude"),
    "acceleration_g": ("acceleration", "accel_g"),
    "velocity2_kmh": ("velocity2", "velocity"),
    "altitude2_km": ("altitude2", "altitude"),
}
CSV_COLUMNS = ["t_video_sec", "time_str_primary", *METRIC_SOURCES]


def clean_telemetry_rows(rows):
    """Filter each output metric once so rejected stage 2 values stay missing."""
    df = pd.DataFrame(rows, columns=CSV_COLUMNS)
    for col in METRIC_SOURCES:
        df[col] = filter_series_outliers(df[col])
    return df.dropna(subset=list(METRIC_SOURCES), how="all").reset_index(drop=True)


def extract_telemetry_to_csv(start_time_sec=None, end_time_sec=None, *,
                             video_path=None, output_dir=None, sample_fps=None,
                             separation_sec=START_SEP_SEC, debug=None, save_raw=False):
    """Extract an inclusive video segment, preserving the existing CSV schema.

    Sample times are anchored to the requested start frame without accumulating
    frame-rounding drift. GUI windows are opt-in. Raw OCR can be saved separately
    for diagnosis; no interpolation is applied to missing measurements.
    """
    video_path = VIDEO_FILENAME if video_path is None else Path(video_path)
    output_dir = OUTPUT_DIR if output_dir is None else Path(output_dir)
    sample_fps = SAMPLE_FPS if sample_fps is None else sample_fps
    debug = DEBUG_SHOW_SAMPLES if debug is None else debug
    start_time_sec = 0.0 if start_time_sec is None else start_time_sec
    if not np.isfinite(sample_fps) or sample_fps <= 0:
        raise ValueError("Sample FPS must be finite and greater than zero.")
    for label, value in (("Start", start_time_sec), ("End", end_time_sec),
                         ("Separation", separation_sec)):
        if value is not None and (not np.isfinite(value) or value < 0):
            raise ValueError(f"{label} time must be finite and nonnegative.")
    if end_time_sec is not None and end_time_sec < start_time_sec:
        raise ValueError("End time is before start time.")

    cap = cv2.VideoCapture(str(video_path))
    rows = []
    try:
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError(f"Video reports an invalid FPS: {fps}")
        count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        total_frames = int(count) if np.isfinite(count) and count > 0 else None
        # Start on the first frame at or after the requested timestamp.
        start_frame = int(np.ceil(start_time_sec * fps))
        end_frame = int(np.floor(end_time_sec * fps)) if end_time_sec is not None else None
        if total_frames is not None:
            if start_frame >= total_frames:
                raise ValueError("Start time is beyond the end of the video.")
            end_frame = min(end_frame, total_frames - 1) if end_frame is not None else total_frames - 1
        if start_frame and not cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame):
            raise RuntimeError("Could not seek to the requested start frame.")
        effective_fps = min(sample_fps, fps)
        print(f"[INFO] Video FPS: {fps:.3f}; sampling at {effective_fps:.3f} Hz")
        frame_idx = next_sample = start_frame
        sample_idx = 0
        while end_frame is None or frame_idx <= end_frame:
            # Avoid converting/copying skipped frames into Python arrays.
            if not cap.grab():
                break
            if frame_idx == next_sample:
                ok, frame = cap.retrieve()
                if not ok or frame is None:
                    raise RuntimeError(f"Could not decode sampled frame {frame_idx}.")
                time_sec = frame_idx / fps
                current_rois = rois_at_time(frame.shape, time_sec, separation_sec)
                roi_images, roi_texts, parsed_data = {}, {}, {}
                for name, roi in current_rois.items():
                    y1, x1 = roi["yx1"]
                    y2, x2 = roi["yx2"]
                    roi_img = frame[y1:y2, x1:x2]
                    roi_images[name] = roi_img
                    text = (ocr_time_robust(roi_img) if name == "time"
                            else ocr_gauge_robust(roi_img, origin=name))
                    roi_texts[name] = text
                    parsed_data[name] = sanitize_parsed(parse_telemetry_text(text, origin=name))
                row = {"frame_idx": frame_idx, "t_video_sec": time_sec,
                       "time_str_primary": parsed_data.get("time", {}).get("time_str")}
                for column, (origin, key) in METRIC_SOURCES.items():
                    row[column] = parsed_data.get(origin, {}).get(key)
                row.update({f"raw_{name}": text for name, text in roi_texts.items()})
                rows.append(row)
                sample_idx += 1
                next_sample = start_frame + round(sample_idx * fps / effective_fps)
                if sample_idx % 50 == 0:
                    print(f"[INFO] Processed {sample_idx} samples (video {time_sec:.1f}s)")
                if debug and show_debug(frame, roi_images, roi_texts, parsed_data, frame_idx, current_rois):
                    print("[INFO] Stopped by user; saving the extracted samples.")
                    break
            frame_idx += 1
    finally:
        cap.release()
        if debug:
            cv2.destroyAllWindows()

    if not rows:
        raise ValueError("No frames sampled; check the video and requested time range.")
    df = clean_telemetry_rows(rows)
    if save_raw:
        output_dir.mkdir(parents=True, exist_ok=True)
        raw_path = output_dir / "falcon9_telemetry_starlink_raw.csv"
        pd.DataFrame(rows).to_csv(raw_path, index=False)
        print(f"[INFO] Saved raw OCR to: {raw_path}")
    if df.empty:
        raise RuntimeError("No valid telemetry found; preview the crops and check the broadcast layout.")
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "falcon9_telemetry_starlink.csv"
    df.to_csv(csv_path, index=False)
    print(f"[INFO] Saved {len(df)} telemetry rows to: {csv_path}")
    return df


def plot_extracted_data(df, *, output_dir=None, show=True):
    """
    Plot altitude, speed, and acceleration over video time.
    Saves a PNG next to the CSV.
    """
    if df is None or df.empty:
        print("[INFO] No data to plot.")
        return

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    t = df["t_video_sec"] - df["t_video_sec"].min()

    def _plot(ax, y_col, label, color):
        mask = t.notna() & df[y_col].notna()
        if mask.any():
            ax.plot(t[mask], df[y_col][mask],'-o', label=label, color=color)
            return True
        return False

    # Speed (Stage 1) and Stage 2 velocity
    ax_speed = axes[0]
    speed_plotted = []
    speed_plotted.append(_plot(ax_speed, "speed_kmh", "Speed (km/h)", "tab:green"))
    speed_plotted.append(_plot(ax_speed, "velocity2_kmh", "Stage 2 Velocity (km/h)", "tab:olive"))
    ax_speed.set_ylabel("Speed (km/h)")
    if any(speed_plotted):
        ax_speed.legend()
    ax_speed.grid(True, alpha=0.3)

    # Altitude (Stage 1) and Stage 2 altitude
    ax_alt = axes[1]
    alt_plotted = []
    alt_plotted.append(_plot(ax_alt, "altitude_km", "Altitude (km)", "tab:purple"))
    alt_plotted.append(_plot(ax_alt, "altitude2_km", "Stage 2 Altitude (km)", "tab:pink"))
    ax_alt.set_ylabel("Altitude (km)")
    if any(alt_plotted):
        ax_alt.legend()
    ax_alt.grid(True, alpha=0.3)

    # Acceleration
    ax_acc = axes[2]
    acc_plotted = _plot(ax_acc, "acceleration_g", "Acceleration (g)", "tab:red")
    if acc_plotted:
        ax_acc.legend()
    ax_acc.set_ylabel("Acceleration (g)")
    ax_acc.set_xlabel("Time since first extracted sample (s)")
    ax_acc.grid(True, alpha=0.3)

    plt.tight_layout()
    output_dir = OUTPUT_DIR if output_dir is None else Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / "falcon9_telemetry_starlink.png"
    fig.savefig(plot_path, dpi=150)
    print(f"[INFO] Saved plot to: {plot_path}")
    if show:
        plt.show()
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, help="Local video; skips downloading")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--start", type=float, default=START_TIME_SEC, help="Video start time in seconds")
    parser.add_argument("--end", type=float, default=END_TIME_SEC, help="Video end time in seconds")
    parser.add_argument("--separation", type=float, default=START_SEP_SEC, help="Phase 2 video time in seconds")
    parser.add_argument("--sample-fps", type=float, default=SAMPLE_FPS)
    parser.add_argument("--debug", action="store_true", help="Show OCR windows; q stops extraction")
    parser.add_argument("--preview", action="store_true", help="Preview crop rectangles and exit")
    parser.add_argument("--save-raw", action="store_true", help="Also save raw OCR readings")
    parser.add_argument("--no-plot", action="store_false", help="Skip plotting for headless runs")
    args = parser.parse_args(argv)
    if args.video is None:
        download_video()
    if args.preview:
        show_sample_frame(args.start, video_path=args.video, separation_sec=args.separation)
        return
    df = extract_telemetry_to_csv(args.start, args.end, video_path=args.video,
                                  output_dir=args.output_dir, sample_fps=args.sample_fps,
                                  separation_sec=args.separation, debug=args.debug, save_raw=args.save_raw)
    if not args.no_plot:
        plot_extracted_data(df, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
