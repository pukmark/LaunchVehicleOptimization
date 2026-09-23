import os
import re
import subprocess
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytesseract


# ========== USER SETTINGS ==========
YOUTUBE_URL = "https://www.youtube.com/watch?v=AcoFvwsuIQI"  # TODO: put Falcon 9 link here
OUTPUT_DIR = Path("falcon9_telemetry")
VIDEO_FILENAME = OUTPUT_DIR / "falcon9.mp4"

# How many frames per second to sample (effective)
SAMPLE_FPS = 1.0  # e.g. 5 samples per second

# --- Telemetry crop (for 1920x1080 you MUST tune these) ---
# Coordinates: [y1:y2, x1:x2]
# Start with something like lower-left box; adjust after preview (see function show_sample_frame)
TELEMETRY_ROI = {
    # Bottom-left widget: SPEED only (km/h).
    "speed": {"yx1": (970, 96),   "yx2": (1008, 217)},   # (y1, x1), (y2, x2) -- adjust if needed
    # Dedicated altitude widget (km).
    "altitude": {"yx1": (970, 260), "yx2": (1008, 381)},  # (y1, x1), (y2, x2) -- tune visually
    # Bottom-right widget: ACCELERATION only (g).
    "acceleration": {"yx1": (970, 1540), "yx2": (1008, 1660)},  # (y1, x1), (y2, x2)
    # Bottom-center timer: T-/T+ hh:mm:ss. Sized to avoid the subtitle line below.
    "time":  {"yx1": (966, 870),  "yx2": (1013, 1090)},  # (y1, x1), (y2, x2)
}

# ROI preview styling
ROI_BORDER_COLOR = (0, 255, 0)   # BGR
ROI_BORDER_THICKNESS = 2
ROI_LABEL_COLOR = (0, 255, 0)    # BGR

# Basic validity ranges for parsed telemetry (used to filter out OCR noise).
VEL_KMH_MIN, VEL_KMH_MAX = 0, 30000   # 0 to ~8.3 km/s
ALT_KM_MIN, ALT_KM_MAX = 0, 400       # 0 to ~400 km
ACC_G_MIN, ACC_G_MAX = -10, 10          # reasonable g-range

# Restrict OCR to digits and some punctuation to improve accuracy
# General config for telemetry gauges (speed/altitude).
TESSERACT_CONFIG = r"--psm 7 -c tessedit_char_whitelist=0123456789:+-.km/hKM/ "
# Dedicated config for the center timer (cleaner whitelist: digits, colon, T, +, -).
TIME_TESSERACT_CONFIG = r"--psm 7 -c tessedit_char_whitelist=0123456789:Tt:+-"

# Optional: start processing at an offset into the video (seconds).
# Set to None to process from the beginning.
START_TIME_SEC = 365 # e.g. skip first 6 minutes of video
# Optional: stop processing at an offset into the video (seconds).
# Set to None to process through the end.
END_TIME_SEC = START_TIME_SEC + 8*60+40
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
    Download the YouTube video using yt-dlp.
    """
    if VIDEO_FILENAME.exists():
        print(f"[INFO] Video already exists: {VIDEO_FILENAME}")
        return

    ensure_output_dir()
    # Use best mp4 format
    cmd = [
        "yt-dlp",
        "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4",
        "-o", str(VIDEO_FILENAME),
        YOUTUBE_URL,
    ]
    run_cmd(cmd)
    print(f"[INFO] Downloaded video to {VIDEO_FILENAME}")


def show_sample_frame(start_time_sec=None):
    """
    Show a single frame with rectangles where ROIs are cropped, for tuning TELEMETRY_ROI.
    Run once, adjust TELEMETRY_ROI based on visual result, then comment it out.
    """
    cap = cv2.VideoCapture(str(VIDEO_FILENAME))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if start_time_sec is not None and fps > 0:
        start_frame = int(start_time_sec * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        print(f"[INFO] Preview starting at {start_time_sec:.2f}s (frame {start_frame})")

    ok, frame = cap.read()
    cap.release()

    if not ok or frame is None:
        print("[ERROR] Could not read first frame from video.")
        return

    h, w, _ = frame.shape
    print(f"[INFO] Frame resolution: {w}x{h}")

    # Draw ROI rectangles
    for name, roi in TELEMETRY_ROI.items():
        (y1, x1) = roi["yx1"]
        (y2, x2) = roi["yx2"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), ROI_BORDER_COLOR, ROI_BORDER_THICKNESS)
        cv2.putText(
            frame, name, (x1, max(y1 - 10, 0)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, ROI_LABEL_COLOR, 2
        )

    cv2.imshow("Sample frame with ROIs", frame)
    print("[INFO] Press any key in the image window to close.")
    cv2.waitKey(0)
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
    text = pytesseract.image_to_string(processed, config=config)
    # Clean up some typical artifacts
    text = text.replace("\n", " ").replace("\r", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def ocr_time_robust(roi_img):
    """
    Try multiple preprocessing strategies for the center timer and return the best-looking result.
    """
    attempts = []
    for use_otsu in (False, True):
        text = ocr_telemetry(
            roi_img,
            config=TIME_TESSERACT_CONFIG,
            invert=True,       # timer is white on black; invert to black on white
            use_otsu=use_otsu
        )
        attempts.append(text)

    # Pick the longest non-empty candidate; fallback to first.
    best = ""
    for cand in attempts:
        if cand and len(cand) > len(best):
            best = cand
    return best or attempts[0]


def _best_numeric_candidate(candidates):
    """Pick the candidate with the most digits; tie-breaker: longest string."""
    def score(s):
        digits = len(re.findall(r"\d", s))
        return (digits, len(s))
    best = ""
    best_score = (-1, -1)
    for cand in candidates:
        sc = score(cand)
        if sc > best_score:
            best_score = sc
            best = cand
    return best


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
    """
    Drop points that deviate too far from the local median (MAD-based).
    """
    s = s.astype(float)
    med = s.rolling(window, center=True, min_periods=1).median()
    mad = (s - med).abs().rolling(window, center=True, min_periods=1).median()
    tol = thresh * mad + 1e-6
    mask = (s - med).abs() > tol
    s_filtered = s.copy()
    s_filtered[mask] = np.nan
    return s_filtered


def ocr_gauge_robust(roi_img):
    """
    Try multiple preprocessing strategies for gauge readouts (speed/altitude).
    Returns the best numeric-looking text.
    """
    candidates = []
    for invert in (False, True):
        for use_otsu in (False, True):
            txt = ocr_telemetry(
                roi_img,
                config=TESSERACT_CONFIG,
                invert=invert,
                use_otsu=use_otsu,
            )
            candidates.append(txt)
    return _best_numeric_candidate(candidates)


def show_debug(frame, roi_images, roi_texts, parsed_data, frame_idx):
    """Display the current sampled frame with ROI boxes, cropped ROIs, and extracted text."""
    frame_vis = frame.copy()
    for name, roi in TELEMETRY_ROI.items():
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
    if parsed_data.get("speed", {}).get("velocity"):
        overlay_lines.append(
            f"Speed (km/h): {parsed_data.get('speed', {}).get('velocity')}"
        )
    if parsed_data.get("altitude", {}).get("altitude"):
        overlay_lines.append(
            f"Altitude (km): {parsed_data.get('altitude', {}).get('altitude')}"
        )
    if parsed_data.get("acceleration", {}).get("accel_g"):
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

    cv2.imshow("Sampled frame", frame_vis)
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
    r"(T[+-]\s*[\d:]+|\b\d{1,2}:\d{2}:\d{2}\b)"  # Accept Falcon-style T+ as well as hh:mm:ss
)
# Gauge numbers are bare or with a single decimal digit (e.g., 123 or 123.4).
VEL_REGEX = re.compile(r"(\d+(?:\.\d)?)")
ALT_REGEX = re.compile(r"(\d+(?:\.\d)?)")
ACC_REGEX = re.compile(r"(\d+(?:\.\d)?)")

def parse_telemetry_text(text, origin=None):
    """
    Parse text according to ROI origin:
      origin == "time": only time
      origin == "speed": only velocity
      origin == "altitude": only altitude
      origin == "acceleration": only accel_g
      None/other: parse all fields

    Returns dict with:
        {
            "time_str": str or None,
            "velocity": float or None,   # in km/h if possible
            "altitude": float or None,   # in km
            "accel_g": float or None,
        }
    """
    result = {"time_str": None, "velocity": None, "altitude": None, "accel_g": None}
    origin = origin or ""
    parse_time = origin == "time" or origin == ""
    parse_vel = origin == "speed" or origin == ""
    parse_alt = origin == "altitude" or origin == ""
    parse_acc = origin == "acceleration" or origin == ""

    # Time
    if parse_time:
        m_time = TIME_REGEX.search(text)
        if m_time:
            result["time_str"] = m_time.group(1).replace(" ", "")

    # Velocity (pure number from ROI; assume km/h upstream)
    if parse_vel:
        v_match = None
        for m in VEL_REGEX.finditer(text):
            v_match = m  # last one
        if v_match:
            v_val_str = v_match.group(1)
            try:
                result["velocity"] = float(v_val_str.replace(",", ""))
            except ValueError:
                pass

    # Altitude (pure number from ROI; assume km upstream)
    if parse_alt:
        a_match = None
        for m in ALT_REGEX.finditer(text):
            a_match = m  # last one
        if a_match:
            a_val_str = a_match.group(1)
            try:
                result["altitude"] = float(a_val_str.replace(",", ""))
            except ValueError:
                pass

    # Acceleration (g, pure number from ROI)
    if parse_acc:
        a_match = None
        for m in ACC_REGEX.finditer(text):
            a_match = m  # last one
        if a_match:
            a_val_str = a_match.group(1)
            try:
                result["accel_g"] = float(a_val_str.replace(",", ""))
            except ValueError:
                pass

    return result


# ========== MAIN EXTRACTION LOOP ==========

def extract_telemetry_to_csv(start_time_sec=None, end_time_sec=None):
    cap = cv2.VideoCapture(str(VIDEO_FILENAME))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {VIDEO_FILENAME}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(int(round(fps / SAMPLE_FPS)), 1)

    print(f"[INFO] Video FPS: {fps:.2f}, total frames: {total_frames}")
    print(f"[INFO] Sampling every {frame_interval} frames ~ {SAMPLE_FPS} Hz")

    # Seek to the desired start time, if provided.
    if start_time_sec is not None and fps > 0:
        start_frame = int(start_time_sec * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        print(f"[INFO] Starting at {start_time_sec:.2f}s (frame {start_frame})")
    else:
        start_frame = 0

    if end_time_sec is not None and fps > 0:
        end_frame = int(end_time_sec * fps)
        if end_frame < start_frame:
            raise ValueError("END_TIME_SEC is before START_TIME_SEC.")
        print(f"[INFO] Stopping at {end_time_sec:.2f}s (frame {end_frame})")
    else:
        end_frame = None

    # Estimate how many samples we'll take for progress reporting.
    if total_frames > 0:
        final_frame = end_frame if end_frame is not None else total_frames - 1
        span_frames = max(final_frame - start_frame + 1, 0)
        total_samples = (span_frames + frame_interval - 1) // frame_interval
    else:
        total_samples = None

    rows = []
    frame_idx = start_frame
    sample_idx = 0

    while True:
        if end_frame is not None and frame_idx > end_frame:
            print("[INFO] Reached end of requested segment.")
            break

        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx % frame_interval != 0:
            frame_idx += 1
            continue

        time_sec = frame_idx / fps if fps > 0 else None

        # For each ROI (speed, altitude, acceleration, time)
        roi_texts = {}
        parsed_data = {}

        roi_images = {}
        for name, roi in TELEMETRY_ROI.items():
            (y1, x1) = roi["yx1"]
            (y2, x2) = roi["yx2"]
            roi_img = frame[y1:y2, x1:x2]
            roi_images[name] = roi_img
            # Use tighter whitelist for the time ROI to improve hh:mm:ss recognition.
            if name == "time":
                text = ocr_time_robust(roi_img)
            elif name in ("speed", "acceleration", "altitude"):
                text = ocr_gauge_robust(roi_img)
            else:
                text = ocr_telemetry(roi_img, config=TESSERACT_CONFIG, invert=False)
            roi_texts[name] = text
            parsed = parse_telemetry_text(text, origin=name)
            parsed = sanitize_parsed(parsed)
            # Speed ROI is dedicated to speed only; drop altitude to avoid false positives.
            if name == "speed":
                parsed["altitude"] = None
            # Right ROI is acceleration gauge; drop speed/altitude so they don't pollute plots.
            if name == "acceleration":
                parsed["velocity"] = None
                parsed["altitude"] = None
            parsed_data[name] = parsed

        # Prefer an explicit time ROI if available.
        time_str_primary = (
            parsed_data.get("time", {}).get("time_str")
        )
        # Speed/Altitude/Accel primaries are single-source per user mapping.
        vel_primary = (
            parsed_data.get("speed", {}).get("velocity")
        )
        alt_primary = (
            parsed_data.get("altitude", {}).get("altitude")
        )
        accel_primary = parsed_data.get("acceleration", {}).get("accel_g")

        # Build row (you can extend this depending on whether ROIs contain different data)
        row = {
            "frame_idx": frame_idx,
            "t_video_sec": time_sec,
            "raw_speed": roi_texts.get("speed"),
            "raw_acceleration": roi_texts.get("acceleration"),
            "raw_altitude_roi": roi_texts.get("altitude"),
            "time_str_speed": parsed_data["speed"]["time_str"],
            "vel_kmh_speed": parsed_data["speed"]["velocity"],
            "alt_km_speed": parsed_data["speed"]["altitude"],  # expected None (speed-only)
            "time_str_acceleration": parsed_data["acceleration"]["time_str"],
            "vel_kmh_acceleration": parsed_data["acceleration"]["velocity"],  # expected None (accel-only)
            "alt_km_acceleration": parsed_data["acceleration"]["altitude"],  # expected None (accel-only)
            "accel_g_acceleration": parsed_data["acceleration"].get("accel_g"),
            "time_str_altitude_roi": parsed_data.get("altitude", {}).get("time_str"),
            "vel_kmh_altitude_roi": parsed_data.get("altitude", {}).get("velocity"),
            "alt_km_altitude_roi": parsed_data.get("altitude", {}).get("altitude"),
            "raw_time": roi_texts.get("time"),
            "time_str_time": parsed_data.get("time", {}).get("time_str"),
            "time_str_primary": time_str_primary,
            "vel_kmh_primary": vel_primary,
            "alt_km_primary": alt_primary,
            "accel_g_primary": accel_primary,
        }

        rows.append(row)

        sample_idx += 1
        if sample_idx % 50 == 0:
            if total_samples:
                pct = min(sample_idx / total_samples * 100, 100.0)
                print(f"[INFO] Processed {sample_idx}/{total_samples} samples (~{pct:.1f}%)...")
            else:
                print(f"[INFO] Processed {sample_idx} samples...")

        if DEBUG_SHOW_SAMPLES:
            stop_debug = show_debug(frame, roi_images, roi_texts, parsed_data, frame_idx)
            if stop_debug:
                print("[INFO] Debug display stopped by user ('q').")
                break

        frame_idx += 1

    cap.release()
    if DEBUG_SHOW_SAMPLES:
        cv2.destroyAllWindows()

    df = pd.DataFrame(rows)

    # Post-filter: drop outliers that deviate too far from local median.
    numeric_cols = [
        "vel_kmh_speed", "vel_kmh_primary",
        "alt_km_speed", "alt_km_acceleration", "alt_km_primary", "alt_km_altitude_roi",
        "accel_g_acceleration", "accel_g_primary",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = filter_series_outliers(df[col])

    # Coalesce to single columns per metric.
    df["speed_kmh"] = df["vel_kmh_speed"].fillna(df["vel_kmh_primary"])
    df["altitude_km"] = df["alt_km_altitude_roi"].fillna(df["alt_km_primary"])
    df["acceleration_g"] = df["accel_g_acceleration"].fillna(df["accel_g_primary"])

    # Keep only the relevant columns and drop rows with no useful telemetry.
    keep_cols = [
        "t_video_sec",
        "time_str_primary",
        "speed_kmh",
        "altitude_km",
        "acceleration_g",
    ]
    metrics_cols = ["speed_kmh", "altitude_km", "acceleration_g"]
    df_reduced = df[keep_cols].copy()
    df_reduced = df_reduced.dropna(subset=metrics_cols, how="all")

    csv_path = OUTPUT_DIR / "falcon9_telemetry.csv"
    df_reduced.to_csv(csv_path, index=False)
    print(f"[INFO] Saved CSV to: {csv_path}")
    return df_reduced


def plot_extracted_data(df):
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

    # Speed
    ax_speed = axes[0]
    speed_plotted = []
    speed_plotted.append(_plot(ax_speed, "speed_kmh", "Speed (km/h)", "tab:green"))
    ax_speed.set_ylabel("Speed (km/h)")
    if any(speed_plotted):
        ax_speed.legend()
    ax_speed.grid(True, alpha=0.3)

    # Altitude
    ax_alt = axes[1]
    alt_plotted = []
    alt_plotted.append(_plot(ax_alt, "altitude_km", "Altitude (km)", "tab:purple"))
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
    ax_acc.set_xlabel("Video time (s)")
    ax_acc.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = OUTPUT_DIR / "falcon9_telemetry.png"
    plt.savefig(plot_path, dpi=150)
    print(f"[INFO] Saved plot to: {plot_path}")
    plt.show()


if __name__ == "__main__":
    ensure_output_dir()
    # download_video()

    # 1) First run: uncomment the next line, run script, and visually tune TELEMETRY_ROI
    # show_sample_frame(START_TIME_SEC)

    # 2) After ROI is correct, comment show_sample_frame() and run extraction:
    # df_out = extract_telemetry_to_csv(START_TIME_SEC, END_TIME_SEC)

    # 3) Reload from disk to ensure plotting uses the saved CSV.
    csv_path = OUTPUT_DIR / "falcon9_telemetry.csv"
    df_plot = pd.read_csv(csv_path)
    plot_extracted_data(df_plot)
