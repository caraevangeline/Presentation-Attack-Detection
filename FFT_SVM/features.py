"""FFT-based feature extraction for screen-attack (recapture) detection.

Rationale
---------
An LCD/OLED screen has a fixed sub-pixel grid and, for video, a fixed
refresh rate. Recapturing that screen with a camera introduces artifacts a
live face in front of the camera does not produce:

  1. Moire aliasing between the screen's pixel grid and the camera
     sensor's pixel grid -> periodic, high-frequency energy that shows up
     as sharp peaks in the 2D Fourier spectrum, instead of the smooth,
     roughly 1/f^2 falloff typical of natural image statistics.
  2. Color-channel-specific sub-pixel striping (RGB stripe / PenTile
     panels) that leaves a signature in individual color channels even
     when it is washed out in luminance.

These are global, texture-level artifacts, so this does not depend on a
face detector -- it works even on frames where a face detector might
struggle (e.g. a screen photographed at an angle).

Deliberate exclusion: raw image resolution / aspect ratio. In this
dataset bona fide images are all exactly 1024x1024 while screen-attack
images have widely varying native resolutions -- an artifact of how the
dataset was assembled, not a real anti-spoofing signal. Every image is
resized to a fixed working resolution before any feature is computed so
the classifier cannot key on original image size.
"""

import cv2
import numpy as np

RESIZE_DIM = 384          # fixed working resolution: keeps feature vector
                           # size constant and removes original-size as a cue
N_RADIAL_BINS = 32


def _load_gray_and_channels(path, size=RESIZE_DIM):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64)
    channels = [img[:, :, i].astype(np.float64) for i in range(3)]  # B, G, R
    return gray, channels


def _radial_power_profile(gray, n_bins=N_RADIAL_BINS):
    h, w = gray.shape
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    power = np.abs(fshift) ** 2

    cy, cx = h // 2, w // 2
    y, x = np.indices((h, w))
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    r_max = r.max()
    bin_edges = np.linspace(0, r_max, n_bins + 1)

    profile = np.zeros(n_bins)
    for i in range(n_bins):
        mask = (r >= bin_edges[i]) & (r < bin_edges[i + 1])
        if mask.any():
            profile[i] = power[mask].mean()
    return profile


def _peakiness(profile):
    # Skip DC + first bin: dominated by low-freq image content, not texture.
    band = profile[2:]
    if band.sum() <= 0:
        return 0.0
    return float(band.max() / (np.median(band) + 1e-8))


def _high_freq_ratio(profile):
    n = len(profile)
    hi = profile[int(n * 0.6):]
    return float(hi.sum() / (profile.sum() + 1e-8))


def _spectral_slope(profile):
    n = len(profile)
    idx = np.arange(2, n)
    vals = profile[2:]
    valid = vals > 0
    if valid.sum() < 3:
        return 0.0
    slope, _ = np.polyfit(np.log(idx[valid]), np.log(vals[valid]), 1)
    return float(slope)


FEATURE_NAMES = (
    [f"radial_bin_{i}" for i in range(N_RADIAL_BINS)]
    + ["peakiness", "high_freq_ratio", "spectral_slope"]
    + ["peakiness_B", "peakiness_G", "peakiness_R"]
    + ["log_laplacian_var"]
)


def extract_features(path):
    """Extract a fixed-length FFT feature vector from an image file."""
    gray, channels = _load_gray_and_channels(path)

    profile = _radial_power_profile(gray)
    log_profile = np.log1p(profile)
    norm_profile = log_profile / (log_profile.sum() + 1e-8)  # shape-only, scale-free

    feats = list(norm_profile)
    feats.append(_peakiness(profile))
    feats.append(_high_freq_ratio(profile))
    feats.append(_spectral_slope(profile))

    # Per-channel peakiness catches sub-pixel color moire even when it is
    # washed out in the luminance channel.
    for ch in channels:
        feats.append(_peakiness(_radial_power_profile(ch)))

    # Sharpness proxy: recaptures often carry different focus/blur statistics.
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    feats.append(float(np.log1p(lap_var)))

    return np.array(feats, dtype=np.float64)
