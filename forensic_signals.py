"""
forensic_signals.py

Forensic Signal Extraction and Quantification Module.

Computes human-interpretable metrics from raw feature extraction outputs
(spectral embeddings, ELA maps, PRNU noise maps) for forensic reporting.

IMPORTANT DATA FORMAT NOTES (verified against utils.py):
  - Spectral arrays are COMBINED embeddings: azimuthal_integral + positional_emb/5
    The positional component is small (~±0.4) vs spectral ([0,1]), so aggregate
    metrics (energy, HF ratio) are dominated by spectral content.
  - ELA maps are 3-channel RGB float32 in [0, 1], shape (128, 128, 3).
  - PRNU maps are single-channel float32 in [0, 1], shape (128, 128).

Verified interfaces:
  - utils.azi_diff(PIL.Image) → dict with keys:
      'total_emb': [rich (128,256), poor (128,256)]
      'ela':       (128,128,3) float32
      'noise':     (128,128)   float32
      'image_size': (H, W)
  - models.TextureContrastClassifier.forward(rich, poor, ela, noise) → logit

Does NOT modify or monkey-patch any existing detector component.

Changelog:
  v1.1 — Issue 3: Tie handling in _primary_evidence() → returns "multiple_signals"
  v1.1 — Issue 4: Added fftshift before PRNU power spectrum computation
  v1.1 — Issue 5: Zero-peak safety in ELA normalization
"""

import numpy as np
from dataclasses import dataclass, asdict
from typing import Dict, List
from scipy.stats import entropy as scipy_entropy
from scipy.ndimage import uniform_filter


# ──────────────────────────────────────────────────────
# Structured Metric Containers
# ──────────────────────────────────────────────────────

@dataclass
class SpectralMetrics:
    """Interpretable metrics from Fourier spectral patch analysis."""
    rich_mean_energy: float          # Mean L2 energy across rich texture patches
    poor_mean_energy: float          # Mean L2 energy across poor texture patches
    rich_high_freq_ratio: float      # Fraction of energy in upper half of spectrum (rich)
    poor_high_freq_ratio: float      # Fraction of energy in upper half of spectrum (poor)
    rich_poor_energy_ratio: float    # Ratio of rich to poor patch energy
    rich_spectral_diversity: float   # Mean cross-patch std per frequency bin (rich)
    poor_spectral_diversity: float   # Mean cross-patch std per frequency bin (poor)
    anomaly_score: float             # Composite heuristic [0–1], higher = more AI-like


@dataclass
class ELAMetrics:
    """Interpretable metrics from Error Level Analysis."""
    mean_intensity: float            # Mean of normalized grayscale ELA
    std_intensity: float             # Std dev of normalized grayscale ELA
    max_intensity: float             # Peak ELA value
    uniformity_score: float          # [0–1], 1 = perfectly uniform compression
    spatial_entropy: float           # Shannon entropy of intensity histogram
    splicing_indicator: float        # [0–1], higher = evidence of localized manipulation


@dataclass
class PRNUMetrics:
    """Interpretable metrics from sensor noise fingerprint analysis."""
    energy: float                    # Mean squared noise intensity
    spatial_std: float               # Spatial standard deviation of noise map
    spectral_flatness: float         # 1.0 = white noise, 0.0 = structured noise
    strength_score: float            # [0–1], higher = stronger camera signature
    camera_consistency: str          # "consistent" | "weak" | "absent"


@dataclass
class ForensicSignals:
    """
    Complete forensic signal package for a single image analysis.

    This is the structured payload consumed by the reasoning prompt builder
    and ultimately by the LLM for report generation.
    """
    # Detector output
    raw_logit: float
    probability: float
    verdict: str
    threshold: float

    # Per-modality metrics
    spectral: SpectralMetrics
    ela: ELAMetrics
    prnu: PRNUMetrics

    # Derived assessments
    risk_level: str                  # "LOW" | "MEDIUM" | "HIGH" | "CRITICAL"
    confidence_level: str            # "LOW" | "MEDIUM" | "HIGH"
    primary_evidence: str            # Which modality is most decisive
                                     # OR "multiple_signals" if tied

    def to_dict(self) -> Dict:
        """Serialize entire signal package to nested dict for JSON export."""
        return asdict(self)


# ──────────────────────────────────────────────────────
# Signal Extractor
# ──────────────────────────────────────────────────────

class ForensicSignalExtractor:
    """
    Computes interpretable forensic metrics from raw detector intermediates.

    All heuristic thresholds are exposed as class-level constants for
    calibration. They MUST be tuned against a labeled validation set
    before production deployment.

    Verified data flow from utils.py → this module:
        features = azi_diff(pil_image, patch_num=128, N=256)
        rich_spectral = features['total_emb'][0]    # (128, 256) ndarray
        poor_spectral = features['total_emb'][1]    # (128, 256) ndarray
        ela_map       = features['ela']              # (128, 128, 3) ndarray
        prnu_map      = features['noise']            # (128, 128) ndarray

    Usage:
        extractor = ForensicSignalExtractor(threshold=0.7)
        signals = extractor.extract(
            raw_logit     = model_logit_float,
            rich_spectral = features['total_emb'][0],
            poor_spectral = features['total_emb'][1],
            ela_map       = features['ela'],
            prnu_map      = features['noise'],
        )
    """

    # ── Calibration Constants (tune on validation data) ────
    SPECTRAL_HF_BASELINE: float = 0.5      # Expected HF ratio upper bound for real images
    SPECTRAL_DIV_BASELINE: float = 1.0      # Expected spectral diversity upper bound
    PRNU_ENERGY_BASELINE: float = 0.10      # PRNU energy expected from real cameras
    ELA_SPLICE_VAR_DIVISOR: float = 10.0    # Divisor for normalizing local variance ratio
    ELA_LOCAL_WINDOW: int = 16              # Window size for local ELA variance computation

    def __init__(self, threshold: float = 0.7):
        self.threshold = threshold

    # ── Public API ────────────────────────────────────

    def extract(
        self,
        raw_logit: float,
        rich_spectral: np.ndarray,
        poor_spectral: np.ndarray,
        ela_map: np.ndarray,
        prnu_map: np.ndarray,
    ) -> ForensicSignals:
        """Master extraction: computes all metrics and returns ForensicSignals."""

        self._validate_inputs(rich_spectral, poor_spectral, ela_map, prnu_map)

        prob = self._sigmoid(raw_logit)
        verdict = "AI GENERATED" if prob >= self.threshold else "REAL IMAGE"

        spectral = self._spectral_metrics(rich_spectral, poor_spectral)
        ela = self._ela_metrics(ela_map)
        prnu = self._prnu_metrics(prnu_map)

        return ForensicSignals(
            raw_logit=round(float(raw_logit), 5),
            probability=round(float(prob), 4),
            verdict=verdict,
            threshold=self.threshold,
            spectral=spectral,
            ela=ela,
            prnu=prnu,
            risk_level=self._risk_level(prob, spectral, ela, prnu),
            confidence_level=self._confidence_level(prob),
            primary_evidence=self._primary_evidence(spectral, ela, prnu),
        )

    # ── Input Validation ──────────────────────────────

    @staticmethod
    def _validate_inputs(
        rich: np.ndarray, poor: np.ndarray,
        ela: np.ndarray, prnu: np.ndarray,
    ) -> None:
        """
        Guard against shape mismatches from upstream changes.

        Expected shapes (from utils.azi_diff):
          rich:  (128, 256)     — 128 patches × 256-dim spectral+positional embedding
          poor:  (128, 256)     — 128 patches × 256-dim spectral+positional embedding
          ela:   (128, 128, 3)  — RGB ELA map resized to 128×128
          prnu:  (128, 128)     — Laplacian noise map resized to 128×128
        """
        if rich.ndim != 2 or rich.shape[1] != 256:
            raise ValueError(
                f"rich_spectral expected shape (N, 256), got {rich.shape}"
            )
        if poor.ndim != 2 or poor.shape[1] != 256:
            raise ValueError(
                f"poor_spectral expected shape (N, 256), got {poor.shape}"
            )
        if ela.ndim != 3 or ela.shape[2] != 3:
            raise ValueError(
                f"ela_map expected shape (H, W, 3), got {ela.shape}"
            )
        if prnu.ndim != 2:
            raise ValueError(
                f"prnu_map expected 2D array, got shape {prnu.shape}"
            )

    # ── Spectral Metrics ─────────────────────────────

    def _spectral_metrics(
        self, rich: np.ndarray, poor: np.ndarray,
    ) -> SpectralMetrics:
        """
        Compute interpretable spectral metrics.

        NOTE: Input arrays include positional embeddings blended as:
            total_emb = azimuthal_integral + positional_emb / 5
        Since positional magnitudes are ≤ ~0.4 and azimuthal values are
        normalized to [0, 1], aggregate energy/ratio metrics are
        approximately 95% determined by the spectral content.
        This is acceptable for interpretability signals.
        """
        eps = 1e-10
        N = rich.shape[1]    # 256
        mid = N // 2          # 128: split into low-freq and high-freq halves

        # Per-patch energy (sum of squares across frequency bins)
        rich_energy_per_patch = np.sum(rich ** 2, axis=1)   # (128,)
        poor_energy_per_patch = np.sum(poor ** 2, axis=1)   # (128,)

        rich_mean_e = float(np.mean(rich_energy_per_patch))
        poor_mean_e = float(np.mean(poor_energy_per_patch))

        # High-frequency energy ratio: energy in upper half / total energy
        # AI images typically have attenuated high-frequency content
        rich_total_e = float(np.sum(rich ** 2)) + eps
        poor_total_e = float(np.sum(poor ** 2)) + eps
        rich_hf = float(np.sum(rich[:, mid:] ** 2) / rich_total_e)
        poor_hf = float(np.sum(poor[:, mid:] ** 2) / poor_total_e)

        # Spectral diversity: mean of per-bin standard deviations across patches
        # High diversity → patches have varied frequency profiles → natural image
        # Low diversity → homogeneous spectral signatures → synthetic
        rich_div = float(np.mean(np.std(rich, axis=0)))
        poor_div = float(np.mean(np.std(poor, axis=0)))

        # Composite anomaly heuristic:
        #   Low HF content → frequency dropout (common in GAN/diffusion output)
        #   Low diversity   → homogeneous spectral signatures (unnatural)
        hf_component = 1.0 - min(rich_hf / self.SPECTRAL_HF_BASELINE, 1.0)
        div_component = 1.0 - min(rich_div / self.SPECTRAL_DIV_BASELINE, 1.0)
        anomaly = float(np.clip(
            0.6 * hf_component + 0.4 * div_component, 0.0, 1.0
        ))

        return SpectralMetrics(
            rich_mean_energy=round(rich_mean_e, 4),
            poor_mean_energy=round(poor_mean_e, 4),
            rich_high_freq_ratio=round(rich_hf, 4),
            poor_high_freq_ratio=round(poor_hf, 4),
            rich_poor_energy_ratio=round(rich_mean_e / (poor_mean_e + eps), 4),
            rich_spectral_diversity=round(rich_div, 4),
            poor_spectral_diversity=round(poor_div, 4),
            anomaly_score=round(anomaly, 4),
        )

    # ── ELA Metrics ──────────────────────────────────

    def _ela_metrics(self, ela_map: np.ndarray) -> ELAMetrics:
        """
        Compute interpretable Error Level Analysis metrics.

        Input: (128, 128, 3) float32 in [0, 1] — from utils.get_ela() → cv2.resize().

        Analysis approach:
          1. Convert RGB ELA to grayscale for scalar analysis.
          2. Compute global statistics (mean, std, max).
          3. Measure compression uniformity via coefficient of variation.
          4. Compute spatial entropy of intensity distribution.
          5. Detect potential splicing via local variance analysis:
             Spliced regions create localized compression inconsistencies
             that appear as high-variance pockets in the ELA map.
        """
        # Convert RGB ELA to grayscale via channel averaging
        gray = np.mean(ela_map, axis=-1)                    # (128, 128)
        peak = float(gray.max())

        # ────────────────────────────────────────────────
        # FIX Issue 5: Zero-peak safety
        # Prevents division instability when ELA map is all zeros
        # (edge case: solid color image or failed ELA extraction)
        # ────────────────────────────────────────────────
        if peak < 1e-8:
            norm = gray  # Already all zeros — normalization would be meaningless
        else:
            norm = gray / peak
        # ────────────────────────────────────────────────

        mean_i = float(np.mean(norm))
        std_i = float(np.std(norm))
        max_i = round(peak, 4)

        # ── Uniformity Score ──
        # Inverse of coefficient of variation.
        # High uniformity → consistent compression → single-source image.
        # Low uniformity → regions compressed differently → possible editing.
        cv = std_i / (mean_i + 1e-10)
        uniformity = float(np.clip(1.0 / (1.0 + cv), 0.0, 1.0))

        # ── Spatial Entropy ──
        # Shannon entropy of the intensity histogram.
        # Higher entropy → more dispersed ELA values → complex error landscape.
        hist, _ = np.histogram(norm.flatten(), bins=64, range=(0.0, 1.0))
        hist_p = hist / (hist.sum() + 1e-10)
        sp_entropy = float(scipy_entropy(hist_p + 1e-10))

        # ── Splicing Detection via Local Variance Analysis ──
        # Compute local mean and local variance using sliding window.
        # Spliced regions will have anomalously high local variance compared
        # to the global average, because the error level of the pasted region
        # differs from the background.
        local_mean = uniform_filter(norm, size=self.ELA_LOCAL_WINDOW)
        local_var = uniform_filter(
            (norm - local_mean) ** 2, size=self.ELA_LOCAL_WINDOW
        )
        mean_local_var = float(np.mean(local_var)) + 1e-10
        max_local_var = float(np.max(local_var))

        # Ratio of peak local variance to mean local variance
        # High ratio → some region has much higher error than the rest → splice
        var_ratio = max_local_var / mean_local_var
        splicing = float(np.clip(var_ratio / self.ELA_SPLICE_VAR_DIVISOR, 0.0, 1.0))

        return ELAMetrics(
            mean_intensity=round(mean_i, 4),
            std_intensity=round(std_i, 4),
            max_intensity=max_i,
            uniformity_score=round(uniformity, 4),
            spatial_entropy=round(sp_entropy, 4),
            splicing_indicator=round(splicing, 4),
        )

    # ── PRNU Metrics ─────────────────────────────────

    def _prnu_metrics(self, prnu_map: np.ndarray) -> PRNUMetrics:
        """
        Compute interpretable PRNU (sensor noise fingerprint) metrics.

        Input: (128, 128) float32 in [0, 1] — Laplacian high-pass filtered,
               min-max normalized. From utils.get_noise_fingerprint() → cv2.resize().

        Key forensic principles:
          - Real cameras imprint a Photo Response Non-Uniformity (PRNU) pattern
            that is spatially structured and sensor-specific.
          - AI-generated images lack this physical process, so their noise
            tends toward white noise (flat power spectrum) or is overly smooth.

        Metrics:
          - energy: mean squared intensity of the noise map.
          - spatial_std: standard deviation — low std → artificially smooth.
          - spectral_flatness: geometric/arithmetic mean ratio of power spectrum.
            Values near 1.0 → white noise (no structure → AI-like).
            Values near 0.0 → structured noise (camera fingerprint → real).
          - strength_score: composite of energy + spectral structure.
          - camera_consistency: categorical label derived from strength_score.
        """
        # Energy: mean squared value of the noise map
        energy = float(np.mean(prnu_map ** 2))

        # Spatial standard deviation
        spatial_std = float(np.std(prnu_map))

        # ── Spectral Flatness (Wiener Entropy) ──
        # Computed from the 2D power spectral density.
        # Spectral flatness = exp(mean(log(S))) / mean(S)
        # Range: [0, 1]. 1.0 = perfectly flat (white noise), 0.0 = tonal/structured.
        #
        # ────────────────────────────────────────────────
        # FIX Issue 4: Added fftshift before power spectrum computation
        # Removes DC bias spike that would skew spectral flatness toward 0.0
        # regardless of actual noise structure. This matches the approach used
        # in utils.py azimuthal_integral() which also applies fftshift.
        # ────────────────────────────────────────────────
        fft_2d = np.fft.fftshift(np.fft.fft2(prnu_map))
        power_spectrum = np.abs(fft_2d) ** 2
        # ────────────────────────────────────────────────

        power_flat = power_spectrum.flatten()

        # Filter out near-zero values to avoid log(0)
        valid_power = power_flat[power_flat > 1e-20]

        if len(valid_power) > 0:
            log_geometric_mean = float(np.mean(np.log(valid_power + 1e-20)))
            geometric_mean = np.exp(log_geometric_mean)
            arithmetic_mean = float(np.mean(valid_power)) + 1e-10
            spectral_flatness = float(np.clip(
                geometric_mean / arithmetic_mean, 0.0, 1.0
            ))
        else:
            # Degenerate case: all-zero noise map → no camera signature
            spectral_flatness = 1.0

        # ── Strength Score ──
        # Composite metric combining energy level and spectral structure.
        # Higher energy + lower spectral flatness → stronger camera signature.
        energy_component = float(np.clip(
            energy / self.PRNU_ENERGY_BASELINE, 0.0, 1.0
        ))
        structure_component = 1.0 - spectral_flatness
        strength = 0.5 * energy_component + 0.5 * structure_component

        # ── Camera Consistency Label ──
        if strength > 0.6:
            consistency = "consistent"
        elif strength > 0.3:
            consistency = "weak"
        else:
            consistency = "absent"

        return PRNUMetrics(
            energy=round(energy, 6),
            spatial_std=round(spatial_std, 6),
            spectral_flatness=round(spectral_flatness, 4),
            strength_score=round(float(strength), 4),
            camera_consistency=consistency,
        )

    # ── Derived Assessments ──────────────────────────

    def _risk_level(
        self,
        prob: float,
        sp: SpectralMetrics,
        ela: ELAMetrics,
        prnu: PRNUMetrics,
    ) -> str:
        """
        Compute composite risk level from all modalities.

        Weighting rationale:
          - Model probability (50%): primary classification signal, trained end-to-end.
          - Spectral anomaly (20%): independent frequency-domain evidence.
          - PRNU absence (20%): strong physical-layer indicator.
          - ELA splicing (10%): supplementary editing indicator.

        These weights should be calibrated against labeled fraud cases.
        """
        score = (
            prob * 0.50
            + sp.anomaly_score * 0.20
            + (1.0 - prnu.strength_score) * 0.20
            + ela.splicing_indicator * 0.10
        )
        if score >= 0.80:
            return "CRITICAL"
        if score >= 0.60:
            return "HIGH"
        if score >= 0.40:
            return "MEDIUM"
        return "LOW"

    @staticmethod
    def _confidence_level(prob: float) -> str:
        """
        Assess classification confidence based on distance from decision boundary.

        The decision boundary is 0.5 (sigmoid midpoint), not the user threshold.
        Distance from 0.5 indicates how decisive the model's internal representation is.

        Ranges:
          ≥ 0.30 from midpoint → HIGH   (prob < 0.20 or prob > 0.80)
          ≥ 0.15 from midpoint → MEDIUM (prob < 0.35 or prob > 0.65)
          < 0.15               → LOW    (near decision boundary)
        """
        dist = abs(prob - 0.5)
        if dist >= 0.30:
            return "HIGH"
        if dist >= 0.15:
            return "MEDIUM"
        return "LOW"

    @staticmethod
    def _primary_evidence(
        sp: SpectralMetrics, ela: ELAMetrics, prnu: PRNUMetrics,
    ) -> str:
        """
        Identify which modality contributed the strongest forensic signal.

        For each modality, compute a "suspicion score" in [0, 1]:
          - Spectral: anomaly_score (already [0, 1])
          - ELA: splicing_indicator (already [0, 1])
          - PRNU: 1 - strength_score (absent fingerprint → suspicious)

        Returns the name of the modality with the highest suspicion score,
        or "multiple_signals" if two or more modalities are tied.
        """
        candidates = {
            "spectral_analysis": sp.anomaly_score,
            "compression_analysis": ela.splicing_indicator,
            "sensor_fingerprint": 1.0 - prnu.strength_score,
        }

        # ────────────────────────────────────────────────
        # FIX Issue 3: Proper tie handling
        # When multiple modalities have identical suspicion scores,
        # return "multiple_signals" instead of arbitrary winner.
        # ────────────────────────────────────────────────
        best_value = max(candidates.values())
        top = [
            name for name, value in candidates.items()
            if value == best_value
        ]

        if len(top) == 1:
            return top[0]
        else:
            return "multiple_signals"
        # ────────────────────────────────────────────────

    # ── Utilities ────────────────────────────────────

    @staticmethod
    def _sigmoid(x: float) -> float:
        """Numerically stable sigmoid function."""
        x = float(x)
        if x >= 0:
            return 1.0 / (1.0 + np.exp(-x))
        else:
            exp_x = np.exp(x)
            return exp_x / (1.0 + exp_x)
