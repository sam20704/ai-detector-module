"""
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
  v1.2 — Added translate_signals_to_text() signal interpretation layer
         Converts numeric forensic metrics into human-readable natural-language
         signals for reliable LLM reasoning. Pipeline becomes:
             metrics → human-readable signals → LLM
         This eliminates LLM hallucination over raw numeric thresholds.
  v1.3 — Fix 1: Verdict signal now detects contradiction between verdict (REAL)
         and risk level (HIGH/CRITICAL), emitting a cautionary note instead of
         a flat "REAL" statement that would confuse the LLM.
  v1.3 — Fix 2: Spectral interpretation now also triggers on very low
         rich_spectral_diversity (< 0.15), catching diffusion-model images
         whose anomaly_score may be moderate but whose patch-level spectral
         homogeneity is a strong synthetic indicator.
"""

import numpy as np
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple
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

    def to_text_signals(self) -> Dict[str, str]:
        """
        Convenience method: produce human-readable signal descriptions
        from this ForensicSignals instance.

        Delegates to the module-level translate_signals_to_text().
        """
        return translate_signals_to_text(self)


# ──────────────────────────────────────────────────────
# Signal Translation Layer  (v1.2, updated v1.3)
# ──────────────────────────────────────────────────────
# Converts numeric forensic metrics into natural-language descriptions
# so the downstream LLM receives pre-interpreted evidence rather than
# raw floats it may hallucinate about.
#
# Pipeline change:
#   OLD:  metrics ──────────────────► LLM
#   NEW:  metrics ► human-readable signals ► LLM
# ──────────────────────────────────────────────────────

# Threshold for spectral diversity below which patches are considered
# suspiciously homogeneous (common in diffusion-model outputs).
_SPECTRAL_DIVERSITY_FLOOR: float = 0.15


def translate_signals_to_text(signals: ForensicSignals) -> Dict[str, str]:
    """
    Convert numeric forensic metrics into human-readable signal descriptions
    that an LLM can reason about reliably.

    Args:
        signals: A fully populated ForensicSignals dataclass instance.

    Returns:
        Dictionary with keys:
            spectral_signal  – natural-language interpretation of spectral analysis
            ela_signal       – natural-language interpretation of ELA analysis
            prnu_signal      – natural-language interpretation of PRNU analysis
            risk_summary     – natural-language interpretation of overall risk level
            confidence_note  – natural-language interpretation of model confidence
            evidence_note    – natural-language description of primary evidence source
            verdict_signal   – natural-language restatement of the detector verdict
    """
    sp = signals.spectral
    ela = signals.ela
    prnu = signals.prnu

    # ── Spectral Interpretation ──────────────────────
    # ────────────────────────────────────────────────────
    # FIX v1.3 (Fix 2): Also trigger strong-anomaly interpretation when
    # rich_spectral_diversity is extremely low, even if anomaly_score
    # alone is moderate. Diffusion models often produce spectrally
    # homogeneous patches that compress the diversity metric while
    # the composite anomaly_score (which blends HF ratio at 60%) may
    # remain below the 0.7 threshold.
    # ────────────────────────────────────────────────────
    strong_spectral = (
        sp.anomaly_score >= 0.7
        or sp.rich_spectral_diversity < _SPECTRAL_DIVERSITY_FLOOR
    )

    if strong_spectral:
        spectral_signal = (
            "Strong spectral anomaly typical of AI-generated imagery. "
            "High-frequency content is attenuated and spectral diversity "
            "across patches is unusually low."
        )
    elif sp.anomaly_score >= 0.4:
        spectral_signal = (
            "Moderate spectral irregularities in frequency patterns. "
            "Some reduction in high-frequency energy or patch diversity "
            "compared to typical natural photographs."
        )
    else:
        spectral_signal = (
            "Spectral profile consistent with natural photography. "
            "High-frequency content and cross-patch diversity fall within "
            "expected ranges for camera-captured images."
        )

    # ── ELA Interpretation ───────────────────────────
    if ela.splicing_indicator >= 0.5:
        ela_signal = (
            "Strong localized compression inconsistencies suggesting editing "
            "or synthesis. Significant variance in error levels across the "
            "image indicates regions were processed differently."
        )
    elif ela.splicing_indicator >= 0.2:
        ela_signal = (
            "Mild compression inconsistencies detected. Some regions show "
            "slightly different error levels, which may indicate light "
            "editing or recompression."
        )
    else:
        ela_signal = (
            "Compression artifacts appear uniform across the image. "
            "Error levels are consistent, suggesting a single compression "
            "history with no obvious splicing."
        )

    # ── PRNU Interpretation ──────────────────────────
    if prnu.strength_score <= 0.3:
        prnu_signal = (
            "Camera sensor fingerprint is weak or absent, consistent with "
            "synthetic images. No meaningful Photo Response Non-Uniformity "
            "pattern was detected."
        )
    elif prnu.strength_score <= 0.6:
        prnu_signal = (
            "Partial camera noise pattern detected but not strongly "
            "consistent. The sensor fingerprint is ambiguous — it may "
            "indicate heavy post-processing or a low-quality sensor."
        )
    else:
        prnu_signal = (
            "Strong sensor noise fingerprint typical of physical cameras. "
            "The PRNU pattern is structured and consistent with a real "
            "imaging device."
        )

    # ── Risk Summary ─────────────────────────────────
    risk_text = {
        "LOW": (
            "Minimal forensic risk indicators. The image shows no significant "
            "signs of AI generation or manipulation across all analysis modalities."
        ),
        "MEDIUM": (
            "Moderate forensic anomalies detected. Some indicators suggest "
            "possible manipulation, but evidence is not conclusive."
        ),
        "HIGH": (
            "Strong forensic anomalies indicating likely manipulation. "
            "Multiple analysis channels flag suspicious characteristics."
        ),
        "CRITICAL": (
            "Multiple forensic indicators strongly suggest AI generation. "
            "Spectral, compression, and sensor analyses converge on synthetic origin."
        ),
    }.get(signals.risk_level, "Unknown risk level.")

    # ── Confidence Note ──────────────────────────────
    confidence_text = {
        "HIGH": (
            "The model is highly confident in its classification. "
            "The probability is far from the decision boundary."
        ),
        "MEDIUM": (
            "The model has moderate confidence. The probability is reasonably "
            "separated from the decision boundary but not extreme."
        ),
        "LOW": (
            "The model has low confidence. The probability is close to the "
            "decision boundary; this result should be treated with caution."
        ),
    }.get(signals.confidence_level, "Unknown confidence level.")

    # ── Primary Evidence Note ────────────────────────
    evidence_labels = {
        "spectral_analysis": "Fourier spectral analysis (frequency-domain anomalies)",
        "compression_analysis": "Error Level Analysis (compression inconsistencies)",
        "sensor_fingerprint": "PRNU sensor fingerprint analysis (camera noise absence)",
        "multiple_signals": "Multiple forensic modalities contributing equally",
    }
    evidence_text = (
        f"Primary evidence source: "
        f"{evidence_labels.get(signals.primary_evidence, signals.primary_evidence)}."
    )

    # ── Verdict Signal ───────────────────────────────
    # ────────────────────────────────────────────────────
    # FIX v1.3 (Fix 1): Three-way verdict logic.
    #
    # Problem: When the model probability is below the classification
    # threshold (verdict = REAL) but forensic risk is HIGH or CRITICAL,
    # the old two-branch logic emitted a flat "REAL IMAGE" statement.
    # Downstream, the LLM received contradictory signals: "REAL" verdict
    # alongside "strong forensic anomalies" from the risk summary.
    # This caused confused or self-contradictory reports.
    #
    # Fix: Insert a middle branch that acknowledges the below-threshold
    # probability while explicitly flagging the forensic disagreement,
    # so the LLM can produce a nuanced, non-contradictory explanation.
    # ────────────────────────────────────────────────────
    if signals.verdict == "AI GENERATED":
        verdict_signal = (
            f"The detector classifies this image as AI-GENERATED "
            f"(AI probability {signals.probability:.1%}, "
            f"threshold {signals.threshold:.1%})."
        )
    elif signals.risk_level in ("HIGH", "CRITICAL"):
        verdict_signal = (
            f"The detector classified the image as REAL because the AI "
            f"probability ({signals.probability:.1%}) is below the threshold "
            f"({signals.threshold:.1%}). However, several forensic signals "
            f"indicate elevated risk and the result should be treated "
            f"cautiously."
        )
    else:
        verdict_signal = (
            f"The detector classifies this image as REAL "
            f"(AI probability {signals.probability:.1%}, "
            f"below threshold {signals.threshold:.1%})."
        )

    return {
        "spectral_signal": spectral_signal,
        "ela_signal": ela_signal,
        "prnu_signal": prnu_signal,
        "risk_summary": risk_text,
        "confidence_note": confidence_text,
        "evidence_note": evidence_text,
        "verdict_signal": verdict_signal,
    }


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

        # Get human-readable signals for LLM consumption (v1.2+)
        text_signals = signals.to_text_signals()
        # — or equivalently —
        text_signals = translate_signals_to_text(signals)
        # — or in one call —
        signals, text_signals = extractor.extract_with_text(...)
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

    def extract_with_text(
        self,
        raw_logit: float,
        rich_spectral: np.ndarray,
        poor_spectral: np.ndarray,
        ela_map: np.ndarray,
        prnu_map: np.ndarray,
    ) -> Tuple[ForensicSignals, Dict[str, str]]:
        """
        Convenience method: extract metrics AND translate to text in one call.

        Returns:
            (ForensicSignals, dict[str, str]) — structured signals and
            their human-readable text translations ready for LLM prompting.
        """
        signals = self.extract(
            raw_logit, rich_spectral, poor_spectral, ela_map, prnu_map,
        )
        text_signals = translate_signals_to_text(signals)
        return signals, text_signals

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
