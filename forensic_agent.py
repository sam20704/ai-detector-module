"""
forensic_agent.py

Forensic Reasoning Agent — Master Orchestrator Module.

Wraps the existing detector pipeline and augments it with:
  1. Interpretable forensic signal extraction (forensic_signals.py)
  2. Structured prompt construction (reasoning_prompt.py)
  3. LLM-powered forensic report generation (Azure OpenAI)
  4. Deterministic fallback when LLM is unavailable
  5. Batch analysis capability
  6. Report export (text / JSON)

Verified data flow (traced against utils.py, models.py, app.py):

  PIL.Image
      ↓
  azi_diff(PIL.Image, patch_num=128, N=256) → dict
      ├── 'total_emb': [rich (128,256), poor (128,256)]   ← spectral+positional
      ├── 'ela':       (128,128,3) float32 ∈ [0,1]        ← RGB ELA
      ├── 'noise':     (128,128) float32 ∈ [0,1]          ← Laplacian PRNU
      └── 'image_size': (H, W)
      ↓
  Tensor preparation (matches app.py exactly):
      rich  = torch.tensor(features['total_emb'][0], dtype=float32).unsqueeze(0)
      poor  = torch.tensor(features['total_emb'][1], dtype=float32).unsqueeze(0)
      ela   = torch.tensor(features['ela'],          dtype=float32).unsqueeze(0)
      noise = torch.tensor(features['noise'],        dtype=float32).unsqueeze(0)
      ↓
  model(rich, poor, ela, noise) → raw logit (B,1)
      ↓
  ForensicSignalExtractor.extract() → ForensicSignals
      ↓
  build_prompt_pair() → system + user prompts
      ↓
  Azure OpenAI → forensic narrative
      ↓
  ForensicReport (structured output)

Does NOT modify any existing detector files.

Changelog:
  v1.0 — Initial implementation
  v1.1 — Issue 1: Removed weights_only=True for PyTorch < 2.2 compat
  v1.1 — Issue 2: Fixed "Signal Agreement" → "Primary Evidence" UI label

Usage:
    # Single image analysis
    agent = ForensicAgent(checkpoint_path="checkpoints/best_model.pth")
    report = agent.analyze("claim_photo.jpg")
    print(report.report_text)

    # Batch analysis
    reports = agent.analyze_batch(["img1.jpg", "img2.jpg", "img3.jpg"])

    # CLI
    python forensic_agent.py photo.jpg --checkpoint checkpoints/best_model.pth
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import PIL.Image
from dotenv import load_dotenv
from openai import AzureOpenAI

# ── Imports from existing detector (READ-ONLY, never modified) ──
from models import TextureContrastClassifier
from utils import azi_diff

# ── Imports from reasoning layer ──
from forensic_signals import ForensicSignalExtractor, ForensicSignals
from reasoning_prompt import (
    build_prompt_pair,
    validate_prompt_completeness,
    get_prompt_metadata,
    ReportFormat,
)

# ── Environment ──
load_dotenv()

# ── Logging ──
logger = logging.getLogger("forensic_agent")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)


# ══════════════════════════════════════════════════════
#  OUTPUT CONTAINERS
# ══════════════════════════════════════════════════════

@dataclass
class ForensicReport:
    """
    Complete output of the forensic reasoning pipeline for a single image.

    Contains both structured machine-readable signals AND the
    LLM-generated (or fallback) human-readable narrative.
    """
    # ── Identification ──
    image_path: str
    case_id: Optional[str] = None
    timestamp: str = ""

    # ── Detector Results ──
    signals: Optional[ForensicSignals] = None

    # ── Report ──
    report_text: str = ""
    report_format: str = "detailed"

    # ── LLM Metadata ──
    llm_model: str = ""
    llm_used: bool = False
    prompt_version: str = ""

    # ── Pipeline Status ──
    success: bool = False
    error: Optional[str] = None
    processing_time_sec: float = 0.0
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """Serialize to nested dict for JSON export."""
        result = {
            "image_path": self.image_path,
            "case_id": self.case_id,
            "timestamp": self.timestamp,
            "signals": self.signals.to_dict() if self.signals else None,
            "report_text": self.report_text,
            "report_format": self.report_format,
            "llm_model": self.llm_model,
            "llm_used": self.llm_used,
            "prompt_version": self.prompt_version,
            "success": self.success,
            "error": self.error,
            "processing_time_sec": self.processing_time_sec,
            "warnings": self.warnings,
        }
        return result

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def save(self, output_dir: str) -> str:
        """
        Save report to disk as JSON.

        Returns the path to the saved file.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        stem = Path(self.image_path).stem
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = f"forensic_report_{stem}_{ts}.json"
        filepath = output_dir / filename

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(self.to_json())

        logger.info("Report saved: %s", filepath)
        return str(filepath)


# ══════════════════════════════════════════════════════
#  FORENSIC AGENT
# ══════════════════════════════════════════════════════

class ForensicAgent:
    """
    End-to-end forensic analysis agent.

    Orchestrates:
        Image → Feature Extraction → Model Inference → Signal Analysis
              → Prompt Construction → Azure OpenAI → Forensic Report

    The agent is stateless per-analysis: each call to analyze() is
    independent. The model and LLM client are initialized once.
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "auto",
        threshold: float = 0.7,
        report_format: ReportFormat = ReportFormat.DETAILED,
        llm_temperature: float = 0.2,
        llm_max_tokens: int = 1024,
        enable_llm: bool = True,
    ):
        """
        Initialize the forensic agent.

        Args:
            checkpoint_path: Path to TextureContrastClassifier checkpoint.
                             Must be a direct state_dict file (verified
                             against app.py loading pattern).
            device:          "cpu", "cuda", or "auto" (auto-detect).
            threshold:       Classification threshold. Higher → fewer
                             false positives. Default 0.7 matches app.py.
            report_format:   Output report format (DETAILED/SUMMARY/JSON).
            llm_temperature: Azure OpenAI temperature. Low = deterministic.
            llm_max_tokens:  Max tokens for LLM response.
            enable_llm:      If False, skip LLM and use fallback reports.
        """
        # ── Device Setup ──
        if device == "auto":
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device(device)

        self.threshold = threshold
        self.report_format = report_format
        self.llm_temperature = llm_temperature
        self.llm_max_tokens = llm_max_tokens
        self.enable_llm = enable_llm

        # ── Load Detector Model ──
        logger.info("Loading detector checkpoint: %s", checkpoint_path)
        self.model = self._load_model(checkpoint_path)
        logger.info("Detector loaded on device: %s", self.device)

        # ── Forensic Signal Extractor ──
        self.signal_extractor = ForensicSignalExtractor(threshold=threshold)

        # ── Prompt Metadata ──
        self.prompt_metadata = get_prompt_metadata()

        # ── Azure OpenAI Client ──
        self.azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "")
        self.azure_api_version = os.getenv(
            "AZURE_OPENAI_VERSION", "2024-02-15-preview"
        )
        self.llm_client: Optional[AzureOpenAI] = None

        if self.enable_llm:
            endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "")
            api_key = os.getenv("AZURE_OPENAI_KEY", "")

            if endpoint and api_key and self.azure_deployment:
                self.llm_client = AzureOpenAI(
                    azure_endpoint=endpoint,
                    api_key=api_key,
                    api_version=self.azure_api_version,
                )
                logger.info(
                    "Azure OpenAI client initialized "
                    "(deployment=%s, api_version=%s)",
                    self.azure_deployment,
                    self.azure_api_version,
                )
            else:
                missing = []
                if not endpoint:
                    missing.append("AZURE_OPENAI_ENDPOINT")
                if not api_key:
                    missing.append("AZURE_OPENAI_KEY")
                if not self.azure_deployment:
                    missing.append("AZURE_OPENAI_DEPLOYMENT")
                logger.warning(
                    "Azure OpenAI disabled — missing env vars: %s. "
                    "Will use fallback reports.",
                    ", ".join(missing),
                )
                self.enable_llm = False
        else:
            logger.info("LLM reasoning disabled by configuration.")

    # ──────────────────────────────────────────────────
    #  Model Loading
    # ──────────────────────────────────────────────────

    def _load_model(self, checkpoint_path: str) -> TextureContrastClassifier:
        """
        Load TextureContrastClassifier from saved checkpoint.

        VERIFIED against app.py (lines 12–18):
            model = TextureContrastClassifier()
            model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))

        The checkpoint file IS the state_dict directly — NOT wrapped in
        a dict with 'model_state_dict' key.
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}. "
                f"Ensure the model file exists before initializing the agent."
            )

        model = TextureContrastClassifier()

        # ────────────────────────────────────────────────
        # FIX Issue 1: Removed weights_only=True for PyTorch < 2.2 compat
        # The checkpoint is a local trusted file — no security concern.
        # Compatible with torch 1.13, 2.0, 2.1, 2.2+
        # ────────────────────────────────────────────────
        state_dict = torch.load(
            checkpoint_path,
            map_location=self.device,
        )
        # ────────────────────────────────────────────────

        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()

        # Freeze all parameters — inference only
        for param in model.parameters():
            param.requires_grad = False

        return model

    # ──────────────────────────────────────────────────
    #  Single Image Analysis (Core Pipeline)
    # ──────────────────────────────────────────────────

    def analyze(
        self,
        image_input: Union[str, PIL.Image.Image, np.ndarray],
        case_id: Optional[str] = None,
        additional_context: Optional[str] = None,
        report_format: Optional[ReportFormat] = None,
    ) -> ForensicReport:
        """
        Run the full forensic analysis pipeline on a single image.

        Accepts:
          - str: file path to image
          - PIL.Image.Image: already-loaded PIL image
          - np.ndarray: numpy RGB array (e.g., from Gradio)

        Returns:
            ForensicReport with structured signals and narrative.
        """
        start_time = time.time()
        fmt = report_format or self.report_format
        image_path_str = self._resolve_image_path(image_input)
        warnings: List[str] = []

        # ── Step 1: Load / Convert Image to PIL ──
        try:
            img_pil = self._to_pil(image_input)
        except Exception as e:
            logger.error("Image loading failed: %s", e)
            return self._error_report(
                image_path_str, f"Image loading failed: {e}", start_time
            )

        # ── Step 2: Feature Extraction ──
        try:
            features = self._extract_features(img_pil)
        except Exception as e:
            logger.error("Feature extraction failed: %s", e)
            return self._error_report(
                image_path_str, f"Feature extraction failed: {e}", start_time
            )

        # ── Step 3: Model Inference ──
        try:
            raw_logit = self._run_inference(features)
        except Exception as e:
            logger.error("Model inference failed: %s", e)
            return self._error_report(
                image_path_str, f"Inference failed: {e}", start_time
            )

        # ── Step 4: Forensic Signal Extraction ──
        try:
            signals = self._extract_signals(raw_logit, features)
        except Exception as e:
            logger.error("Signal extraction failed: %s", e)
            return self._error_report(
                image_path_str, f"Signal extraction failed: {e}", start_time
            )

        # ── Step 5: Prompt Validation ──
        validation_warnings = validate_prompt_completeness(signals)
        if validation_warnings:
            for w in validation_warnings:
                logger.warning("Prompt validation: %s", w)
            warnings.extend(validation_warnings)

            # If any CRITICAL warnings, skip LLM — signals are unreliable
            has_critical = any(
                w.startswith("CRITICAL") for w in validation_warnings
            )
            if has_critical:
                logger.error(
                    "Critical validation failures — using fallback report."
                )
                report_text = self._fallback_report(signals)
                return self._build_report(
                    image_path=image_path_str,
                    case_id=case_id,
                    signals=signals,
                    report_text=report_text,
                    fmt=fmt,
                    llm_used=False,
                    warnings=warnings,
                    start_time=start_time,
                )

        # ── Step 6: LLM Reasoning (or Fallback) ──
        llm_used = False
        if self.enable_llm and self.llm_client is not None:
            try:
                report_text = self._generate_llm_report(
                    signals=signals,
                    report_format=fmt,
                    case_id=case_id,
                    additional_context=additional_context,
                )
                llm_used = True
            except Exception as e:
                logger.warning(
                    "LLM reasoning failed — falling back to deterministic "
                    "report: %s", e,
                )
                warnings.append(f"LLM call failed: {e}")
                report_text = self._fallback_report(signals)
        else:
            report_text = self._fallback_report(signals)

        # ── Step 7: Build Final Report ──
        logger.info(
            "Analysis complete: %s | %s (%.1f%%) | %s",
            image_path_str,
            signals.verdict,
            signals.probability * 100,
            "LLM" if llm_used else "fallback",
        )

        return self._build_report(
            image_path=image_path_str,
            case_id=case_id,
            signals=signals,
            report_text=report_text,
            fmt=fmt,
            llm_used=llm_used,
            warnings=warnings,
            start_time=start_time,
        )

    # ──────────────────────────────────────────────────
    #  Batch Analysis
    # ──────────────────────────────────────────────────

    def analyze_batch(
        self,
        image_paths: List[str],
        case_id_prefix: Optional[str] = None,
        report_format: Optional[ReportFormat] = None,
        save_dir: Optional[str] = None,
    ) -> List[ForensicReport]:
        """
        Analyze multiple images sequentially.

        Args:
            image_paths:    List of file paths.
            case_id_prefix: If set, generates case IDs as
                            "{prefix}-001", "{prefix}-002", etc.
            report_format:  Override default format for batch.
            save_dir:       If set, auto-save each report to this directory.

        Returns:
            List of ForensicReport objects.
        """
        reports: List[ForensicReport] = []
        total = len(image_paths)
        logger.info("Starting batch analysis: %d images", total)

        for idx, path in enumerate(image_paths, 1):
            case_id = (
                f"{case_id_prefix}-{idx:03d}"
                if case_id_prefix else None
            )

            logger.info("Batch [%d/%d]: %s", idx, total, path)

            report = self.analyze(
                image_input=path,
                case_id=case_id,
                report_format=report_format,
            )
            reports.append(report)

            if save_dir and report.success:
                try:
                    report.save(save_dir)
                except Exception as e:
                    logger.warning(
                        "Failed to save report for %s: %s", path, e
                    )

        # ── Batch Summary ──
        successful = sum(1 for r in reports if r.success)
        ai_count = sum(
            1 for r in reports
            if r.signals and r.signals.verdict == "AI GENERATED"
        )
        logger.info(
            "Batch complete: %d/%d successful | %d flagged as AI",
            successful, total, ai_count,
        )

        return reports

    # ──────────────────────────────────────────────────
    #  Gradio Integration Helper
    # ──────────────────────────────────────────────────

    def analyze_for_gradio(
        self,
        input_img: np.ndarray,
        threshold: float = 0.7,
    ) -> tuple:
        """
        Gradio-compatible analysis method.

        Designed to be a drop-in enhancement for app.py's predict()
        function. Returns the same output types plus the forensic report.

        Args:
            input_img:  numpy RGB array from Gradio Image component.
            threshold:  classification threshold from Gradio slider.

        Returns:
            (result_html, ela_viz, noise_viz, report_text)

        Integration in app.py (future):
            from forensic_agent import ForensicAgent
            agent = ForensicAgent(checkpoint_path=MODEL_PATH)
            # In predict():
            html, ela, noise, report = agent.analyze_for_gradio(
                input_img, threshold
            )
        """
        if input_img is None:
            return (
                "Please upload an image.",
                None,
                None,
                "No image provided.",
            )

        # Temporarily update threshold if different
        original_threshold = self.threshold
        if threshold != self.threshold:
            self.threshold = threshold
            self.signal_extractor = ForensicSignalExtractor(
                threshold=threshold
            )

        try:
            # Run full analysis (extracts features internally)
            report = self.analyze(image_input=input_img)

            if not report.success:
                return (
                    f"<div style='color:red'>"
                    f"Analysis failed: {report.error}</div>",
                    None,
                    None,
                    report.report_text,
                )

            signals = report.signals

            # ── Format HTML (matches app.py style) ──
            is_ai = signals.verdict == "AI GENERATED"
            label = (
                "🚨 AI GENERATED or EDITED" if is_ai
                else "✅ REAL PHOTOGRAPH"
            )

            # Confidence calculation matching app.py logic exactly
            prob = signals.probability
            if is_ai:
                confidence = (prob - threshold) / (1.0 - threshold)
            else:
                confidence = (threshold - prob) / threshold
            confidence = max(0.0, min(1.0, confidence))

            color = "red" if is_ai else "green"

            # ────────────────────────────────────────────
            # FIX Issue 2: "Signal Agreement" → "Primary Evidence"
            # The UI label was misleading — primary_evidence identifies
            # the most decisive modality, not signal agreement status.
            # ────────────────────────────────────────────
            # Format primary evidence for display
            primary_display = signals.primary_evidence.replace(
                '_', ' '
            ).title()
            if signals.primary_evidence == "multiple_signals":
                primary_display = "Multiple Signals (Tied)"

            result_html = f"""
            <div style="text-align: center; padding: 15px;
                        border-radius: 10px;
                        background-color: rgba(0,0,0,0.05);
                        border: 2px solid {color};">
                <h2 style="color: {color}; margin-bottom: 5px;">
                    {label}
                </h2>
                <p style="font-size: 1.2em;">
                    Forensic Confidence:
                    <b>{confidence * 100:.2f}%</b>
                </p>
                <p style="font-size: 0.9em; color: gray;">
                    (Probability: {prob:.4f} | Threshold: {threshold})
                </p>
                <p style="font-size: 0.85em; color: gray;">
                    Risk: {signals.risk_level} |
                    Primary Evidence: {primary_display}
                </p>
            </div>
            """
            # ────────────────────────────────────────────

            # ── ELA / PRNU Visualizations ──
            # Re-extract features for visualization
            # (we need the raw maps, not tensors)
            img_pil = self._to_pil(input_img)
            features = self._extract_features(img_pil)

            ela_viz = (features["ela"] * 255).astype(np.uint8)

            noise_raw = features["noise"]
            noise_min = noise_raw.min()
            noise_max = noise_raw.max()
            noise_viz = (
                (noise_raw - noise_min)
                / (noise_max - noise_min + 1e-8)
                * 255
            ).astype(np.uint8)

            return result_html, ela_viz, noise_viz, report.report_text

        finally:
            # Restore original threshold
            if threshold != original_threshold:
                self.threshold = original_threshold
                self.signal_extractor = ForensicSignalExtractor(
                    threshold=original_threshold
                )

    # ══════════════════════════════════════════════════
    #  INTERNAL PIPELINE METHODS
    # ══════════════════════════════════════════════════

    # ── Image Input Handling ──────────────────────────

    @staticmethod
    def _to_pil(
        image_input: Union[str, PIL.Image.Image, np.ndarray],
    ) -> PIL.Image.Image:
        """
        Convert any supported input type to PIL.Image.Image in RGB mode.

        This matches the input type that azi_diff() expects:
            azi_diff(img: PIL.Image.Image, ...)
        """
        if isinstance(image_input, PIL.Image.Image):
            return image_input.convert("RGB")
        elif isinstance(image_input, np.ndarray):
            # Gradio provides numpy RGB arrays
            return PIL.Image.fromarray(image_input).convert("RGB")
        elif isinstance(image_input, str):
            if not os.path.exists(image_input):
                raise FileNotFoundError(
                    f"Image not found: {image_input}"
                )
            return PIL.Image.open(image_input).convert("RGB")
        else:
            raise TypeError(
                f"Unsupported image input type: {type(image_input)}. "
                f"Expected str, PIL.Image, or np.ndarray."
            )

    @staticmethod
    def _resolve_image_path(
        image_input: Union[str, PIL.Image.Image, np.ndarray],
    ) -> str:
        """Get a string identifier for the image source."""
        if isinstance(image_input, str):
            return image_input
        elif isinstance(image_input, PIL.Image.Image):
            return (
                getattr(image_input, "filename", "<PIL.Image>")
                or "<PIL.Image>"
            )
        else:
            return "<numpy_array>"

    # ── Feature Extraction ────────────────────────────

    def _extract_features(self, img_pil: PIL.Image.Image) -> Dict:
        """
        Call azi_diff() and return the raw feature dictionary.

        VERIFIED against utils.py azi_diff() signature:
            def azi_diff(img: PIL.Image.Image, patch_num=128, N=256) → dict

        VERIFIED return keys:
            'total_emb':  [rich_array (128,256), poor_array (128,256)]
            'ela':        (128,128,3) float32 in [0,1]
            'noise':      (128,128) float32 in [0,1]
            'image_size': (H, W) tuple
        """
        features = azi_diff(img_pil, patch_num=128, N=256)

        # Sanity check: verify expected keys are present
        required_keys = {"total_emb", "ela", "noise", "image_size"}
        actual_keys = set(features.keys())
        missing = required_keys - actual_keys
        if missing:
            raise KeyError(
                f"azi_diff() returned dict missing keys: {missing}. "
                f"Got: {actual_keys}"
            )

        return features

    # ── Model Inference ───────────────────────────────

    def _run_inference(self, features: Dict) -> float:
        """
        Prepare tensors and run model forward pass.

        VERIFIED against app.py predict() function (lines 24–33):

            rich  = torch.tensor(features['total_emb'][0],
                                 dtype=torch.float32).unsqueeze(0).to(DEVICE)
            poor  = torch.tensor(features['total_emb'][1],
                                 dtype=torch.float32).unsqueeze(0).to(DEVICE)
            ela   = torch.tensor(features['ela'],
                                 dtype=torch.float32).unsqueeze(0).to(DEVICE)
            noise = torch.tensor(features['noise'],
                                 dtype=torch.float32).unsqueeze(0).to(DEVICE)

            output = model(rich, poor, ela, noise)

        VERIFIED against models.py forward(self, rich, poor, ela, noise):
            rich:  (B, 128, 256) → permute(1,0,2) → Attention
            poor:  (B, 128, 256) → permute(1,0,2) → Attention
            ela:   (B, 128, 128, 3) → permute(0,3,1,2) → (B,3,128,128)
            noise: (B, 128, 128) → unsqueeze(1) → (B,1,128,128)
            visual = cat([ela_permuted, noise_unsqueezed], dim=1)
                   → (B,4,128,128) → CNN → (B,2048)
        """
        # ── Tensor preparation (exact match to app.py) ──
        rich_tensor = torch.tensor(
            features["total_emb"][0], dtype=torch.float32
        ).unsqueeze(0).to(self.device)
        # Shape: (1, 128, 256)

        poor_tensor = torch.tensor(
            features["total_emb"][1], dtype=torch.float32
        ).unsqueeze(0).to(self.device)
        # Shape: (1, 128, 256)

        ela_tensor = torch.tensor(
            features["ela"], dtype=torch.float32
        ).unsqueeze(0).to(self.device)
        # Shape: (1, 128, 128, 3)

        noise_tensor = torch.tensor(
            features["noise"], dtype=torch.float32
        ).unsqueeze(0).to(self.device)
        # Shape: (1, 128, 128)

        # ── Forward pass ──
        with torch.no_grad():
            output = self.model(
                rich_tensor, poor_tensor, ela_tensor, noise_tensor
            )
            # output shape: (1, 1)

        # Extract scalar logit
        raw_logit = output.squeeze().cpu().item()

        return float(raw_logit)

    # ── Signal Extraction ─────────────────────────────

    def _extract_signals(
        self, raw_logit: float, features: Dict,
    ) -> ForensicSignals:
        """
        Compute interpretable forensic signals from raw features
        and detector output.

        Input arrays match azi_diff() return format:
            rich_spectral = features['total_emb'][0]  → (128, 256)
            poor_spectral = features['total_emb'][1]  → (128, 256)
            ela_map       = features['ela']            → (128, 128, 3)
            prnu_map      = features['noise']          → (128, 128)
        """
        signals = self.signal_extractor.extract(
            raw_logit=raw_logit,
            rich_spectral=np.asarray(features["total_emb"][0]),
            poor_spectral=np.asarray(features["total_emb"][1]),
            ela_map=np.asarray(features["ela"]),
            prnu_map=np.asarray(features["noise"]),
        )

        return signals

    # ── LLM Report Generation ─────────────────────────

    def _generate_llm_report(
        self,
        signals: ForensicSignals,
        report_format: ReportFormat,
        case_id: Optional[str] = None,
        additional_context: Optional[str] = None,
    ) -> str:
        """
        Build prompts and call Azure OpenAI for forensic reasoning.

        Uses build_prompt_pair() from reasoning_prompt.py which
        generates format-specific system + user prompts.
        """
        # Build prompt pair
        prompts = build_prompt_pair(
            signals=signals,
            report_format=report_format,
            case_id=case_id,
            additional_context=additional_context,
        )

        logger.info(
            "Calling Azure OpenAI (deployment=%s, format=%s)…",
            self.azure_deployment,
            report_format.value,
        )

        # Adjust max_tokens for summary format
        max_tokens = self.llm_max_tokens
        if report_format == ReportFormat.SUMMARY:
            max_tokens = min(max_tokens, 512)

        response = self.llm_client.chat.completions.create(
            model=self.azure_deployment,
            messages=[
                {"role": "system", "content": prompts["system"]},
                {"role": "user", "content": prompts["user"]},
            ],
            temperature=self.llm_temperature,
            max_tokens=max_tokens,
            top_p=0.95,
        )

        report_text = response.choices[0].message.content.strip()

        # Log token usage
        usage = response.usage
        if usage:
            logger.info(
                "LLM tokens — prompt: %d, completion: %d, total: %d",
                usage.prompt_tokens,
                usage.completion_tokens,
                usage.total_tokens,
            )

        logger.info("LLM report generated (%d chars)", len(report_text))
        return report_text

    # ── Fallback Report (Deterministic, No LLM) ──────

    @staticmethod
    def _fallback_report(signals: ForensicSignals) -> str:
        """
        Generate a structured forensic report without LLM.

        Uses only the computed metrics from ForensicSignals.
        Fully deterministic: same signals → same report.
        Used when:
          - LLM is disabled
          - Azure OpenAI call fails
          - Critical validation warnings are present
        """
        sp = signals.spectral
        ela = signals.ela
        prnu = signals.prnu

        # Format primary evidence display
        if signals.primary_evidence == "multiple_signals":
            primary_display = "Multiple Signals (Tied)"
        else:
            primary_display = signals.primary_evidence.replace(
                '_', ' '
            ).title()

        # Spectral anomaly severity label
        if sp.anomaly_score >= 0.7:
            sp_severity = "HIGH"
        elif sp.anomaly_score >= 0.4:
            sp_severity = "MODERATE"
        else:
            sp_severity = "LOW"

        # ELA uniformity description
        ela_uniform_desc = (
            "uniform" if ela.uniformity_score > 0.7 else "varied"
        )

        # ELA splicing description
        if ela.splicing_indicator >= 0.4:
            splice_desc = "evidence of localized manipulation"
        else:
            splice_desc = "no splicing detected"

        # PRNU flatness description
        if prnu.spectral_flatness > 0.7:
            prnu_flat_desc = "white noise / AI-like"
        else:
            prnu_flat_desc = "structured / camera-like"

        lines = [
            f"VERDICT: {signals.verdict} "
            f"({signals.probability * 100:.1f}% probability)",
            "",
            "EVIDENCE SUMMARY",
            "",
            "• Spectral Analysis:",
            f"  Anomaly score: {sp.anomaly_score} "
            f"({sp_severity} anomaly)",
            f"  Rich HF ratio: {sp.rich_high_freq_ratio} | "
            f"Poor HF ratio: {sp.poor_high_freq_ratio}",
            f"  Rich diversity: {sp.rich_spectral_diversity} | "
            f"Poor diversity: {sp.poor_spectral_diversity}",
            "",
            "• Compression Analysis (ELA):",
            f"  Uniformity: {ela.uniformity_score} "
            f"({ela_uniform_desc} compression)",
            f"  Splicing indicator: {ela.splicing_indicator} "
            f"({splice_desc})",
            f"  Spatial entropy: {ela.spatial_entropy}",
            "",
            "• Sensor Fingerprint (PRNU):",
            f"  Strength: {prnu.strength_score} | "
            f"Camera consistency: {prnu.camera_consistency}",
            f"  Spectral flatness: {prnu.spectral_flatness} "
            f"({prnu_flat_desc})",
            "",
            f"RISK ASSESSMENT: {signals.risk_level}",
            "",
            f"CONFIDENCE: {signals.confidence_level}",
            "",
            f"PRIMARY EVIDENCE: {primary_display}",
            "",
            "RECOMMENDATION:",
        ]

        # Risk-appropriate recommendations
        if signals.risk_level in ("CRITICAL", "HIGH"):
            lines.append(
                "  Flag for immediate manual forensic review. "
                "Do not approve claim without expert validation."
            )
        elif signals.risk_level == "MEDIUM":
            lines.append(
                "  Recommend secondary review. Evidence is inconclusive — "
                "request additional documentation from claimant."
            )
        else:
            lines.append(
                "  No immediate forensic concerns detected. "
                "Standard claim processing may proceed."
            )

        lines.extend([
            "",
            "─── NOTE: This report was generated without LLM reasoning. ───",
            "─── Metrics above are computed deterministically from the   ───",
            "─── detector's signal extraction pipeline.                  ───",
        ])

        return "\n".join(lines)

    # ── Report Builder ────────────────────────────────

    def _build_report(
        self,
        image_path: str,
        case_id: Optional[str],
        signals: ForensicSignals,
        report_text: str,
        fmt: ReportFormat,
        llm_used: bool,
        warnings: List[str],
        start_time: float,
    ) -> ForensicReport:
        """Assemble the final ForensicReport dataclass."""
        return ForensicReport(
            image_path=image_path,
            case_id=case_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            signals=signals,
            report_text=report_text,
            report_format=fmt.value,
            llm_model=self.azure_deployment if llm_used else "",
            llm_used=llm_used,
            prompt_version=self.prompt_metadata.get(
                "prompt_version", ""
            ),
            success=True,
            error=None,
            processing_time_sec=round(time.time() - start_time, 3),
            warnings=warnings,
        )

    # ── Error Report ──────────────────────────────────

    @staticmethod
    def _error_report(
        image_path: str,
        error_msg: str,
        start_time: float,
    ) -> ForensicReport:
        """Build a ForensicReport representing a pipeline failure."""
        return ForensicReport(
            image_path=image_path,
            timestamp=datetime.now(timezone.utc).isoformat(),
            report_text=f"ANALYSIS FAILED: {error_msg}",
            success=False,
            error=error_msg,
            processing_time_sec=round(time.time() - start_time, 3),
        )


# ══════════════════════════════════════════════════════
#  CLI ENTRY POINT
# ══════════════════════════════════════════════════════

def main():
    """Command-line interface for single and batch forensic analysis."""
    import argparse

    parser = argparse.ArgumentParser(
        description="🛡️ Forensic AI Image Analysis Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:

  Single image:
    python forensic_agent.py photo.jpg \\
        --checkpoint checkpoints/best_model.pth

  Batch analysis:
    python forensic_agent.py img1.jpg img2.jpg img3.jpg \\
        --checkpoint checkpoints/best_model.pth \\
        --batch-prefix INS-2024

  Save reports to disk:
    python forensic_agent.py photo.jpg \\
        --checkpoint checkpoints/best_model.pth \\
        --save-dir ./reports

  Without LLM (deterministic fallback):
    python forensic_agent.py photo.jpg \\
        --checkpoint checkpoints/best_model.pth \\
        --no-llm
        """,
    )

    parser.add_argument(
        "images",
        nargs="+",
        type=str,
        help="Path(s) to image file(s) for analysis.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint (state_dict .pth file).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Compute device (default: auto-detect).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.7,
        help="Classification threshold (default: 0.7).",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="detailed",
        choices=["detailed", "summary", "json"],
        help="Report format (default: detailed).",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Disable LLM reasoning — use deterministic fallback.",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="Directory to save JSON reports.",
    )
    parser.add_argument(
        "--batch-prefix",
        type=str,
        default=None,
        help="Case ID prefix for batch analysis (e.g., INS-2024).",
    )

    args = parser.parse_args()

    # ── Map format string to enum ──
    format_map = {
        "detailed": ReportFormat.DETAILED,
        "summary": ReportFormat.SUMMARY,
        "json": ReportFormat.STRUCTURED_JSON,
    }
    report_format = format_map[args.format]

    # ── Initialize Agent ──
    try:
        agent = ForensicAgent(
            checkpoint_path=args.checkpoint,
            device=args.device,
            threshold=args.threshold,
            report_format=report_format,
            enable_llm=not args.no_llm,
        )
    except Exception as e:
        logger.error("Failed to initialize agent: %s", e)
        raise SystemExit(1)

    # ── Run Analysis ──
    if len(args.images) == 1:
        # ── Single Image Mode ──
        report = agent.analyze(
            image_input=args.images[0],
            case_id=args.batch_prefix,
        )

        print(f"\n{'=' * 60}")
        print(" FORENSIC ANALYSIS REPORT")
        print(f"{'=' * 60}")
        print(report.report_text)
        print(f"{'=' * 60}")
        print(f" Processing time: {report.processing_time_sec:.2f}s")
        print(f" LLM used: {report.llm_used}")
        if report.warnings:
            print(f" Warnings: {len(report.warnings)}")
            for w in report.warnings:
                print(f"   ⚠ {w}")
        print(f"{'=' * 60}")

        if args.save_dir:
            saved_path = report.save(args.save_dir)
            print(f"\n📁 Report saved: {saved_path}")

    else:
        # ── Batch Mode ──
        reports = agent.analyze_batch(
            image_paths=args.images,
            case_id_prefix=args.batch_prefix,
            report_format=report_format,
            save_dir=args.save_dir,
        )

        # ── Batch Summary Table ──
        print(f"\n{'=' * 80}")
        print(" BATCH ANALYSIS SUMMARY")
        print(f"{'=' * 80}")
        print(
            f"{'#':<4} {'Image':<30} {'Verdict':<18} "
            f"{'Prob':>6} {'Risk':<10} {'Time':>6}"
        )
        print("-" * 80)

        for i, r in enumerate(reports, 1):
            if r.success and r.signals:
                s = r.signals
                print(
                    f"{i:<4} "
                    f"{Path(r.image_path).name:<30} "
                    f"{s.verdict:<18} "
                    f"{s.probability * 100:>5.1f}% "
                    f"{s.risk_level:<10} "
                    f"{r.processing_time_sec:>5.2f}s"
                )
            else:
                print(
                    f"{i:<4} "
                    f"{Path(r.image_path).name:<30} "
                    f"{'FAILED':<18} "
                    f"{'—':>6} "
                    f"{'—':<10} "
                    f"{r.processing_time_sec:>5.2f}s"
                )

        print("-" * 80)
        total_time = sum(r.processing_time_sec for r in reports)
        success_count = sum(1 for r in reports if r.success)
        ai_count = sum(
            1 for r in reports
            if r.signals and r.signals.verdict == "AI GENERATED"
        )
        print(
            f"Total: {success_count}/{len(reports)} successful | "
            f"{ai_count} flagged AI | "
            f"{total_time:.2f}s total"
        )
        print(f"{'=' * 80}")

        # ── Print Individual Reports ──
        for i, r in enumerate(reports, 1):
            if r.success:
                print(f"\n{'─' * 60}")
                print(f" Report #{i}: {Path(r.image_path).name}")
                print(f"{'─' * 60}")
                print(r.report_text)


if __name__ == "__main__":
    main()
