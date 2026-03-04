"""
reasoning_prompt.py

Master Prompt Construction Module for Forensic Reasoning.

Builds structured system and user prompts consumed by Azure OpenAI
for generating forensic image analysis reports.

Design Principles:
  1. DETERMINISTIC: Same ForensicSignals → same prompt (no randomness).
  2. CONSTRAINED: LLM cannot override detector verdict or fabricate metrics.
  3. GROUNDED: Every forensic claim must trace to a provided metric.
  4. AUDITABLE: Prompt version tracked for legal/compliance reproducibility.
  5. MODULAR: Multiple report formats (detailed, summary, JSON) supported.

Verified compatibility:
  - Consumes ForensicSignals from forensic_signals.py
  - All field references match SpectralMetrics, ELAMetrics, PRNUMetrics dataclasses
  - Handles "multiple_signals" primary_evidence value (Issue 3 tie handling)
  - No imports from models.py or utils.py (prompt layer is detector-agnostic)

Does NOT modify any existing detector files.

Changelog:
  v2.0.0 — Initial production version
  v2.1.0 — Updated for Issue 3: "multiple_signals" handling in validation
            and system prompt edge case instructions
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

from forensic_signals import (
    ForensicSignals,
    SpectralMetrics,
    ELAMetrics,
    PRNUMetrics,
)


# ──────────────────────────────────────────────────────
# Prompt Versioning (for audit trails)
# ──────────────────────────────────────────────────────

PROMPT_VERSION = "2.1.0"
PROMPT_SCHEMA = "forensic-reasoning-v2"


# ──────────────────────────────────────────────────────
# Report Format Enum
# ──────────────────────────────────────────────────────

class ReportFormat(Enum):
    """Supported forensic report output formats."""
    DETAILED = "detailed"       # Full narrative with all evidence
    SUMMARY = "summary"         # Condensed investigator brief
    STRUCTURED_JSON = "json"    # Machine-parseable JSON report


# ──────────────────────────────────────────────────────
# Valid primary_evidence values
# ──────────────────────────────────────────────────────

VALID_PRIMARY_EVIDENCE = {
    "spectral_analysis",
    "compression_analysis",
    "sensor_fingerprint",
    "multiple_signals",       # Issue 3: tie handling
}


# ──────────────────────────────────────────────────────
# Signal Agreement Pre-Computation
# ──────────────────────────────────────────────────────

@dataclass
class SignalAgreement:
    """
    Pre-computed agreement analysis between forensic modalities.

    Computed BEFORE the LLM call to provide structured context
    rather than asking the LLM to infer agreement from raw numbers.
    """
    spectral_vote: str          # "AI" | "REAL" | "INCONCLUSIVE"
    ela_vote: str               # "AI" | "REAL" | "INCONCLUSIVE"
    prnu_vote: str              # "AI" | "REAL" | "INCONCLUSIVE"
    agreement_status: str       # "UNANIMOUS" | "MAJORITY" | "SPLIT" | "INCONCLUSIVE"
    agreeing_count: int         # Number of modalities agreeing with verdict
    dissenting_signals: List[str]   # Names of signals that disagree
    narrative_hint: str         # One-line context for LLM


def compute_signal_agreement(signals: ForensicSignals) -> SignalAgreement:
    """
    Analyze whether the three forensic modalities corroborate each other.

    Thresholds for per-modality voting:
      - Spectral:  anomaly_score ≥ 0.5 → votes AI
      - ELA:       splicing_indicator ≥ 0.4 → votes AI
      - PRNU:      strength_score ≤ 0.3 → votes AI (absent fingerprint)

    Inconclusive zone: values within ±0.1 of threshold.

    This is computed deterministically — no LLM involvement.
    """
    sp = signals.spectral
    ela = signals.ela
    prnu = signals.prnu

    # ── Per-modality votes ──
    def _vote(value: float, threshold: float, invert: bool = False) -> str:
        """Vote AI/REAL/INCONCLUSIVE based on threshold with dead zone."""
        if invert:
            value = 1.0 - value
        if value >= threshold + 0.1:
            return "AI"
        elif value <= threshold - 0.1:
            return "REAL"
        return "INCONCLUSIVE"

    spectral_vote = _vote(sp.anomaly_score, 0.5)
    ela_vote = _vote(ela.splicing_indicator, 0.4)
    prnu_vote = _vote(prnu.strength_score, 0.3, invert=True)

    votes = [spectral_vote, ela_vote, prnu_vote]
    vote_names = ["spectral_analysis", "compression_analysis", "sensor_fingerprint"]

    # ── Agreement analysis ──
    is_ai_verdict = signals.verdict == "AI GENERATED"
    expected_vote = "AI" if is_ai_verdict else "REAL"

    agreeing = [
        name for v, name in zip(votes, vote_names)
        if v == expected_vote
    ]
    dissenting = [
        name for v, name in zip(votes, vote_names)
        if v != expected_vote and v != "INCONCLUSIVE"
    ]
    inconclusive = [
        name for v, name in zip(votes, vote_names)
        if v == "INCONCLUSIVE"
    ]

    agreeing_count = len(agreeing)
    definitive_votes = [v for v in votes if v != "INCONCLUSIVE"]

    if len(definitive_votes) == 0:
        status = "INCONCLUSIVE"
        hint = "All three modalities fall in ambiguous ranges — low interpretive certainty."
    elif all(v == definitive_votes[0] for v in definitive_votes):
        if len(definitive_votes) == 3:
            status = "UNANIMOUS"
            hint = "All three forensic modalities independently corroborate the verdict."
        else:
            status = "MAJORITY"
            hint = (
                f"{len(definitive_votes)} of 3 modalities agree; "
                f"{len(inconclusive)} inconclusive."
            )
    elif agreeing_count >= 2:
        status = "MAJORITY"
        hint = (
            f"{agreeing_count} modalities support the verdict; "
            f"{', '.join(dissenting)} dissent."
        )
    else:
        status = "SPLIT"
        hint = (
            "Forensic modalities provide conflicting signals — "
            "manual expert review strongly recommended."
        )

    return SignalAgreement(
        spectral_vote=spectral_vote,
        ela_vote=ela_vote,
        prnu_vote=prnu_vote,
        agreement_status=status,
        agreeing_count=agreeing_count,
        dissenting_signals=dissenting,
        narrative_hint=hint,
    )


# ══════════════════════════════════════════════════════
#  SYSTEM PROMPTS
# ══════════════════════════════════════════════════════

# ── Detailed Report System Prompt ────────────────────

SYSTEM_PROMPT_DETAILED = f"""\
You are a **Senior Forensic Image Analyst** embedded in an automated insurance \
claim verification system. Your forensic reports may be used as supporting \
evidence in fraud investigations and legal proceedings.

Prompt schema: {PROMPT_SCHEMA} | Version: {PROMPT_VERSION}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 ROLE AND AUTHORITY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

You receive structured forensic signal data from a multi-modal AI image \
detection system. The detector has ALREADY classified the image. Your role \
is strictly to INTERPRET and EXPLAIN the detector's findings. You are an \
analyst, not a classifier.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 ABSOLUTE RULES (VIOLATION = REPORT REJECTION)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. NEVER override, contradict, or re-interpret the detector's verdict.
   The verdict and probability are GROUND TRUTH from the neural network.
2. NEVER fabricate, invent, or hallucinate metric values.
   You may ONLY reference metrics explicitly provided in the data payload.
3. NEVER speculate about image content, identity, location, or context.
   You analyze SIGNALS, not semantics.
4. NEVER use hedging language that undermines the detector when confidence \
   is HIGH. State findings directly.
5. When confidence is LOW or signals DISAGREE, you MUST explicitly flag \
   uncertainty. Do not paper over ambiguity.
6. NEVER add preamble, sign-off, apology, or commentary outside the \
   report template.
7. Every interpretive statement MUST be traceable to a specific metric \
   in the provided data.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 FORENSIC DOMAIN KNOWLEDGE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

You must apply the following domain knowledge when interpreting metrics:

## Spectral Analysis (Fourier Domain)
The system divides the image into 16×16 patches, sorted into "rich texture" \
(edges, grass, complex structures) and "poor texture" (sky, walls, smooth \
surfaces) groups. Each patch undergoes 2D FFT → azimuthal integration to \
produce a 256-dimensional radial frequency profile.

Key interpretive rules:
- **High-frequency ratio** measures energy in the upper half of the \
  frequency spectrum. Real photographs from physical cameras contain \
  natural high-frequency content from sensor noise, lens effects, and \
  scene complexity. AI generators (GANs, diffusion models) often exhibit \
  attenuated or unnatural high-frequency patterns.
- **Spectral diversity** measures how much the frequency profiles vary \
  across patches. Natural images have diverse patch spectra because \
  different scene regions have different frequency characteristics. \
  AI-generated images often show suspiciously uniform spectral signatures.
- **Anomaly score** is a composite of HF attenuation and low diversity. \
  Score ≥ 0.7: strong spectral anomaly. Score 0.4–0.7: moderate. \
  Score < 0.4: appears spectrally normal.
- **Rich/Poor energy ratio** indicates texture contrast. Extremely high \
  or low ratios can indicate unnatural texture distribution.

## Error Level Analysis (ELA)
ELA re-saves the image at a fixed JPEG quality and measures the pixel-level \
difference. Regions that have been edited, spliced, or generated differently \
will show inconsistent error levels compared to the rest of the image.

Key interpretive rules:
- **Uniformity score** near 1.0 means compression errors are evenly \
  distributed — consistent with a single-source image (either entirely \
  real OR entirely AI-generated). This does NOT prove authenticity.
- **Uniformity score** well below 1.0 means some regions compress \
  differently — possible splicing, inpainting, or localized editing.
- **Splicing indicator** measures localized variance anomalies. \
  Score ≥ 0.5: strong evidence of localized manipulation. \
  Score 0.2–0.5: mild inconsistency. Score < 0.2: no splicing detected.
- **Spatial entropy** reflects the complexity of the error landscape. \
  Very low entropy suggests artificial uniformity.

## Sensor Noise Fingerprint (PRNU)
Physical camera sensors imprint a unique Photo Response Non-Uniformity \
(PRNU) pattern — a consistent high-frequency noise signature caused by \
manufacturing variations in sensor pixels. This acts as a "fingerprint" \
for the physical device.

Key interpretive rules:
- **Strength score** near 1.0: strong, structured noise consistent with \
  a physical camera sensor.
- **Strength score** near 0.0: noise is absent or unstructured — \
  inconsistent with physical camera capture.
- **Spectral flatness** near 1.0: noise resembles white noise (flat \
  power spectrum) — typical of AI-generated images or heavy denoising. \
  Near 0.0: noise has spectral structure — typical of real sensor noise.
- **Camera consistency = "absent"**: the image almost certainly did not \
  originate from a physical camera sensor.
- **Camera consistency = "consistent"**: strong evidence of physical \
  camera origin, BUT does not rule out post-capture manipulation.

## Signal Agreement
When modalities agree, findings are more reliable. When they disagree, \
the report MUST acknowledge the conflict and recommend caution.

## Primary Evidence
When the primary evidence is listed as "multiple_signals", it means two \
or more forensic modalities produced equally strong suspicion scores. \
In this case, do not single out one modality — describe the combined \
evidence pattern instead.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 OUTPUT TEMPLATE (EXACT FORMAT REQUIRED)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Respond ONLY with the following template, filled in based on the data:
VERDICT: (% probability)

EVIDENCE SUMMARY

• Spectral Analysis:
<2–3 sentences interpreting spectral metrics. Reference specific numbers.>

• Compression Analysis (ELA):
<2–3 sentences interpreting ELA metrics. Reference specific numbers.>

• Sensor Fingerprint (PRNU):
<2–3 sentences interpreting PRNU metrics. Reference specific numbers.>

SIGNAL AGREEMENT: <agreement_status>
<1–2 sentences on whether modalities corroborate. Note any dissenting signals.>

RISK ASSESSMENT: <risk_level>
<1 sentence justifying the risk level based on the evidence above.>

CONFIDENCE: <confidence_level>
<1 sentence explaining what drives confidence up or down.>

PRIMARY EVIDENCE: <primary_evidence_modality>
<1 sentence identifying which signal was most decisive and why.
If "multiple_signals", describe the combined evidence pattern.>

RECOMMENDATION:
<2–3 actionable sentences for the insurance fraud investigator.
Include specific next steps based on the risk level and evidence pattern.>

text

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 EDGE CASE HANDLING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

- If probability is between 0.45 and 0.55: explicitly state the model \
  is near its decision boundary and findings are unreliable without \
  additional evidence.
- If confidence is LOW: recommend manual expert review regardless of verdict.
- If signals SPLIT: do not present the verdict as certain. Frame it as \
  "the detector leans toward X, but forensic signals are mixed."
- If PRNU is consistent but spectral anomaly is high: note that the image \
  may be a real photograph that has been heavily post-processed.
- If ELA splicing is high but overall verdict is REAL: flag possible \
  localized editing on an otherwise authentic image.
- If primary evidence is "multiple_signals": describe the converging \
  evidence from all tied modalities rather than picking one.
"""


# ── Summary Report System Prompt ─────────────────────

SYSTEM_PROMPT_SUMMARY = f"""\
You are a forensic image analyst producing BRIEF investigation summaries \
for insurance claim triage.

Prompt schema: {PROMPT_SCHEMA} | Version: {PROMPT_VERSION}

RULES:
1. Never override the detector's verdict or probability.
2. Never fabricate metrics — only reference provided data.
3. Keep the entire response under 150 words.
4. If primary evidence is "multiple_signals", mention all contributing modalities.

OUTPUT FORMAT (exact):
VERDICT: (% probability)
RISK: <risk_level> | CONFIDENCE: <confidence_level>

KEY FINDINGS:
• <Most important signal finding — 1 sentence>
• <Second most important finding — 1 sentence>
• <Signal agreement status — 1 sentence>

ACTION: <1 sentence recommendation>

text
"""


# ── Structured JSON System Prompt ────────────────────

SYSTEM_PROMPT_JSON = f"""\
You are a forensic signal interpreter. Produce a machine-parseable JSON \
forensic report from the provided signal data.

Prompt schema: {PROMPT_SCHEMA} | Version: {PROMPT_VERSION}

RULES:
1. Never override the detector's verdict.
2. Never fabricate metrics.
3. Output ONLY valid JSON — no markdown fences, no commentary.
4. If primary_evidence is "multiple_signals", list all tied modalities in the \
   "primary_evidence" field as a comma-separated string.

JSON SCHEMA:
{{
  "verdict": "<string>",
  "probability_pct": <float>,
  "risk_level": "<string>",
  "confidence_level": "<string>",
  "evidence": {{
    "spectral": {{
      "interpretation": "<string: 1-2 sentences>",
      "anomaly_detected": <boolean>,
      "key_metric": "<string: most relevant metric name and value>"
    }},
    "ela": {{
      "interpretation": "<string: 1-2 sentences>",
      "splicing_detected": <boolean>,
      "key_metric": "<string>"
    }},
    "prnu": {{
      "interpretation": "<string: 1-2 sentences>",
      "camera_signature_present": <boolean>,
      "key_metric": "<string>"
    }}
  }},
  "signal_agreement": "<UNANIMOUS|MAJORITY|SPLIT|INCONCLUSIVE>",
  "primary_evidence": "<string>",
  "recommendation": "<string: 1-2 sentences>"
}}
"""


# Prompt registry for format selection
_SYSTEM_PROMPTS: Dict[ReportFormat, str] = {
    ReportFormat.DETAILED: SYSTEM_PROMPT_DETAILED,
    ReportFormat.SUMMARY: SYSTEM_PROMPT_SUMMARY,
    ReportFormat.STRUCTURED_JSON: SYSTEM_PROMPT_JSON,
}


# ══════════════════════════════════════════════════════
#  USER PROMPT BUILDER
# ══════════════════════════════════════════════════════

def build_user_prompt(
    signals: ForensicSignals,
    report_format: ReportFormat = ReportFormat.DETAILED,
    case_id: Optional[str] = None,
    additional_context: Optional[str] = None,
) -> str:
    """
    Build the user-turn prompt containing all forensic signal data.

    This prompt is DETERMINISTIC: same ForensicSignals → same prompt.
    All metrics are presented as raw data without interpretation,
    leaving reasoning entirely to the LLM.

    Args:
        signals:            Complete forensic signal package from detector.
        report_format:      Desired output format (affects instruction phrasing).
        case_id:            Optional insurance claim identifier for traceability.
        additional_context: Optional investigator notes (appended verbatim).

    Returns:
        Formatted user prompt string.
    """
    sp = signals.spectral
    ela = signals.ela
    prnu = signals.prnu

    # Pre-compute signal agreement (deterministic, no LLM)
    agreement = compute_signal_agreement(signals)

    # ── Header ──
    sections: List[str] = []

    if case_id:
        sections.append(f"CASE ID: {case_id}")

    sections.append(
        f"Analyze the following forensic signal data and produce a "
        f"{report_format.value} forensic report."
    )

    # ── Detector Output ──
    sections.append(_section(
        "DETECTOR OUTPUT",
        f"Verdict           : {signals.verdict}",
        f"Probability (AI)  : {signals.probability * 100:.1f}%",
        f"Raw Logit         : {signals.raw_logit}",
        f"Threshold Used    : {signals.threshold}",
    ))

    # ── Spectral Analysis ──
    sections.append(_section(
        "SPECTRAL ANALYSIS (Fourier Domain)",
        f"Rich Patch Mean Energy         : {sp.rich_mean_energy}",
        f"Poor Patch Mean Energy         : {sp.poor_mean_energy}",
        f"Rich/Poor Energy Ratio         : {sp.rich_poor_energy_ratio}",
        f"Rich High-Frequency Ratio      : {sp.rich_high_freq_ratio}  "
        f"[0.0=no HF, 1.0=all HF]",
        f"Poor High-Frequency Ratio      : {sp.poor_high_freq_ratio}  "
        f"[0.0=no HF, 1.0=all HF]",
        f"Rich Spectral Diversity (std)  : {sp.rich_spectral_diversity}  "
        f"[higher=more natural variation]",
        f"Poor Spectral Diversity (std)  : {sp.poor_spectral_diversity}  "
        f"[higher=more natural variation]",
        f"Spectral Anomaly Score         : {sp.anomaly_score}  "
        f"[0.0=normal, 1.0=highly anomalous]",
    ))

    # ── ELA ──
    sections.append(_section(
        "ERROR LEVEL ANALYSIS (ELA)",
        f"Mean Intensity       : {ela.mean_intensity}  "
        f"[average compression error level]",
        f"Std Intensity        : {ela.std_intensity}  "
        f"[spread of error levels]",
        f"Max Intensity        : {ela.max_intensity}  "
        f"[peak error level]",
        f"Uniformity Score     : {ela.uniformity_score}  "
        f"[0.0=highly varied, 1.0=perfectly uniform]",
        f"Spatial Entropy      : {ela.spatial_entropy}  "
        f"[complexity of error distribution]",
        f"Splicing Indicator   : {ela.splicing_indicator}  "
        f"[0.0=no splice evidence, 1.0=strong splice evidence]",
    ))

    # ── PRNU ──
    sections.append(_section(
        "SENSOR NOISE FINGERPRINT (PRNU)",
        f"PRNU Energy            : {prnu.energy}  "
        f"[mean squared noise intensity]",
        f"PRNU Spatial Std Dev   : {prnu.spatial_std}  "
        f"[noise spatial variation]",
        f"Spectral Flatness      : {prnu.spectral_flatness}  "
        f"[1.0=white noise/AI-like, 0.0=structured/camera-like]",
        f"PRNU Strength Score    : {prnu.strength_score}  "
        f"[0.0=absent, 1.0=strong camera signature]",
        f"Camera Consistency     : {prnu.camera_consistency}  "
        f"[consistent|weak|absent]",
    ))

    # ── Signal Agreement (pre-computed) ──
    sections.append(_section(
        "SIGNAL AGREEMENT ANALYSIS (pre-computed)",
        f"Spectral Vote       : {agreement.spectral_vote}",
        f"ELA Vote            : {agreement.ela_vote}",
        f"PRNU Vote           : {agreement.prnu_vote}",
        f"Agreement Status    : {agreement.agreement_status}",
        f"Agreeing Modalities : {agreement.agreeing_count} / 3",
        f"Dissenting Signals  : {', '.join(agreement.dissenting_signals) or 'None'}",
        f"Summary             : {agreement.narrative_hint}",
    ))

    # ── Derived Assessments ──
    sections.append(_section(
        "DERIVED ASSESSMENTS",
        f"Risk Level          : {signals.risk_level}",
        f"Confidence Level    : {signals.confidence_level}",
        f"Primary Evidence    : {signals.primary_evidence}",
    ))

    # ── Additional Context (if any) ──
    if additional_context:
        sections.append(_section(
            "INVESTIGATOR NOTES",
            additional_context,
        ))

    # ── Closing Instruction ──
    format_instructions = {
        ReportFormat.DETAILED: (
            "Produce your DETAILED forensic report now, following the "
            "exact template from your instructions. Reference specific "
            "metric values to support every interpretive statement."
        ),
        ReportFormat.SUMMARY: (
            "Produce your BRIEF summary report now, under 150 words, "
            "following the exact template from your instructions."
        ),
        ReportFormat.STRUCTURED_JSON: (
            "Produce the JSON forensic report now. Output ONLY valid JSON, "
            "no markdown fences or surrounding text."
        ),
    }

    sections.append(format_instructions[report_format])

    return "\n\n".join(sections)


# ══════════════════════════════════════════════════════
#  PUBLIC API
# ══════════════════════════════════════════════════════

def get_system_prompt(
    report_format: ReportFormat = ReportFormat.DETAILED,
) -> str:
    """
    Return the system prompt for the specified report format.

    Args:
        report_format: Desired output format.

    Returns:
        System prompt string.
    """
    return _SYSTEM_PROMPTS[report_format]


def build_prompt_pair(
    signals: ForensicSignals,
    report_format: ReportFormat = ReportFormat.DETAILED,
    case_id: Optional[str] = None,
    additional_context: Optional[str] = None,
) -> Dict[str, str]:
    """
    Build both system and user prompts as a ready-to-use dict.

    Convenience method for the forensic agent to construct the
    complete message payload in one call.

    Args:
        signals:            ForensicSignals from the detector pipeline.
        report_format:      Desired output format.
        case_id:            Optional case identifier.
        additional_context: Optional investigator notes.

    Returns:
        Dict with keys 'system' and 'user' containing prompt strings.

    Usage in forensic_agent.py:
        prompts = build_prompt_pair(signals, ReportFormat.DETAILED)
        messages = [
            {"role": "system", "content": prompts["system"]},
            {"role": "user",   "content": prompts["user"]},
        ]
    """
    return {
        "system": get_system_prompt(report_format),
        "user": build_user_prompt(
            signals,
            report_format=report_format,
            case_id=case_id,
            additional_context=additional_context,
        ),
    }


def get_prompt_metadata() -> Dict[str, str]:
    """
    Return prompt version metadata for audit logging.

    Should be stored alongside every generated report for
    reproducibility and compliance traceability.
    """
    return {
        "prompt_version": PROMPT_VERSION,
        "prompt_schema": PROMPT_SCHEMA,
        "supported_formats": [f.value for f in ReportFormat],
    }


# ══════════════════════════════════════════════════════
#  INTERNAL HELPERS
# ══════════════════════════════════════════════════════

def _section(title: str, *lines: str) -> str:
    """
    Format a labeled section with visual separators.

    Produces:
        ─── TITLE ───
        line1
        line2
        ...
    """
    separator = f"─── {title} ───"
    body = "\n".join(lines)
    return f"{separator}\n{body}"


# ══════════════════════════════════════════════════════
#  PROMPT VALIDATION UTILITY
# ══════════════════════════════════════════════════════

def validate_prompt_completeness(signals: ForensicSignals) -> List[str]:
    """
    Check that all required fields in ForensicSignals are populated
    before building a prompt. Returns a list of warnings (empty = OK).

    This is a safety net to catch upstream extraction failures
    before they reach the LLM and produce garbage reports.
    """
    warnings: List[str] = []

    # ── Detector output checks ──
    if signals.probability is None:
        warnings.append("CRITICAL: probability is None — detector may have failed.")
    if signals.verdict not in ("AI GENERATED", "REAL IMAGE"):
        warnings.append(
            f"WARNING: unexpected verdict value '{signals.verdict}'."
        )

    # ── Spectral checks ──
    sp = signals.spectral
    if sp is None:
        warnings.append("CRITICAL: spectral metrics missing entirely.")
    else:
        if not (0.0 <= sp.anomaly_score <= 1.0):
            warnings.append(
                f"WARNING: spectral anomaly_score out of range: {sp.anomaly_score}"
            )
        if sp.rich_mean_energy <= 0:
            warnings.append(
                "WARNING: rich_mean_energy is zero/negative — possible extraction failure."
            )

    # ── ELA checks ──
    ela = signals.ela
    if ela is None:
        warnings.append("CRITICAL: ELA metrics missing entirely.")
    else:
        if not (0.0 <= ela.uniformity_score <= 1.0):
            warnings.append(
                f"WARNING: ELA uniformity_score out of range: {ela.uniformity_score}"
            )
        if not (0.0 <= ela.splicing_indicator <= 1.0):
            warnings.append(
                f"WARNING: ELA splicing_indicator out of range: {ela.splicing_indicator}"
            )

    # ── PRNU checks ──
    prnu = signals.prnu
    if prnu is None:
        warnings.append("CRITICAL: PRNU metrics missing entirely.")
    else:
        if prnu.camera_consistency not in ("consistent", "weak", "absent"):
            warnings.append(
                f"WARNING: unexpected camera_consistency: '{prnu.camera_consistency}'"
            )
        if not (0.0 <= prnu.strength_score <= 1.0):
            warnings.append(
                f"WARNING: PRNU strength_score out of range: {prnu.strength_score}"
            )

    # ── Derived assessment checks ──
    if signals.risk_level not in ("LOW", "MEDIUM", "HIGH", "CRITICAL"):
        warnings.append(
            f"WARNING: unexpected risk_level: '{signals.risk_level}'"
        )
    if signals.confidence_level not in ("LOW", "MEDIUM", "HIGH"):
        warnings.append(
            f"WARNING: unexpected confidence_level: '{signals.confidence_level}'"
        )

    # ────────────────────────────────────────────────
    # Updated for Issue 3: validate "multiple_signals" as valid value
    # ────────────────────────────────────────────────
    if signals.primary_evidence not in VALID_PRIMARY_EVIDENCE:
        warnings.append(
            f"WARNING: unexpected primary_evidence: '{signals.primary_evidence}'. "
            f"Valid values: {VALID_PRIMARY_EVIDENCE}"
        )
    # ────────────────────────────────────────────────

    return warnings


# ══════════════════════════════════════════════════════
#  MODULE SELF-TEST
# ══════════════════════════════════════════════════════

if __name__ == "__main__":
    """
    Self-test: build prompts from synthetic signals and print them.
    Run with: python reasoning_prompt.py
    """

    # ── Test 1: Standard signals (single primary evidence) ──
    test_signals = ForensicSignals(
        raw_logit=2.345,
        probability=0.9125,
        verdict="AI GENERATED",
        threshold=0.7,
        spectral=SpectralMetrics(
            rich_mean_energy=18.4521,
            poor_mean_energy=12.1034,
            rich_high_freq_ratio=0.2134,
            poor_high_freq_ratio=0.1876,
            rich_poor_energy_ratio=1.5245,
            rich_spectral_diversity=0.3210,
            poor_spectral_diversity=0.2891,
            anomaly_score=0.7432,
        ),
        ela=ELAMetrics(
            mean_intensity=0.1523,
            std_intensity=0.0891,
            max_intensity=0.7845,
            uniformity_score=0.6312,
            spatial_entropy=3.2145,
            splicing_indicator=0.1234,
        ),
        prnu=PRNUMetrics(
            energy=0.001234,
            spatial_std=0.034521,
            spectral_flatness=0.8765,
            strength_score=0.1523,
            camera_consistency="absent",
        ),
        risk_level="HIGH",
        confidence_level="HIGH",
        primary_evidence="sensor_fingerprint",
    )

    # ── Test 2: Tied signals (multiple_signals) ──
    test_signals_tied = ForensicSignals(
        raw_logit=1.200,
        probability=0.7685,
        verdict="AI GENERATED",
        threshold=0.7,
        spectral=SpectralMetrics(
            rich_mean_energy=15.0,
            poor_mean_energy=10.0,
            rich_high_freq_ratio=0.25,
            poor_high_freq_ratio=0.20,
            rich_poor_energy_ratio=1.5,
            rich_spectral_diversity=0.30,
            poor_spectral_diversity=0.25,
            anomaly_score=0.6,
        ),
        ela=ELAMetrics(
            mean_intensity=0.15,
            std_intensity=0.09,
            max_intensity=0.75,
            uniformity_score=0.65,
            spatial_entropy=3.1,
            splicing_indicator=0.4,
        ),
        prnu=PRNUMetrics(
            energy=0.004,
            spatial_std=0.05,
            spectral_flatness=0.6,
            strength_score=0.4,
            camera_consistency="weak",
        ),
        risk_level="MEDIUM",
        confidence_level="MEDIUM",
        primary_evidence="multiple_signals",
    )

    for label, signals in [
        ("STANDARD", test_signals),
        ("TIED (multiple_signals)", test_signals_tied),
    ]:
        print(f"\n{'#' * 70}")
        print(f" TEST CASE: {label}")
        print(f"{'#' * 70}")

        # Validate
        warnings = validate_prompt_completeness(signals)
        if warnings:
            print("⚠️  Validation warnings:")
            for w in warnings:
                print(f"   {w}")
        else:
            print("✅ All validation checks passed.")

        # Signal agreement
        agreement = compute_signal_agreement(signals)
        print(f"\n📊 Signal Agreement: {agreement.agreement_status}")
        print(f"   Spectral: {agreement.spectral_vote}")
        print(f"   ELA:      {agreement.ela_vote}")
        print(f"   PRNU:     {agreement.prnu_vote}")
        print(f"   Hint:     {agreement.narrative_hint}")

        # Build and display prompts for each format
        for fmt in ReportFormat:
            print(f"\n{'=' * 60}")
            print(f" FORMAT: {fmt.value.upper()}")
            print(f"{'=' * 60}")

            prompts = build_prompt_pair(
                signals,
                report_format=fmt,
                case_id="INS-2024-00451",
            )

            print(f"\n--- SYSTEM PROMPT ({len(prompts['system'])} chars) ---")
            # Print first 500 chars to avoid flooding terminal
            print(prompts["system"][:500] + "...\n")

            print(f"--- USER PROMPT ({len(prompts['user'])} chars) ---")
            print(prompts["user"])

    # Show metadata
    print(f"\n{'=' * 60}")
    print(" PROMPT METADATA")
    print(f"{'=' * 60}")
    meta = get_prompt_metadata()
    for k, v in meta.items():
        print(f"  {k}: {v}")

    print("\n✅ Self-test complete.")
