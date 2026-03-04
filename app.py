import gradio as gr
import torch
import numpy as np
import PIL.Image
import os

from models import TextureContrastClassifier
from utils import azi_diff
from forensic_agent import ForensicAgent

# --- Configuration ---
MODEL_PATH = './checkpoints/best_model.pth'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DEFAULT_THRESHOLD = 0.7  # Higher threshold = Lower False Positives

# --- Initialize Reasoning Agent ---
# ── IMPROVEMENT 1: Safe initialization with try/except ──
try:
    forensic_agent = ForensicAgent(
        checkpoint_path=MODEL_PATH,
        device="auto",
        threshold=DEFAULT_THRESHOLD,
        enable_llm=True
    )
except Exception as e:
    print(f"❌ Failed to initialize forensic agent: {e}")
    forensic_agent = None

# --- Prediction ---
# ── IMPROVEMENT 2: Null check + exception handling ──
def predict(input_img, threshold):

    if input_img is None:
        return "Please upload an image.", None, None, ""

    if forensic_agent is None:
        return "Model failed to initialize.", None, None, ""

    try:
        html, ela_viz, noise_viz, report = forensic_agent.analyze_for_gradio(
            input_img,
            threshold
        )
        return html, ela_viz, noise_viz, report

    except Exception as e:
        return f"Error during analysis: {str(e)}", None, None, ""

# --- Gradio UI ---
with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue")) as demo:
    gr.HTML("<h1 style='text-align: center;'>🛡️ Multi-Modal AI Image Detector</h1>")
    gr.HTML("<p style='text-align: center;'>Insurance Claim Forensic Verification System</p>")

    with gr.Tabs():
        with gr.TabItem("Analysis"):
            with gr.Row():
                with gr.Column(scale=1):
                    input_ui = gr.Image(label="Upload Image", type="numpy")
                    threshold_slider = gr.Slider(
                        minimum=0.5, maximum=0.95, value=DEFAULT_THRESHOLD, step=0.05,
                        label="Sensitivity Threshold",
                        info="Higher values reduce False Positives (Real images flagged as AI)."
                    )
                    submit_btn = gr.Button("🔍 Run Forensic Analysis", variant="primary")

                with gr.Column(scale=3):
                    output_html = gr.HTML(label="Verdict")

                    # ── IMPROVEMENT 3: Cleaner report section with header ──
                    gr.Markdown("### 🧠 Forensic Reasoning Report")
                    reasoning_report = gr.Markdown()

                    with gr.Row():
                        ela_ui = gr.Image(label="ELA (Compression Inconsistency)", height=450)
                        noise_ui = gr.Image(label="PRNU (Sensor Noise Fingerprint)", height=450)

            gr.Markdown("---")
            gr.Markdown("### Forensic Visualization Interpretation")
            with gr.Row():
                gr.Info("💡 **ELA Heatmap:** Bright spots indicate areas with inconsistent JPEG compression, often a sign of generative artifacts or splicing.")
                gr.Info("💡 **PRNU Map:** Highlights high-frequency noise. Authentic photos contain sensor 'grain,' whereas AI images often show unnatural smoothness.")

        with gr.TabItem("Thesis Metrics & Methodology"):
            gr.Markdown("### Methodology: 4-Branch Late Fusion")
            gr.Markdown("""
            To maximize **Accuracy** and minimize **False Positives**, this system analyzes:
            * **Azimuthal Integrals (Spectral):** Captures frequency artifacts left by GANs/Diffusion models.
            * **ELA Branch:** Detects digital manipulation via quantization error levels.
            * **Noise Branch (PRNU):** Identifies the absence of unique physical sensor fingerprints.
            """)

            # Placeholder for your Unseen Accuracy results
            gr.Markdown("#### Final Validation Scores")
            gr.Markdown("| Metric | Internal (Val) | External (Unseen) |")
            gr.Markdown("| :--- | :--- | :--- |")
            gr.Markdown("| Accuracy | 78.68% | 72.47% |")
            gr.Markdown("| False Positive Rate | 0.29 | 0.27 |")

    submit_btn.click(
        fn=predict,
        inputs=[input_ui, threshold_slider],
        outputs=[output_html, ela_ui, noise_ui, reasoning_report]
    )

if __name__ == "__main__":
    demo.launch(debug=True)
