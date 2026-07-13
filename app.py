import json

import gradio as gr
import numpy as np
import spaces
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input as eff_preprocess

MODEL_PATH = "best_model_efficientnetb0_finetuned.keras"
CLASS_INDICES_PATH = "class_indices.json"

model = tf.keras.models.load_model(MODEL_PATH)

with open(CLASS_INDICES_PATH, "r") as f:
    idx_to_class = json.load(f)
    idx_to_class = {int(k): v for k, v in idx_to_class.items()}


@spaces.GPU
def predict_oral_disease(img):
    if img is None:
        return None

    img = img.resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = eff_preprocess(img_array)

    preds = model.predict(img_array)[0]
    return {idx_to_class[i]: float(preds[i]) for i in range(len(preds))}


custom_css = """
.gradio-container {
    font-family: 'Segoe UI', sans-serif;
    max-width: 1000px !important;
    margin: auto !important;
}
#header {
    text-align: center;
    padding: 30px 20px;
    background: linear-gradient(135deg, #0f766e, #0369a1);
    border-radius: 16px;
    color: white;
    margin-bottom: 20px;
}
#header h1 {
    font-size: 32px;
    margin-bottom: 8px;
}
#header p {
    font-size: 15px;
    opacity: 0.9;
}
.class-card, .class-card * {
    color: #ffffff !important;
}
.class-card {
    text-align: center;
    padding: 10px;
    border-radius: 10px;
    background: #1e293b;
    font-size: 13px;
}
footer {visibility: hidden}
"""

with gr.Blocks(theme=gr.themes.Soft(primary_hue="teal", secondary_hue="cyan"), css=custom_css) as demo:

    gr.HTML(
        """
        <div id="header">
            <h1>🦷 Oral Disease Classifier</h1>
            <p>AI-powered detection of oral health conditions from images — powered by EfficientNetB0</p>
        </div>
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Upload an Image")
            image_input = gr.Image(type="pil", label="", height=300)
            predict_btn = gr.Button("Analyze Image", variant="primary", size="lg")

        with gr.Column(scale=1):
            gr.Markdown("### Results")
            output_label = gr.Label(num_top_classes=6, label="")

    gr.Markdown("### 🩺 Detectable Conditions")
    with gr.Row():
        for cls in ["Calculus", "Caries", "Gingivitis", "Hypodontia", "Tooth Discoloration", "Ulcers"]:
            gr.Markdown(f"**{cls}**", elem_classes="class-card")

    gr.Markdown(
        "---\n"
        "**Model:** EfficientNetB0 (fine-tuned)  |  **Test Accuracy:** 91.30%  |  "
        "Built with TensorFlow & Gradio\n\n"
        "*For educational purposes only — not a substitute for professional dental diagnosis.*"
    )

    predict_btn.click(fn=predict_oral_disease, inputs=image_input, outputs=output_label)
    image_input.change(fn=predict_oral_disease, inputs=image_input, outputs=output_label)


if __name__ == "__main__":
    demo.launch()
