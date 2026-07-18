import gradio as gr
from PIL import Image
from model import predict

title = "Alzheimer's MRI Classification"

description = """
Upload a brain MRI image.

The model predicts:

- Mild Dementia
- Moderate Dementia
- Non Demented
- Very Mild Dementia
"""

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=[
        gr.Label(num_top_classes=4),
        gr.Textbox(label="Prediction")
    ],
    title=title,
    description=description,
)

demo.launch()