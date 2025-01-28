python yolo_gradio_app.py

from ultralytics import YOLO
import gradio as gr


model = YOLO("https://github.com/baypogiMD/MelanomaENV2/blob/main/MelanomaENV2.pt")


def classify_image(image):
    results = model(image)
    predicted_class = results.names[results.probs.argmax()]
    confidence = results.probs.max().item() * 100
    return f"Predicted Class: {predicted_class}\nConfidence: {confidence:.2f}%"


title = "Dermoscopy Image Prediction (Melanoma versus NonMelanoma)"
description = "Upload a dermoscopy image to classify it whether it is a melanoma or not"

# Gradio Interface
interface = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type="filepath"),
    outputs=gr.Textbox(label="Prediction"),
    title=title,
    description=description,
    theme="default",
)


if __name__ == "__main__":
    interface.launch()
