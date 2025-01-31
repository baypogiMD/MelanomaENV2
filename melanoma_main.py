import os
import gradio as gr
from ultralytics import YOLO

# Load the YOLO classification model
MODEL_PATH = os.getenv("YOLO_MODEL_PATH", "weights/last.pt")

def load_model(model_path):
    """Loads the YOLO model from the given path."""
    try:
        model = YOLO(model_path)
        print("✅ Model loaded successfully!")
        return model
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        raise

# Load the model
model = load_model(MODEL_PATH)

def classify_image(image_path):
    """Runs YOLO classification on the input image."""
    try:
        results = model.predict(source=image_path, save=False)

        if not results:
            return "Error: No predictions returned. Please check the input image."

        result = results[0]
        if result.probs is not None:
            predicted_class_idx = result.probs.top1
            predicted_class = model.names[predicted_class_idx]
            confidence = result.probs.top1conf.item() * 100

            return f"Predicted Class: {predicted_class}\nConfidence: {confidence:.2f}%"
        else:
            return "Error: Model did not return class probabilities. Ensure it is a classification model."

    except Exception as e:
        return f"Error during prediction: {e}"

def main():
    """Launches the Gradio interface."""
    title = "Dermoscopy Image Prediction (Melanoma vs. NonMelanoma)"
    description = "Upload a dermoscopy image to classify it as melanoma or nonmelanoma."

    interface = gr.Interface(
        fn=classify_image,
        inputs=gr.Image(type="filepath", label="Upload Dermoscopy Image"),
        outputs=gr.Textbox(label="Prediction"),
        title=title,
        description=description,
        theme="default",
    )

    interface.launch(server_name="0.0.0.0", server_port=7860, debug=True)

if __name__ == "__main__":
    main()
