from datetime import datetime, timezone
import numpy as np
import cv2
import onnxruntime as ort
import make87
from huggingface_hub import hf_hub_download
from make87_messages.image.compressed.image_jpeg import ImageJPEG
from make87_messages.text.string_pb2 import String

# Download model.onnx and labels.txt from Hugging Face if not present
def download_resources():
    onnx_path = hf_hub_download(
        repo_id="chriamue/bird-species-classifier",
        filename="model.onnx"
    )
    labels_path = hf_hub_download(
        repo_id="chriamue/bird-species-dataset",
        filename="birds_labels.txt"
    )
    return onnx_path, labels_path

def load_labels(filepath):
    with open(filepath, "r") as f:
        labels = [line.strip() for line in f.readlines()]
    return labels

def preprocess_image(jpeg_bytes):
    img_array = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Failed to decode image")
    rgb = cv2.cvtColor(cv2.resize(bgr, (224, 224)), cv2.COLOR_BGR2RGB)
    arr = rgb.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std
    return np.transpose(arr, (2, 0, 1))[None, ...]

def main():
    make87.initialize()
    onnx_path, labels_path = download_resources()
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    labels = load_labels(labels_path)

    subscriber = make87.get_subscriber(
        name="DETECTED_CHANGED_IMAGE",
        message_type=ImageJPEG
    )
    publisher_score = make87.get_publisher(
        name="BIRD_CLASS",
        message_type=String
    )
    publisher_image = make87.get_publisher(
        name="BIRD_IMAGE",
        message_type=String
    )

    def callback(msg: ImageJPEG):
        try:
            tensor = preprocess_image(msg.data)
            logits = session.run(None, {input_name: tensor})[0][0]
            idx = int(np.argmax(logits))
            probs = np.exp(logits - np.max(logits))
            probs /= probs.sum()
            conf = float(probs[idx])
            species = labels[idx] if idx < len(labels) else f"cls_{idx}"
            result = (
                f"Species: {species}, Confidence: {conf*100:.1f}% "
                f"(captured at {msg.header.timestamp.ToDatetime().replace(tzinfo=timezone.utc).isoformat()})"
            )
            out = String(data=result)
            publisher_score.publish(message=out)
            publisher_image.publish(message=msg)
            print(result)
        except Exception as e:
            print("Classification error:", e)

    subscriber.subscribe(callback)
    make87.loop()

if __name__ == "__main__":
    main()
