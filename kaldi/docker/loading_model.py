import onnxruntime as ort


def load_model(model_path):
    # Load the ONNX model
    session = ort.InferenceSession(model_path)
    return session