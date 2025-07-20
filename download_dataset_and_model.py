import gdown
import os

# Output filenames
dataset_file = "phishing_dataset.csv"
model_file = "phishing_detector.onnx"

# Google Drive file IDs
dataset_id = "14UsrWAcb3W8lj5FUHHMP8VPx8FfSz9Pu"
model_id = "1gPIRz-n4suvZcAcIrUYdW60qtws16uq7"

# Download URLs
dataset_url = f"https://drive.google.com/uc?id={dataset_id}"
model_url = f"https://drive.google.com/uc?id={model_id}"

# Output directory (optional: create a subfolder for assets)
os.makedirs("assets", exist_ok=True)

print("Downloading dataset...")
gdown.download(dataset_url, f"assets/{dataset_file}", quiet=False)

print("Downloading ONNX model...")
gdown.download(model_url, f"assets/{model_file}", quiet=False)

print("\nAll files downloaded to ./assets/")