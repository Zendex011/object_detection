# object_detection

## HomeObjects3K YOLOv8 Object Detection

A complete, end-to-end pipeline to train, evaluate, export, and deploy a YOLOv8 object detection model for common home objects, with deployment guidance for Raspberry Pi (TFLite). This repository contains the training notebook, export steps, and a Raspberry Pi inference script template.

### Introduction
- **Purpose**: Detect household objects (e.g., bed, sofa, chair, table, lamp) using a compact YOLOv8 model.
- **Motivation**: Enable lightweight, affordable edge AI for home automation, robotics, and smart monitoring.
- **Problem it solves**: Provides a reproducible pipeline from dataset setup through training, evaluation, and edge deployment on resource-constrained devices.

### System Architecture
- **Data source**: `homeobjects-3K` dataset with labeled images under `images/{train,val}` and labels in YOLO format.
- **Training**: Ultralytics YOLOv8n with augmentations; runs in Google Colab with GPU/CPU fallback.
- **Evaluation/Visualization**: In-notebook inference on validation images and web images.
- **Export**: Convert the trained `.pt` model to TFLite via Ultralytics export flow.
- **Deployment**: Copy TFLite model to Raspberry Pi and run inference with `tflite-runtime` and OpenCV.

Data flow (high-level):
1. Zip in Drive → unzip in Colab workspace
2. Generate `data.yaml` → train YOLOv8n
3. Save `best.pt` → export to `best_float32.tflite`
4. Transfer `.tflite` → Raspberry Pi → run inference script

### Methodology
1. **Environment & Data Prep**
   - Mount Google Drive, unzip `homeobjects-3K` into `/content/homeobjects-3K`.
   - Auto-create `data.yaml` with 12 classes.
2. **Model & Training**
   - Use `ultralytics` YOLOv8n pretrained weights (`yolov8n.pt`).
   - Train with augmentations (HSV shifts, flips, mosaic, mixup, perspective, shear, etc.).
   - Track outputs under `runs/detect/<experiment>/`.
3. **Evaluation**
   - Run quick inference on a validation image and a web image; visualize predictions.
4. **Export**
   - Export `best.pt` to ONNX and TFLite using Ultralytics export utilities.
5. **Deployment (Raspberry Pi)**
   - Install `tflite-runtime` on Pi and run provided `run_inference.py` to detect objects in an image.

### Screenshots & Results (Placeholders)
- Insert a training results plot (loss/metrics) here.
  - Placeholder: `docs/images/training_curves.png`
- Insert an inference visualization on a validation image.
  - Placeholder: `docs/images/val_prediction.jpg`
- Insert an inference visualization from Raspberry Pi run.
  - Placeholder: `docs/images/pi_prediction.jpg`
- Suggested metrics to show: mAP50, mAP50-95, precision, recall, inference time on Pi.

### Challenges & Solutions
- **TFLite output parsing differences**: TFLite export formats can vary. Validate output tensor shapes and adjust post-processing (transpose, indices) accordingly.
- **Preprocessing vs letterboxing**: Simple resize is convenient but may reduce accuracy. For best results, implement YOLO-style letterboxing and reverse-transform boxes.
- **NMS coordinate mismatch**: Ensure NMS receives boxes in the expected format. Convert between `xywh` and `xyxy` as needed or use a compatible NMS utility.
- **Model size vs speed on Pi**: Prefer `yolov8n` or quantized TFLite for real-time performance; consider INT8 if accuracy is acceptable.
- **File transfer from Colab**: Use Drive sync or explicit upload/download to persist artifacts; then `scp` to Pi.

### Conclusion & Future Plans
- The project delivers a reproducible path from training to edge deployment for home-object detection.
- **Future improvements**:
  - Add proper letterboxing and de-letterboxing.
  - Provide a video/webcam inference script for Pi Camera.
  - Explore INT8 quantization with calibration for faster inference.
  - Add Dockerfile for local training and export.
  - Integrate continuous training/evaluation workflow.

### How to Run

#### 1) Train in Google Colab
1. Open `od_final.ipynb` in Google Colab.
2. Ensure `homeobjects-3K.zip` exists in Drive → run the cells to mount, unzip, and train.
3. Verify training outputs under `/content/runs/detect/<experiment>/weights/best.pt`.

#### 2) Export to TFLite in Colab
1. Run the export cell to produce TFLite artifacts under `runs/detect/<experiment>/weights/`.
2. Confirm the TFLite path (e.g., `best_float32.tflite`).
3. Upload the `.tflite` file to Drive or download locally.

#### 3) Set up Raspberry Pi
```bash
sudo apt update
sudo apt install -y python3-pip python3-opencv
sudo apt install -y python3-tflite-runtime || pip3 install --no-cache-dir tflite-runtime
```

#### 4) Run Inference on Raspberry Pi
1. Copy `best_float32.tflite` and a test image to the Pi.
2. Save the provided script as `run_inference.py`.
3. Run:
```bash
python3 run_inference.py
```
4. The script saves `output_detections.jpg` with bounding boxes and labels.

### Tech Stack
- **Languages**: Python 3
- **Training/Model**: Ultralytics YOLOv8, PyTorch (in Colab)
- **Export**: ONNX, TensorFlow SavedModel, TFLite
- **Edge Inference**: tflite-runtime, OpenCV (cv2), NumPy
- **Environment**: Google Colab, Raspberry Pi OS

### Installation (Local/Dev)
If you prefer local development:
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -U pip
pip install ultralytics opencv-python numpy
```

### Usage
- Open and run `od_final.ipynb` in Colab for training and export.
- Use `run_inference.py` on Raspberry Pi with the exported `.tflite` model.

### Contributing
Contributions are welcome! Please:
1. Fork the repo
2. Create a feature branch
3. Commit changes with clear messages
4. Open a pull request describing your changes

### License
Specify a license for your project. Example:

```
MIT License

Copyright (c) 2025 <Your Name>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## HomeObjects3K YOLOv8 Object Detection

A complete, end-to-end pipeline to train, evaluate, export, and deploy a YOLOv8 object detection model for common home objects, with deployment guidance for Raspberry Pi (TFLite). This repository contains the training notebook, export steps, and a Raspberry Pi inference script template.

### Introduction
- **Purpose**: Detect household objects (e.g., bed, sofa, chair, table, lamp) using a compact YOLOv8 model.
- **Motivation**: Enable lightweight, affordable edge AI for home automation, robotics, and smart monitoring.
- **Problem it solves**: Provides a reproducible pipeline from dataset setup through training, evaluation, and edge deployment on resource-constrained devices.

### System Architecture
- **Data source**: `homeobjects-3K` dataset with labeled images under `images/{train,val}` and labels in YOLO format.
- **Training**: Ultralytics YOLOv8n with augmentations; runs in Google Colab with GPU/CPU fallback.
- **Evaluation/Visualization**: In-notebook inference on validation images and web images.
- **Export**: Convert the trained `.pt` model to TFLite via Ultralytics export flow.
- **Deployment**: Copy TFLite model to Raspberry Pi and run inference with `tflite-runtime` and OpenCV.

Data flow (high-level):
1. Zip in Drive → unzip in Colab workspace
2. Generate `data.yaml` → train YOLOv8n
3. Save `best.pt` → export to `best_float32.tflite`
4. Transfer `.tflite` → Raspberry Pi → run inference script

### Methodology
1. **Environment & Data Prep**
   - Mount Google Drive, unzip `homeobjects-3K` into `/content/homeobjects-3K`.
   - Auto-create `data.yaml` with 12 classes.
2. **Model & Training**
   - Use `ultralytics` YOLOv8n pretrained weights (`yolov8n.pt`).
   - Train with augmentations (HSV shifts, flips, mosaic, mixup, perspective, shear, etc.).
   - Track outputs under `runs/detect/<experiment>/`.
3. **Evaluation**
   - Run quick inference on a validation image and a web image; visualize predictions.
4. **Export**
   - Export `best.pt` to ONNX and TFLite using Ultralytics export utilities.
5. **Deployment (Raspberry Pi)**
   - Install `tflite-runtime` on Pi and run provided `run_inference.py` to detect objects in an image.

### Screenshots & Results (Placeholders)
- Insert a training results plot (loss/metrics) here.
  - Placeholder: `docs/images/training_curves.png`
- Insert an inference visualization on a validation image.
  - Placeholder: `docs/images/val_prediction.jpg`
- Insert an inference visualization from Raspberry Pi run.
  - Placeholder: `docs/images/pi_prediction.jpg`
- Suggested metrics to show: mAP50, mAP50-95, precision, recall, inference time on Pi.

### Challenges & Solutions
- **TFLite output parsing differences**: TFLite export formats can vary. Validate output tensor shapes and adjust post-processing (transpose, indices) accordingly.
- **Preprocessing vs letterboxing**: Simple resize is convenient but may reduce accuracy. For best results, implement YOLO-style letterboxing and reverse-transform boxes.
- **NMS coordinate mismatch**: Ensure NMS receives boxes in the expected format. Convert between `xywh` and `xyxy` as needed or use a compatible NMS utility.
- **Model size vs speed on Pi**: Prefer `yolov8n` or quantized TFLite for real-time performance; consider INT8 if accuracy is acceptable.
- **File transfer from Colab**: Use Drive sync or explicit upload/download to persist artifacts; then `scp` to Pi.

### Conclusion & Future Plans
- The project delivers a reproducible path from training to edge deployment for home-object detection.
- **Future improvements**:
  - Add proper letterboxing and de-letterboxing.
  - Provide a video/webcam inference script for Pi Camera.
  - Explore INT8 quantization with calibration for faster inference.
  - Add Dockerfile for local training and export.
  - Integrate continuous training/evaluation workflow.

### How to Run

#### 1) Train in Google Colab
1. Open `od_final.ipynb` in Google Colab.
2. Ensure `homeobjects-3K.zip` exists in Drive → run the cells to mount, unzip, and train.
3. Verify training outputs under `/content/runs/detect/<experiment>/weights/best.pt`.

#### 2) Export to TFLite in Colab
1. Run the export cell to produce TFLite artifacts under `runs/detect/<experiment>/weights/`.
2. Confirm the TFLite path (e.g., `best_float32.tflite`).
3. Upload the `.tflite` file to Drive or download locally.

#### 3) Set up Raspberry Pi
```bash
sudo apt update
sudo apt install -y python3-pip python3-opencv
sudo apt install -y python3-tflite-runtime || pip3 install --no-cache-dir tflite-runtime
```

#### 4) Run Inference on Raspberry Pi
1. Copy `best_float32.tflite` and a test image to the Pi.
2. Save the provided script as `run_inference.py`.
3. Run:
```bash
python3 run_inference.py
```
4. The script saves `output_detections.jpg` with bounding boxes and labels.

### Tech Stack
- **Languages**: Python 3
- **Training/Model**: Ultralytics YOLOv8, PyTorch (in Colab)
- **Export**: ONNX, TensorFlow SavedModel, TFLite
- **Edge Inference**: tflite-runtime, OpenCV (cv2), NumPy
- **Environment**: Google Colab, Raspberry Pi OS

### Installation (Local/Dev)
If you prefer local development:
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -U pip
pip install ultralytics opencv-python numpy
```

### Usage
- Open and run `od_final.ipynb` in Colab for training and export.
- Use `run_inference.py` on Raspberry Pi with the exported `.tflite` model.

### Contributing
Contributions are welcome! Please:
1. Fork the repo
2. Create a feature branch
3. Commit changes with clear messages
4. Open a pull request describing your changes

### License
Specify a license for your project. Example:

```
MIT License

Copyright (c) 2025 <Your Name>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```


