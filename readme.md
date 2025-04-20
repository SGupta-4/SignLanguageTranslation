Absolutely! Here's an updated and detailed `README.md` with **"How to Use the Project"** instructions, including setup, model usage, and app execution.

---

# 🧠 Real-Time Sign Language Translation System

A real-time sign language translation system that uses **YOLOv8** for hand detection and a custom **Transformer-based SignModel** for gesture classification. The system captures webcam video, detects hand gestures, and classifies them into one of 2000 American Sign Language (ASL) labels.

---

## 🚀 Features

- 🖐️ Real-time hand detection using YOLOv8.
- 🔤 Gesture classification using a Transformer-based model.
- 🎥 Live webcam feed with overlayed prediction.
- 🧠 Trained on [WLASL v0.3 Dataset](https://github.com/dxli94/WLASL).
- 🔍 2000 ASL labels supported.
- 📦 Packaged as a Streamlit app for ease of use.

---

## 📂 Folder Structure

```
SignLanguageTranslator/
├── app.py                      # Main Streamlit app
├── sign_model.pt               # Trained Transformer-based gesture classifier
├── labels.txt                  # Label file (0 to 1999)
└── README.md                   # This file
```

---

## ⚙️ Setup Instructions

1. **Clone the Repository**

```bash
git clone https://github.com/yourusername/SignLanguageTranslator.git
cd SignLanguageTranslator
```

2. **Install Dependencies**

```bash
pip install -r requirements.txt
```

Dependencies include:

- `streamlit`
- `torch`
- `torchvision`
- `opencv-python`
- `ultralytics` (for YOLOv8)
- `numpy`, `Pillow`, etc.

3. **Download Trained Models**

Place the following files in the project root:

- `sign_model.pt` — your trained gesture classification model.
- `yolov8n.pt` — YOLOv8 Nano model (or any other YOLOv8 variant).
- `labels.txt` — label mapping file with format:
  ```
  0	hello
  1	thank you
  ...
  1999	whistle
  ```

---

## 🧪 How to Use the Project

1. **Launch the App**

```bash
streamlit run app.py
```

2. **Upload Required Files**

- Upload `sign_model.pt` when prompted.
- Upload `labels.txt` containing your 2000 ASL labels.

3. **Click “Start Webcam”**

- The webcam starts capturing frames.
- YOLOv8 detects hands and crops frames around them.
- Cropped video is passed to the `SignModel`.
- Prediction is displayed **live** on the video feed.

---

## 🧠 Model Architecture

### Hand Detection (YOLOv8)

- Model: `yolov8n.pt` (lightweight real-time capable)
- Detects hands in each frame.
- Only hand regions are cropped and passed for classification.

### Gesture Classification (Transformer)

- Base CNN: ResNet18 (extracts spatial features)
- Transformer Encoder: Captures temporal dependencies across frames.
- Fully Connected Layer: Classifies into 2000 ASL labels.

---

## 🧾 References

- [WLASL Dataset](https://github.com/dxli94/WLASL)
- [YOLOv8 by Ultralytics](https://github.com/ultralytics/ultralytics)
- [Streamlit](https://streamlit.io/)
- [ResNet](https://arxiv.org/abs/1512.03385)
- [Transformer](https://arxiv.org/abs/1706.03762)

---

## 📸 Screenshots

> (Optional) Add visuals of the running app, predictions on webcam, etc.

---

## 👨‍💻 Author

**Sarthak Gupta**

