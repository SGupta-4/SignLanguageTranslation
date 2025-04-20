# # # # # # import streamlit as st
# # # # # # import cv2
# # # # # # import torch
# # # # # # import torchvision.transforms as transforms
# # # # # # import torch.nn as nn
# # # # # # import torchvision.models as models
# # # # # # import numpy as np
# # # # # # import tempfile
# # # # # # import time
# # # # # # import os

# # # # # # # =====================
# # # # # # # Model Definition
# # # # # # # =====================
# # # # # # class SignModel(nn.Module):
# # # # # #     def __init__(self, num_classes):
# # # # # #         super(SignModel, self).__init__()
# # # # # #         self.cnn = models.resnet18(pretrained=False)
# # # # # #         self.cnn.fc = nn.Identity()
# # # # # #         self.transformer = nn.TransformerEncoder(
# # # # # #             nn.TransformerEncoderLayer(d_model=512, nhead=8), num_layers=2)
# # # # # #         self.classifier = nn.Linear(512, num_classes)

# # # # # #     def forward(self, x):
# # # # # #         B, T, C, H, W = x.shape
# # # # # #         x = x.view(B * T, C, H, W)
# # # # # #         feat = self.cnn(x)
# # # # # #         feat = feat.view(B, T, -1)
# # # # # #         feat = self.transformer(feat.permute(1, 0, 2))
# # # # # #         out = self.classifier(feat.mean(0))
# # # # # #         return out

# # # # # # # =====================
# # # # # # # Load Model
# # # # # # # =====================
# # # # # # @st.cache_resource
# # # # # # def load_model(model_path, num_classes):
# # # # # #     model = SignModel(num_classes)
# # # # # #     model = nn.DataParallel(model)
# # # # # #     model.load_state_dict(torch.load('sign_model.pt',map_location='cpu'))

# # # # # #     # model.load_state_dict(torch.load(model_path, map_location='cpu'))
# # # # # #     model.eval()
# # # # # #     return model

# # # # # # # =====================
# # # # # # # Preprocessing
# # # # # # # =====================
# # # # # # def preprocess_video(frames, max_frames=16):
# # # # # #     transform = transforms.Compose([
# # # # # #         transforms.ToPILImage(),
# # # # # #         transforms.ToTensor()
# # # # # #     ])

# # # # # #     processed = []
# # # # # #     for frame in frames[:max_frames]:
# # # # # #         frame = cv2.resize(frame, (224, 224))
# # # # # #         tensor = transform(frame)
# # # # # #         processed.append(tensor)
    
# # # # # #     while len(processed) < max_frames:
# # # # # #         processed.append(processed[-1])
    
# # # # # #     return torch.stack(processed).unsqueeze(0)  # [1, T, 3, 224, 224]

# # # # # # # =====================
# # # # # # # Webcam Video Capture
# # # # # # # =====================
# # # # # # def capture_frames(duration=3, fps=8):
# # # # # #     cap = cv2.VideoCapture(0)
# # # # # #     frames = []
# # # # # #     interval = 1.0 / fps
# # # # # #     start = time.time()

# # # # # #     st.info("Recording for {} seconds...".format(duration))
# # # # # #     while time.time() - start < duration:
# # # # # #         ret, frame = cap.read()
# # # # # #         if not ret:
# # # # # #             break
# # # # # #         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
# # # # # #         frames.append(frame)
# # # # # #         time.sleep(interval)

# # # # # #     cap.release()
# # # # # #     return frames

# # # # # # # =====================
# # # # # # # Streamlit App UI
# # # # # # # =====================
# # # # # # st.title("🧠 Real-Time Sign Language Translator")
# # # # # # st.markdown("This app uses a Transformer model to translate your sign language via webcam.")

# # # # # # model_file = st.file_uploader("Upload your trained PyTorch model (.pt)", type=["pt"])
# # # # # # label_file = st.file_uploader("Upload your label mapping (.txt)", type=["txt"])

# # # # # # if model_file and label_file:
# # # # # #     # Save uploaded model to temp file
# # # # # #     with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as tmp:
# # # # # #         tmp.write(model_file.read())
# # # # # #         model_path = tmp.name

# # # # # #     labels = [line.strip() for line in label_file.readlines()]
# # # # # #     label_map = {i: label for i, label in enumerate(labels)}

# # # # # #     model = load_model(model_path, num_classes=2000)

# # # # # #     if st.button("Start Webcam and Translate"):
# # # # # #         frames = capture_frames(duration=3, fps=8)
# # # # # #         video_tensor = preprocess_video(frames)

# # # # # #         # with torch.no_grad():
# # # # # #         #     output = model(video_tensor)
# # # # # #         #     pred_idx = torch.argmax(output, dim=1).item()
# # # # # #         #     prediction = label_map[pred_idx]
# # # # # #         #     predicted_label = predicted_label.decode() if isinstance(predicted_label, bytes) else predicted_label
# # # # # #         # video_path = "recorded_video.mp4"
# # # # # #         # with open(video_path, "wb") as f:
# # # # # #         #     f.write(video_bytes)
# # # # # #         # st.video(video_path)  # This plays the video
# # # # # #         with torch.no_grad():
# # # # # #             output = model(video_tensor)
# # # # # #             pred_idx = torch.argmax(output, dim=1).item()
# # # # # #             prediction = label_map[pred_idx]

# # # # # #         st.success(f"✅ Predicted Sign: **{prediction.upper()}**")

# # # # # # # Optional: Show the captured video as frames
# # # # # #         for frame in frames:
# # # # # #             st.image(frame, width=300)



# # # # # #         st.success(f"✅ Predicted Sign: **{prediction.upper()}**")
# # # # # #         #st.video(np.array(frames).astype(np.uint8))
# # # # # import streamlit as st
# # # # # import cv2
# # # # # import torch
# # # # # import torchvision.transforms as transforms
# # # # # import torch.nn as nn
# # # # # import torchvision.models as models
# # # # # import numpy as np
# # # # # import tempfile
# # # # # import time
# # # # # from ultralytics import YOLO

# # # # # # =====================
# # # # # # Model Definition
# # # # # # =====================
# # # # # class SignModel(nn.Module):
# # # # #     def __init__(self, num_classes):
# # # # #         super(SignModel, self).__init__()
# # # # #         self.cnn = models.resnet18(pretrained=False)
# # # # #         self.cnn.fc = nn.Identity()
# # # # #         self.transformer = nn.TransformerEncoder(
# # # # #             nn.TransformerEncoderLayer(d_model=512, nhead=8), num_layers=2)
# # # # #         self.classifier = nn.Linear(512, num_classes)

# # # # #     def forward(self, x):
# # # # #         B, T, C, H, W = x.shape
# # # # #         x = x.view(B * T, C, H, W)
# # # # #         feat = self.cnn(x)
# # # # #         feat = feat.view(B, T, -1)
# # # # #         feat = self.transformer(feat.permute(1, 0, 2))
# # # # #         out = self.classifier(feat.mean(0))
# # # # #         return out

# # # # # # =====================
# # # # # # Load Models
# # # # # # =====================
# # # # # @st.cache_resource
# # # # # def load_model(model_path, num_classes):
# # # # #     model = SignModel(num_classes)
# # # # #     model = nn.DataParallel(model)
# # # # #     model.load_state_dict(torch.load(model_path, map_location='cpu'))
# # # # #     model.eval()
# # # # #     return model

# # # # # @st.cache_resource
# # # # # def load_yolo_model():
# # # # #     return YOLO("yolov8n.pt")  # or yolov5s.pt

# # # # # # =====================
# # # # # # Preprocessing
# # # # # # =====================
# # # # # # def preprocess_video(frames, max_frames=16):
# # # # # #     transform = transforms.Compose([
# # # # # #         transforms.ToPILImage(),
# # # # # #         transforms.ToTensor()
# # # # # #     ])
# # # # # #     processed = []
# # # # # #     for frame in frames[:max_frames]:
# # # # # #         frame = cv2.resize(frame, (224, 224))
# # # # # #         tensor = transform(frame)
# # # # # #         processed.append(tensor)
    
# # # # # #     while len(processed) < max_frames:
# # # # # #         processed.append(processed[-1])
    
# # # # # #     return torch.stack(processed).unsqueeze(0)  # [1, T, 3, 224, 224]
# # # # # def preprocess_video(frames, max_frames=16):
# # # # #     transform = transforms.Compose([
# # # # #         transforms.ToPILImage(),
# # # # #         transforms.ToTensor()
# # # # #     ])

# # # # #     processed = []

# # # # #     for frame in frames[:max_frames]:
# # # # #         if frame is None:
# # # # #             continue
# # # # #         frame = cv2.resize(frame, (224, 224))
# # # # #         tensor = transform(frame)
# # # # #         processed.append(tensor)

# # # # #     # ❗ Fix: Check if any valid frames were processed
# # # # #     if len(processed) == 0:
# # # # #         raise ValueError("No valid frames captured from webcam.")

# # # # #     # Pad if necessary
# # # # #     while len(processed) < max_frames:
# # # # #         processed.append(processed[-1])  # duplicate last frame

# # # # #     return torch.stack(processed).unsqueeze(0)  # shape: [1, T, 3, 224, 224]


# # # # # # =====================
# # # # # # Webcam Capture with YOLO Visualization
# # # # # # =====================
# # # # # def capture_frames_with_yolo(duration=3, fps=8, show_live=True):
# # # # #     cap = cv2.VideoCapture(0)
# # # # #     frames = []
# # # # #     interval = 1.0 / fps
# # # # #     start = time.time()
# # # # #     yolo = load_yolo_model()

# # # # #     stframe = st.empty()

# # # # #     while time.time() - start < duration:
# # # # #         ret, frame = cap.read()
# # # # #         if not ret:
# # # # #             break
# # # # #         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# # # # #         # Run YOLO detection
# # # # #         results = yolo(frame_rgb, verbose=False)
# # # # #         annotated_frame = results[0].plot()

# # # # #         frames.append(frame_rgb)

# # # # #         if show_live:
# # # # #             stframe.image(annotated_frame, channels="RGB", caption="Live YOLO Feed")

# # # # #         time.sleep(interval)
# # # # #     if len(frames) == 0:
# # # # #         st.error("No frames captured. Please try again or check your webcam.")
# # # # #     else:
# # # # #     # proceed with preprocess_video(frames)
# # # # #         cap.release()
# # # # #     return frames

# # # # # # =====================
# # # # # # Streamlit App UI
# # # # # # =====================
# # # # # st.title("🧠 Real-Time Sign Language Translator + YOLO Live Preview")
# # # # # st.markdown("This app uses YOLO for live detection and a Transformer model to translate sign language.")

# # # # # model_file = st.file_uploader("Upload your trained PyTorch model (.pt)", type=["pt"])
# # # # # label_file = st.file_uploader("Upload your label mapping (.txt)", type=["txt"])

# # # # # if model_file and label_file:
# # # # #     with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as tmp:
# # # # #         tmp.write(model_file.read())
# # # # #         model_path = tmp.name

# # # # #     labels = [line.strip() for line in label_file.readlines()]
# # # # #     label_map = {i: label for i, label in enumerate(labels)}

# # # # #     model = load_model(model_path, num_classes=len(label_map))

# # # # #     if st.button("Start Webcam + Translate"):
# # # # #         frames = capture_frames_with_yolo(duration=3, fps=8, show_live=True)
# # # # #         video_tensor = preprocess_video(frames)

# # # # #         with torch.no_grad():
# # # # #             output = model(video_tensor)
# # # # #             pred_idx = torch.argmax(output, dim=1).item()
# # # # #             prediction = label_map[pred_idx]

# # # # #         st.success(f"✅ Predicted Sign: **{prediction.upper()}**")

# # # # #         for frame in frames:
# # # # #             st.image(frame, width=300)

# # # # import streamlit as st
# # # # import cv2
# # # # import torch
# # # # import torchvision.transforms as transforms
# # # # import torch.nn as nn
# # # # import torchvision.models as models
# # # # import numpy as np
# # # # import tempfile
# # # # import time
# # # # from PIL import Image

# # # # # =======================
# # # # # Model Definition
# # # # # =======================
# # # # class SignModel(nn.Module):
# # # #     def __init__(self, num_classes):
# # # #         super(SignModel, self).__init__()
# # # #         self.cnn = models.resnet18(pretrained=False)
# # # #         self.cnn.fc = nn.Identity()
# # # #         self.transformer = nn.TransformerEncoder(
# # # #             nn.TransformerEncoderLayer(d_model=512, nhead=8),
# # # #             num_layers=2
# # # #         )
# # # #         self.classifier = nn.Linear(512, num_classes)

# # # #     def forward(self, x):
# # # #         B, T, C, H, W = x.shape
# # # #         x = x.view(B * T, C, H, W)
# # # #         feat = self.cnn(x)
# # # #         feat = feat.view(B, T, -1)
# # # #         feat = self.transformer(feat.permute(1, 0, 2))
# # # #         out = self.classifier(feat.mean(0))
# # # #         return out

# # # # # =======================
# # # # # Load Model
# # # # # =======================
# # # # @st.cache_resource
# # # # def load_model(model_path, num_classes):
# # # #     model = SignModel(num_classes)
# # # #     model = nn.DataParallel(model)
# # # #     model.load_state_dict(torch.load(model_path, map_location='cpu'))
# # # #     model.eval()
# # # #     return model

# # # # # =======================
# # # # # Preprocess Frames
# # # # # =======================
# # # # def preprocess_frames(frames, max_frames=16):
# # # #     transform = transforms.Compose([
# # # #         transforms.ToPILImage(),
# # # #         transforms.ToTensor()
# # # #     ])
# # # #     processed = []

# # # #     for frame in frames[:max_frames]:
# # # #         frame = cv2.resize(frame, (224, 224))
# # # #         tensor = transform(frame)
# # # #         processed.append(tensor)

# # # #     if len(processed) == 0:
# # # #         return None

# # # #     while len(processed) < max_frames:
# # # #         processed.append(processed[-1])
    
# # # #     return torch.stack(processed).unsqueeze(0)

# # # # # =======================
# # # # # Streamlit App
# # # # # =======================
# # # # st.title("🧠 Real-Time Sign Language Recognition")

# # # # model_file = st.file_uploader("Upload your model (.pt)", type=["pt"])
# # # # label_file = st.file_uploader("Upload your labels.txt", type=["txt"])

# # # # if model_file and label_file:
# # # #     with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as tmp:
# # # #         tmp.write(model_file.read())
# # # #         model_path = tmp.name

# # # #     labels = [line.decode("utf-8").strip() if isinstance(line, bytes) else line.strip() for line in label_file.readlines()]
# # # #     label_map = {i: label for i, label in enumerate(labels)}

# # # #     model = load_model(model_path, num_classes=len(label_map))

# # # #     if st.button("Start Webcam"):
# # # #         cap = cv2.VideoCapture(0)
# # # #         frame_window = st.image([])  # Live feed display
# # # #         label_placeholder = st.empty()  # Label display

# # # #         frames = []
# # # #         try:
# # # #             while True:
# # # #                 ret, frame = cap.read()
# # # #                 if not ret:
# # # #                     st.warning("Couldn't read frame.")
# # # #                     break

# # # #                 frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
# # # #                 frames.append(frame_rgb)

# # # #                 # Show webcam feed
# # # #                 frame_window.image(frame_rgb, channels="RGB")

# # # #                 # Every 16 frames, predict
# # # #                 if len(frames) >= 16:
# # # #                     video_tensor = preprocess_frames(frames[-16:])
# # # #                     if video_tensor is not None:
# # # #                         with torch.no_grad():
# # # #                             output = model(video_tensor)
# # # #                             pred_idx = torch.argmax(output, dim=1).item()
# # # #                             prediction = label_map[pred_idx]
# # # #                             label_placeholder.markdown(f"### ✨ Predicted: **{prediction.upper()}**")
                
# # # #                 time.sleep(0.03)  # ~30 FPS

# # # #         except KeyboardInterrupt:
# # # #             cap.release()
# # # #             st.info("Webcam stopped.")
# # # import streamlit as st
# # # import cv2
# # # import torch
# # # import torchvision.transforms as transforms
# # # import torch.nn as nn
# # # import torchvision.models as models
# # # import numpy as np
# # # from PIL import Image
# # # import tempfile
# # # import time
# # # import os

# # # # ============ YOLO Load =============
# # # from ultralytics import YOLO
# # # yolo = YOLO("yolov8n.pt")  # you can use a custom hand-detection model here

# # # # ============ Your Transformer Model =============
# # # class SignModel(nn.Module):
# # #     def __init__(self, num_classes):
# # #         super(SignModel, self).__init__()
# # #         self.cnn = models.resnet18(pretrained=False)
# # #         self.cnn.fc = nn.Identity()
# # #         self.transformer = nn.TransformerEncoder(
# # #             nn.TransformerEncoderLayer(d_model=512, nhead=8), num_layers=2)
# # #         self.classifier = nn.Linear(512, num_classes)

# # #     def forward(self, x):
# # #         B, T, C, H, W = x.shape
# # #         x = x.view(B * T, C, H, W)
# # #         feat = self.cnn(x)
# # #         feat = feat.view(B, T, -1)
# # #         feat = self.transformer(feat.permute(1, 0, 2))
# # #         out = self.classifier(feat.mean(0))
# # #         return out

# # # # ============ Helper Functions =============
# # # def preprocess_single_frame(frame):
# # #     transform = transforms.Compose([
# # #         transforms.ToPILImage(),
# # #         transforms.Resize((224, 224)),
# # #         transforms.ToTensor()
# # #     ])
# # #     return transform(frame)

# # # @st.cache_resource
# # # def load_model(model_path, num_classes):
# # #     model = SignModel(num_classes)
# # #     model = nn.DataParallel(model)
# # #     model.load_state_dict(torch.load(model_path, map_location="cpu"))
# # #     model.eval()
# # #     return model

# # # # ============ UI =============
# # # st.title("✋ Real-Time Sign Language Recognition")

# # # model_file = st.file_uploader("Upload trained Transformer Model (.pt)", type=["pt"])
# # # label_file = st.file_uploader("Upload labels.txt", type=["txt"])

# # # if model_file and label_file:
# # #     # Save model temporarily
# # #     with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as tmp:
# # #         tmp.write(model_file.read())
# # #         model_path = tmp.name

# # #     labels = [line.strip() for line in label_file.readlines()]
# # #     label_map = {i: label for i, label in enumerate(labels)}

# # #     model = load_model(model_path, num_classes=len(labels))

# # #     stframe = st.empty()

# # #     # Start Webcam
# # #     cap = cv2.VideoCapture(0)
# # #     frames_buffer = []

# # #     st.info("Press 'Stop' button to stop the prediction.")

# # #     stop_button = st.button("Stop")

# # #     while cap.isOpened() and not stop_button:
# # #         ret, frame = cap.read()
# # #         if not ret:
# # #             break

# # #         # Hand Detection (YOLO)
# # #         results = yolo(frame)
# # #         boxes = results[0].boxes.xyxy.cpu().numpy()

# # #         cropped_frames = []
# # #         for box in boxes:
# # #             x1, y1, x2, y2 = map(int, box)
# # #             cropped = frame[y1:y2, x1:x2]
# # #             if cropped.size > 0:
# # #                 cropped_tensor = preprocess_single_frame(cropped)
# # #                 cropped_frames.append(cropped_tensor)

# # #         if len(cropped_frames) > 0:
# # #             # Add to frame buffer (rolling window)
# # #             frames_buffer.append(torch.stack(cropped_frames).mean(0))  # average if multiple hands

# # #             if len(frames_buffer) > 16:
# # #                 frames_buffer.pop(0)

# # #             if len(frames_buffer) == 16:
# # #                 video_tensor = torch.stack(frames_buffer).unsqueeze(0)  # [1, T, 3, 224, 224]
# # #                 with torch.no_grad():
# # #                     output = model(video_tensor)
# # #                     pred_idx = torch.argmax(output, dim=1).item()
# # #                     prediction = label_map[pred_idx]
# # #         else:
# # #             prediction = "No Hands"

# # #         # Draw prediction text
# # #         cv2.putText(frame, f"Prediction: {prediction}", (10, 30),
# # #                     cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

# # #         # Show bounding boxes
# # #         for box in boxes:
# # #             x1, y1, x2, y2 = map(int, box)
# # #             cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

# # #         # Show in Streamlit
# # #         stframe.image(frame, channels="RGB")

# # #     cap.release()
# # #     st.success("Webcam stopped.")
import streamlit as st
import cv2
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torchvision.models as models
import numpy as np
import tempfile
import os
from PIL import Image
from ultralytics import YOLO

# =====================
# Model Definition
# =====================
class SignModel(nn.Module):
    def __init__(self, num_classes):
        super(SignModel, self).__init__()
        self.cnn = models.resnet18(pretrained=False)
        self.cnn.fc = nn.Identity()
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=512, nhead=8), num_layers=2)
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        feat = self.cnn(x)
        feat = feat.view(B, T, -1)
        feat = self.transformer(feat.permute(1, 0, 2))
        out = self.classifier(feat.mean(0))
        return out

# =====================
# Load Model
# =====================
@st.cache_resource
def load_model(model_path, num_classes):
    model = SignModel(num_classes)
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model

# =====================
# Load Labels
# =====================
def load_labels(label_file):
    lines = label_file.read().decode().splitlines()
    label_map = {}
    for line in lines:
        if '\t' in line:
            idx, label = line.strip().split('\t')
            label_map[int(idx)] = label
    return label_map

# =====================
# Preprocess
# =====================
def preprocess_frames(frames, max_frames=16):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor()
    ])

    processed = []
    for frame in frames[:max_frames]:
        frame = cv2.resize(frame, (224, 224))
        tensor = transform(frame)
        processed.append(tensor)

    while len(processed) < max_frames:
        processed.append(processed[-1].clone())  # Duplicate last frame

    return torch.stack(processed).unsqueeze(0)  # [1, T, 3, 224, 224]

# =====================
# Continuous Webcam + YOLO + Prediction
# =====================
def run_realtime(model, label_map, yolo_model):
    stframe = st.empty()
    cap = cv2.VideoCapture(0)
    frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Hand detection using YOLO
        results = yolo_model(frame)
        annotated_frame = results[0].plot()

        # Collect 16 frames
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)

        if len(frames) == 16:
            video_tensor = preprocess_frames(frames)
            with torch.no_grad():
                output = model(video_tensor)
                pred_idx = torch.argmax(output, dim=1).item()
                predicted_label = label_map.get(pred_idx, "Unknown")
            frames = []  # Clear for next batch

        # Overlay prediction
        if 'predicted_label' in locals():
            cv2.putText(annotated_frame, f'{predicted_label.upper()}', (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 2)

        stframe.image(annotated_frame, channels="BGR", use_column_width=True)

    cap.release()

# =====================
# Streamlit UI
# =====================
st.title("🧠 Real-Time Sign Language Translator with YOLO")
st.markdown("Live sign detection using Transformer + YOLOv5/YOLOv8")

model_file = st.file_uploader("Upload your trained PyTorch model (.pt)", type=["pt"])
label_file = st.file_uploader("Upload your label mapping (.txt)", type=["txt"])

if model_file and label_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as tmp:
        tmp.write(model_file.read())
        model_path = tmp.name

    label_map = load_labels(label_file)
    model = load_model(model_path, num_classes=len(label_map))
    yolo_model = YOLO('yolov5s.pt')  # or 'yolov8n.pt' if using YOLOv8

    if st.button("Start Real-Time Prediction"):
        run_realtime(model, label_map, yolo_model)
# import torch
# import torch.nn as nn
# import torchvision.transforms as transforms
# import cv2
# import numpy as np
# from model import SignModel  # Ensure your model definition is in model.py
# from PIL import Image

# # Check if GPU is available
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Initialize the model
# model = SignModel(num_classes=100)  # Adjust `num_classes` as per your setup

# # Load the model checkpoint (handling DataParallel)
# checkpoint = torch.load('sign_model_checkpoint.pth', map_location=device)
# new_state_dict = {}
# for k, v in checkpoint.items():
#     if k.startswith('module.'):
#         new_state_dict[k[7:]] = v  # Strip "module."
#     else:
#         new_state_dict[k] = v

# model.load_state_dict(new_state_dict)
# model.to(device)
# model.eval()

# # Define image preprocessing
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor()
# ])

# # Define class labels (replace with your actual labels)
# class_labels = [f"Sign_{i}" for i in range(100)]  # or load from a file

# def preprocess_frame(frame):
#     image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
#     image = transform(image)
#     return image.unsqueeze(0)  # Add batch dimension

# def predict(frame):
#     tensor = preprocess_frame(frame).to(device)
#     with torch.no_grad():
#         output = model(tensor)
#         predicted = torch.argmax(output, dim=1).item()
#         label = class_labels[predicted]
#     return label

# def run_app():
#     cap = cv2.VideoCapture(0)  # Use webcam

#     if not cap.isOpened():
#         print("Error: Could not open webcam.")
#         return

#     print("Press 'q' to quit...")

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         label = predict(frame)

#         cv2.putText(frame, f"Predicted: {label}", (10, 30),
#                     cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

#         cv2.imshow("Sign Language Translator", frame)

#         if cv2.waitKey(1) & 0xFF == ord("q"):
#             break

#     cap.release()
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     run_app()
# import streamlit as st
# import cv2
# import torch
# import torchvision.transforms as transforms
# import torch.nn as nn
# import torchvision.models as models
# import numpy as np
# import tempfile
# import time

# # =====================
# # Model Definition
# # =====================
# class SignModel(nn.Module):
#     def __init__(self, num_classes):
#         super(SignModel, self).__init__()
#         self.cnn = models.resnet18(pretrained=False)
#         self.cnn.fc = nn.Identity()
#         self.transformer = nn.TransformerEncoder(
#             nn.TransformerEncoderLayer(d_model=512, nhead=8), num_layers=2)
#         self.classifier = nn.Linear(512, num_classes)

#     def forward(self, x):
#         B, T, C, H, W = x.shape
#         x = x.view(B * T, C, H, W)
#         feat = self.cnn(x)
#         feat = feat.view(B, T, -1)
#         feat = self.transformer(feat.permute(1, 0, 2))
#         out = self.classifier(feat.mean(0))
#         return out

# # =====================
# # Load Model
# # =====================
# @st.cache_resource
# def load_model(model_path, num_classes):
#     model = SignModel(num_classes)
#     model = nn.DataParallel(model)
#     model.load_state_dict(torch.load(model_path, map_location='cpu'))
#     model.eval()
#     return model

# # =====================
# # Load Labels
# # =====================
# def load_labels(label_file):
#     lines = label_file.read().decode().splitlines()
#     label_map = {}
#     for line in lines:
#         if '\t' in line:
#             idx, label = line.strip().split('\t')
#             label_map[int(idx)] = label
#     return label_map

# # =====================
# # Preprocessing
# # =====================
# def preprocess_video(frames, max_frames=16):
#     transform = transforms.Compose([
#         transforms.ToPILImage(),
#         transforms.ToTensor()
#     ])

#     processed = []
#     for frame in frames[:max_frames]:
#         frame = cv2.resize(frame, (224, 224))
#         tensor = transform(frame)
#         processed.append(tensor)

#     while len(processed) < max_frames:
#         processed.append(processed[-1].clone())  # Duplicate last frame if needed

#     return torch.stack(processed).unsqueeze(0)  # [1, T, 3, 224, 224]

# # =====================
# # Webcam Video Capture
# # =====================
# def capture_frames(duration=3, fps=8):
#     cap = cv2.VideoCapture(0)
#     frames = []
#     interval = 1.0 / fps
#     start = time.time()

#     st.info("Recording for {} seconds...".format(duration))
#     while time.time() - start < duration:
#         ret, frame = cap.read()
#         if not ret:
#             break
#         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         frames.append(frame)
#         time.sleep(interval)

#     cap.release()
#     return frames

# # =====================
# # Streamlit App UI
# # =====================
# st.title("🧠 Real-Time Sign Language Translator")
# st.markdown("This app uses a Transformer model to translate your sign language via webcam.")

# model_file = st.file_uploader("Upload your trained PyTorch model (.pt)", type=["pt"])
# label_file = st.file_uploader("Upload your label mapping (.txt)", type=["txt"])

# if model_file and label_file:
#     # Save uploaded model to temp file
#     with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as tmp:
#         tmp.write(model_file.read())
#         model_path = tmp.name

#     label_map = load_labels(label_file)
#     model = load_model(model_path, num_classes=len(label_map))

#     if st.button("Start Webcam and Translate"):
#         frames = capture_frames(duration=3, fps=8)
#         video_tensor = preprocess_video(frames)

#         with torch.no_grad():
#             output = model(video_tensor)
#             pred_idx = torch.argmax(output, dim=1).item()
#             prediction = label_map.get(pred_idx, "Unknown")

#         st.success(f"✅ Predicted Sign: **{prediction.upper()}**")
