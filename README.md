# 🧠 NameThatThing
Identify real-world objects using Python & OpenCV

NameThatThing is an AI-powered computer vision project that identifies objects in real time using your device’s camera or image input. Built with Python, OpenCV, and a pre-trained deep learning model, it can recognize common everyday objects and display their names instantly.

# 🚀 Features

🎥 Real-time Detection — Detects objects live from your webcam feed.

🖼️ Static Image Mode — Identify objects in uploaded images.

🧩 Pre-trained Model Support — Uses a deep learning model (e.g., MobileNet SSD or YOLO).

📦 Lightweight & Easy Setup — Works seamlessly on most systems with Python & OpenCV.

💬 On-screen Labels — Displays detected object names with bounding boxes.

# 🛠️ Tech Stack

Language: Python

Libraries: OpenCV, NumPy, argparse (optional)

Model: MobileNet SSD / YOLOv3 (customizable)

# 🧠 How It Works

Loads a pre-trained object detection model (like MobileNet SSD).

Captures frames from webcam or reads an image.

Passes the frame through the model to detect objects.

Draws bounding boxes and labels detected objects with confidence scores.

Output

# 📋 Requirements

Python 3.8+

OpenCV 4.x

NumPy

You can install all dependencies using:

pip install -r requirements.txt
