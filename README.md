 SimuDrive-AI 🚗🧠

SimuDrive-AI is a Python-based self-driving car simulation project. It integrates multiple computer vision modules including object detection using YOLOv5, lane detection, and traffic sign recognition using deep learning. This project is built as a part of a research initiative and is modular, allowing easy expansion and testing.



## 🔧 Features

- 🚦 **Traffic Sign Recognition** using a trained CNN model
- 🛣️ **Lane Detection** using OpenCV and gradient masking
- 🧍 **Object Detection** with YOLOv5 (local repo)
- 📹 Video-based inference from simulated or recorded driving footage
- 📁 Modular code with clear separation of models and utilities

---
🧠 Model Info
Traffic Sign Classifier
Input: Resized sign images

Output: Predicted class (e.g., Speed Limit, Stop, etc.)

Trained using: TensorFlow/Keras

YOLOv5s
Source: Local yolov5-master directory

Pretrained weights are used (yolov5s.pt)
OpenAI (ChatGPT) – Development Support

Ultralytics – YOLOv5 Framework

