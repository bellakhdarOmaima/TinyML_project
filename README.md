# TinyML-Based Voice Command Recognition System for Real-Time LED Activation on Raspberry Pi
 
# Project Overview
This project is designed for beginners interested in learning about TinyML and its applications. The core functionality is a simple voice command recognition system that activates an LED on a Raspberry Pi in response to a “stop” keyword. Although this application is straightforward, it illustrates many essential steps of building and deploying a TinyML model, including data preprocessing, model training, optimization, and real-time deployment.

 # TinyML Introduction
As a starting point, we've created a presentation to introduce TinyML. This presentation covers the following:

  What is TinyML?
  How does TinyML work?
  Popular TinyML frameworks and tools.
  Benefits and challenges of deploying machine learning on embedded systems.
 # Note: Due to file size limitations on GitHub (the presentation is over 120MB), it is hosted on Google Drive. You can access it here.
 ' https://drive.google.com/drive/folders/1uMDKYxIjdhfLypIoT2NR3PPfkhUkzmnR?usp=drive_link '

# Voice Command Recognition System
The voice command system leverages a convolutional neural network (CNN) to classify audio data as "stop" or "not stop." It uses the Mel-frequency Cepstral Coefficient (MFCC) technique to extract features from audio recordings of various words spoken by different people. The trained model is optimized using TensorFlow Lite and deployed to a Raspberry Pi, where it listens for the "stop" keyword to trigger the LED.

# Project Files
The repository includes the following main files:

MFCC Feature Extraction (mfcc_extraction.py): This script extracts MFCC features from audio samples.
Classifier (mfcc_classifier.py): Uses MFCC features as input to a CNN model, classifying the audio as "stop" or "not stop."
Model Optimization (model_optimization_tflite.py): Converts the trained model to TensorFlow Lite format for deployment on the Raspberry Pi.
Raspberry Pi Deployment Code (raspberry_pi_inference.py): Imports the TensorFlow Lite model and configures GPIO settings on the Raspberry Pi to control the LED.
# Getting Started
To get started, clone this repository and follow the setup instructions below to install dependencies and prepare your environment.

bash
Copy code
git clone https://github.com/YourUsername/TinyML-Voice-Command-LED
cd TinyML-Voice-Command-LED
# Requirements
Raspberry Pi (any model with GPIO support)
Python 3.7+
TensorFlow (for training and conversion to TensorFlow Lite)
Raspbian OS on Raspberry Pi
Audio dataset with voice recordings of “stop” and other words
You’ll also need RPi.GPIO for Raspberry Pi GPIO control and SoundDevice or similar libraries for audio input.

# Project Pipeline
**1. Feature Extraction with MFCC**
We use mfcc_extraction.py to extract MFCC features from the dataset, providing a compact representation of the audio signals.
**2. CNN Classifier for Keyword Detection**
mfcc_classifier.py trains a CNN using the extracted MFCC features to recognize the “stop” keyword.
**3. Model Optimization**
In model_optimization_tflite.py, we convert our model to a TensorFlow Lite format, making it efficient enough for edge devices like the Raspberry Pi.
**4. Deployment on Raspberry Pi**
The raspberry_pi_inference.py script loads the optimized model and configures the Raspberry Pi GPIO to activate an LED whenever the "stop" keyword is detected.
 # Demonstration Video
A demonstration video is included to showcase the working model on the Raspberry Pi, where the LED lights up in response to the voice command.

 # Contact
If you have any questions or need assistance, please feel free to reach out! ; omaimabellakhdar@gmail.com
