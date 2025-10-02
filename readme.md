# AI-Powered Diabetic Foot Ulcer (DFU) Analysis Pipeline
---
# Project Overview
This project is a Streamlit web application developed as a demonstration for a research paper on the automated analysis of Diabetic Foot Ulcer (DFU) images. It implements a multi-stage pipeline utilizing deep learning models for image analysis and a Large Language Model (LLM) for report synthesis.

The tool is designed to assist in the preliminary, objective assessment of DFUs by providing quantitative measurements, visual model explanations, and a summarized report based on both computer vision analysis and patient-reported symptoms.

---
# Key Features
AI Classification: Automatically determines if an uploaded image contains a 'Normal' or 'Abnormal' DFU with a corresponding confidence score.

Wound Segmentation: Precisely outlines the boundary of the ulcer using a U-Net model, separating it from healthy skin.

Explainable AI (XAI):

Grad-CAM: Generates heatmaps to visualize the regions of interest for the classification model.

LIME: Provides a human-interpretable explanation of the model's prediction.

Automated Measurements: Calculates the area (in pixels and estimated mm²) and maximum width (in pixels and estimated mm) of the segmented ulcer.

Depth Estimation: Uses the MiDaS model to estimate the relative depth of the wound, providing insights into its three-dimensional characteristics.

Interactive Symptom Input: A user-friendly form to input patient-reported symptoms (e.g., redness, swelling, odor) for a more comprehensive analysis.

LLM-Generated Report: Integrates with the Groq API to generate a clear, easy-to-understand summary based on the combined findings from the image analysis and the symptom questionnaire.

---
# Technical Stack and Libraries
Framework: Streamlit

Deep Learning: PyTorch, Torchvision

Computer Vision: OpenCV, Pillow, scikit-image

AI Models:

timm (for EfficientNet-B0 classifier)

segmentation-models-pytorch (for U-Net)

Intel-ISL/MiDaS (for depth estimation)

LLM Integration: groq (for Llama 3.1)

Core Libraries: NumPy, Matplotlib, Albumentations

---

# Setup and Installation
Follow these steps to run the application locally.

1. Prerequisites
Python 3.9+
Git 

2. Clone the Repository
```bash
git clone <your-repository-url>
cd <your-repository-folder>
```
4. Create a Virtual Environment (Recommended)
# For Windows
```
python -m venv venv
venv\Scripts\activate
```

4. Install Dependencies

Install all required Python packages from the requirements.txt file.
```
pip install -r requirements.txt
```
5. Place the Model Files
Download and place the following pre-trained model files into the root directory of the project:

dfu_classifier.pth

unet_segmentation_model.pth

midas_small.pt

6. Set Up API Key
Create a directory named .streamlit in the root of your project.

Inside this directory, create a file named secrets.toml.

Add your Groq API key to this file as follows:
```
# .streamlit/secrets.toml
GROQ_API_KEY = "gsk_YourActualGroqApiKeyHere"
```
7. Run the Application
Once all dependencies and models are in place, run the Streamlit app from your terminal:
```
streamlit run your_app_name.py
```
The application should now be open and running in your local web browser.

---
Usage
Open the App: Launch the application using the command above.

Upload Image: Use the sidebar to upload a .jpg, .jpeg, or .png image of a diabetic foot ulcer.

Configure Parameters: Adjust the calibration and XAI settings in the sidebar as needed.

Analyze: Click the "Analyze Image" button to start the computer vision pipeline.

Review CV Results: The app will display the classification, segmentation mask, XAI heatmaps, and calculated measurements.

Answer Questionnaire: If the ulcer is classified as "Abnormal", a symptom form will appear. Fill it out based on the patient's condition.

Generate Report: Click the "Generate Final Report" button to send the combined data to the Groq LLM.

Read Summary: The final, comprehensive report will be displayed at the bottom of the page.

---
Disclaimer
This is a research and demonstration tool. It is not a medical device and should not be used for actual diagnosis or to make treatment decisions. The analysis provided is for informational purposes only. Always consult a qualified healthcare professional for any medical concerns or conditions.
