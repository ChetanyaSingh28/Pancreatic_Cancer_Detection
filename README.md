# Pancreas CT Segmentation using U-Net 🩺

This repository contains a deep learning web application built with Streamlit that automatically segments the pancreas from CT scan slices. It uses a custom U-Net architecture trained on the Pancreas CT Dataset.

## Features

* **Interactive UI:** Built with Streamlit for a clean, user-friendly interface.

* **On-the-Fly Prediction:** Upload a raw CT scan slice in `.npy` format to instantly generate a prediction mask.

* **Side-by-Side Visualization:** View the original input CT scan alongside the predicted binary mask.

* **Optimized for CPU:** Uses `tensorflow-cpu` to ensure lightweight, fast inference in cloud environments without needing a GPU.

## Project Structure

* `app.py`: The main Streamlit application containing the UI, preprocessing logic, and model inference.

* `requirements.txt`: Lightweight dependencies required to run the app.

* `pancreas_unet.h5`: The trained U-Net model weights (custom Dice/IoU metrics included in script).

* `README.md`: Project documentation.

## How to Run Locally

1. **Clone the repository:**

   ```
   git clone https://github.com/ChetanyaSingh28/Pancreatic_Cancer_Detection.git
   cd pancreas-ct-segmentation
   
   ```

2. **Create a virtual environment (optional but recommended):**

   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   
   ```

3. **Install dependencies:**

   ```
   pip install -r requirements.txt
   
   ```

4. **Run the Streamlit app:**

   ```
   streamlit run app.py
   
   ```

## Dataset

The model was trained using the [Pancreas CT Dataset](https://www.kaggle.com/datasets/tahsin/pancreasct-dataset) available on Kaggle.

## Deployment

This application is ready to be deployed on [Streamlit Community Cloud](https://streamlit.io/cloud). Simply connect this repository to your Streamlit account and select `app.py` as the main file path.