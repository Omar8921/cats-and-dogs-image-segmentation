
# Cats and Dogs Image Segmentation

This project is an image segmentation application that segments **cats and dogs** from images using a deep learning model.  
The goal of the project is to demonstrate a complete pipeline starting from a trained segmentation model to a working web application.

The backend performs model inference, while a minimal frontend allows users to upload images and visualize the segmentation results directly in the browser.

---

## Model

- Architecture: **U-Net with ResNet34 encoder**
- Task: Binary image segmentation (cats and dogs)
- Framework: **PyTorch**

The model was trained on a preprocessed version of the Oxford-IIIT Pet dataset adapted specifically for image segmentation.

### Dataset

The dataset is hosted on Kaggle and can be found here:

https://www.kaggle.com/datasets/omaralmadani/oxford-iiit-pet-preprocessed-for-segmentation

---

## Application Overview

The application consists of two main parts:

### Backend
- Built with **FastAPI**
- Handles image upload and preprocessing
- Runs segmentation inference using the trained PyTorch model
- Returns:
  - the predicted segmentation mask
  - an overlay of the mask on the original image

### Frontend
- Minimal **HTML, CSS, and JavaScript**
- Sends images to the backend using a POST request
- Displays the returned segmentation results

The focus of the frontend is simplicity and clarity rather than heavy UI frameworks.

---

## Project Structure

```
.
├── backend/
│   ├── app/
│   │   ├── api/
│   │   ├── services/
│   │   └── main.py
├── deep_learning/
│   ├── inference.py
│   ├── model/
│   └── config.py
│   └── evaluate.py
│   └── main.py
│   └── preprocess.py
│   └── train.py
│   └── transforms.py
│   └── utils.py
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── script.js
└── README.md
```

---

## How It Works

1. A user uploads an image through the web interface.
2. The image is sent to the FastAPI backend.
3. The backend preprocesses the image and runs the segmentation model.
4. The predicted mask and overlay are generated.
5. The results are returned to the frontend and displayed in the browser.

---

## Use Cases

- Visualizing image segmentation models
- Learning how to serve deep learning models through an API
- Building small demos or research prototypes
- Understanding end-to-end ML deployment workflows

---

## Notes

- This project is intended for demonstration and learning purposes.
- The code prioritizes readability and a clear separation between API, service, and inference logic.
