# Deepfake Detection & Analysis Framework

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This project provides an end-to-end pipeline to analyze video files and images for deepfakes. It uses an MTCNN model to detect and extract faces, which are then fed to an ensemble of specialist deep learning models to render a final verdict. The framework also includes a complete pipeline for training, evaluating, and comparing new specialist models.

## Key Features
- **Specialist Model Training:** A two-stage pipeline to train highly effective models on specific deepfake manipulation types.
- **Video Analysis Pipeline:** An end-to-end tool that takes a video file, extracts faces, and provides a final verdict.
- **Single Image Analysis:** A lightweight tool for quick tests on individual image files.
- **Structured & Reusable Code:** The core logic is organized into a professional Python package (`src`), making the code maintainable and easy to use.
- **Visual Reporting:** Generates clear visualizations, including confusion matrices, performance charts, and annotated frame grids.

***

Live Demo on Kaggle 🚀
The easiest way to run this project is by using the public notebooks on Kaggle. All required models and data are pre-attached, so you can run them instantly without any setup.

Click a link below to open a notebook on Kaggle. To run it, just click the "Copy & Edit" button to get your own interactive version, then run the cells from top to bottom.

➡️ [Run the training-specialiezd-xception-model-on-ff](https://www.kaggle.com/code/ahmedwaliid/training-specialiezd-xception-model-on-ff)

➡️ [Run the image_DeepFake_Detection](https://www.kaggle.com/code/adham7elmy/Image-deepfake-detection)

➡️ [Run the video_DeepFake_Detection](https://www.kaggle.com/code/adham7elmy/video-deepfake-detection)

➡️ [Run the models-statistics](https://www.kaggle.com/code/adham7elmy/models-statistics)

***

## Project Structure
The project uses a hybrid structure that separates reusable Python code from the demonstration and analysis notebooks.

DeepFake-Detection/
├── models/
│   └── (Place your pre-trained .keras models here)
│
├── notebooks/
│   ├── 5-specialist-model-training.ipynb
│   ├── image_DeepFake_Detection.ipynb
│   ├── model_statistics.ipynb
│   └── Video_DeepFake_Detection.ipynb
│
├── src/
│   ├── init.py
│   ├── config.py
│   ├── data_handler.py
│   ├── evaluator.py
│   ├── face_processor.py
│   ├── model_manager.py
│   ├── trainer.py
│   └── visualization.py
│
├── .gitignore
├── LICENSE
└── README.md

***

## Setup and Installation
1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/AdhamHelmy44/DeepFake-Detection](https://github.com/AdhamHelmy44/DeepFake-Detection)
    cd DeepFake-Detection
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```
3.  **Install dependencies:**
    *(You should create a `requirements.txt` file with libraries like tensorflow, mtcnn, opencv-python, etc.)*
    ```bash
    pip install -r requirements.txt
    ```
4.  **Download Models:** Place any pre-trained `.keras` models you wish to use for analysis into the `/models` directory.

***

## Usage Workflow
The project is designed to be used through the notebooks, which provide a clear, step-by-step interface for each task.

### 1. Quick Test on a Single Image
For a fast check on an individual image file.

* **Run:** `notebooks/image_DeepFake_Detection.ipynb`
* **Action:** Update the `IMAGE_TO_TEST_PATH` in the first cell and run the notebook to see the verdict.

### 2. Analyze a Full Video
The complete pipeline for processing a video file.

* **Run:** `notebooks/Video_DeepFake_Detection.ipynb`
* **Action:** Update the `SINGLE_VIDEO_PATH` in the first cell and run the notebook to see the frame-by-frame analysis and the final video verdict.

### 3. Train a New Specialist Model
This notebook contains the pipeline to train your own models on a specific deepfake type.

* **Run:** `notebooks/5_specialist_model_training.ipynb`
* **Action:**
    1.  **Prepare Your Data**: Organize your dataset as shown below. The script handles the `train/validation/test` split automatically.
        ```
        /your-dataset/
        ├── real/
        │   ├── 000/
        │   └── 001/
        └── fake/
            ├── Deepfakes/
            │   └── 000_003/
            └── FaceSwap/
                └── 001_004/
        ```
    2.  **Configure Settings**: In the notebook's first cell, update the paths and training parameters as needed.
    3.  **Execute**: Run the cells to start the training process. The best model will be saved to the output directory.

### 4. Compare Model Performance
After training multiple specialist models, use this notebook to find the best one.

* **Run:** `notebooks/model_statistics.ipynb`
* **Action:**
    1.  **Prepare a Test Set**: Create a test directory with `real` and `fake` subfolders containing images the models have not seen.
    2.  **Configure Paths**: In the notebook, update the path to the folder containing your trained models and the path to your test set.
    3.  **Execute**: Run the cells to generate a performance leaderboard and comparison charts.
