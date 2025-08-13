# 🎞️ Video Preprocessing Pipeline

This Python script is a comprehensive preprocessing pipeline for video files, designed to extract various features for downstream analysis, such as emotion recognition or behavioral studies. The script leverages several libraries to handle video, audio, and facial feature extraction.

***

## 🚀 Features

The script performs the following key preprocessing tasks:

* **Video Frame Extraction**: Reads video files, resizes frames, and performs uniform temporal subsampling to create consistent video clips.
* **Audio Feature Extraction**: Extracts audio from videos and computes **Mel-Frequency Cepstral Coefficients (MFCCs)** 🎵. The MFCCs are then saved as a visual representation (a `.png` image).
* **Facial Feature Extraction**: Utilizes the **OpenFace** library to extract a wide range of facial features, including:
    * **Action Units (AUs)**: Quantifies facial muscle movements.
    * **2D Facial Landmarks**
    * **Head Pose**
    * **Gaze Direction**
* **Video Visualization**: Creates a new video file that overlays the extracted Action Unit (AU) information directly onto the video frames. This is a great way to visualize the output of OpenFace and inspect the data.

***

## ⚙️ Configuration

Before running the script, you must configure the following settings and prerequisites:

### Prerequisites

* **OpenFace**: This script relies on the OpenFace toolkit to extract facial features. You need to download and install OpenFace. The `OPENFACE_DIR` and `OPENFACE_PATH` variables must point to the correct installation directory and executable file.
* **Python Libraries**: Ensure you have all required Python libraries installed by running `pip install -r requirements.txt` (if a `requirements.txt` file exists) or manually installing them:
    * `decord`
    * `moviepy`
    * `scipy`
    * `numpy`
    * `matplotlib`
    * `tensorflow`
    * `pandas`
    * `opencv-python`
    * `Pillow`

### Script Settings

Adjust the following constants in the script as needed:

* `input_size = 224`: The target resolution for video frames.
* `num_frame = 8`: The number of frames to sample per video clip.
* `fps = 25`: The frames per second for the output videos.
* `OPENFACE_DIR`: The path to the OpenFace installation directory. **Change this to your specific path.**
* `OPENFACE_PATH`: The path to the OpenFace `FeatureExtraction.exe` executable. **Change this to your specific path.**
