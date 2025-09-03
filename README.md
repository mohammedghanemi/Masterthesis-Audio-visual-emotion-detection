# 🎞️ Audio-Visual Emotion Detection with CREMA-D, eNTERFACE, MFAW, and Mixed Datasets

## Overview
This project implements **audio-visual emotion recognition** using the **CREMA-D**, **eNTERFACE**, and **MFAW** datasets. The system integrates audio and visual data through preprocessing, fusion strategies, and a fine-tuned **TF VideoMAE** model to predict both **discrete emotion categories** and **continuous arousal/valence values**. Mixed datasets are also supported for cross-dataset generalization.

---

## Datasets

1. **CREMA-D**
   - Audio-visual clips labeled with six emotions: anger, happiness, neutral, disgust, fear, sadness.
   - Audio: WAV/MP3; Video: MP4.

2. **eNTERFACE**
   - Synchronized audio-visual recordings for emotion recognition.
   - Official dataset: [eNTERFACE05](https://enterface.net/enterface05/main.php?frame=emotion)

3. **MFAW**
   - Multimodal dataset with audio and video recordings for emotion recognition tasks.

4. **Mixed Datasets**
   - Combines CREMA-D, eNTERFACE, and MFAW for cross-dataset experiments.
   - Ensures consistent multimodal input sequences for training.

---

### Model Architecture

1. **Video Preprocessing**
   - Frame extraction, resizing, and temporal sampling.
   - Facial feature extraction (Action Units, head pose, gaze) using OpenFace.

2. **Audio Preprocessing**
   - Audio extraction from video clips.
   - Generation of **MFCC** and **Mel spectrogram images**.

3. **Audio-Visual Fusion**
   - Combines video frames and audio feature images into unified multimodal sequences.
   - Supports early, late, and hybrid fusion strategies.


## TF VideoMAE Fine-Tuned Model

The project uses a **TF VideoMAE encoder**, fine-tuned on the target datasets.

- **Base Model**: VideoMAE pretrained on Kinetics-400
  - Repository: [VideoMAE](https://github.com/innat/VideoMAE)
  - Pretrained Model: [TFVideoMAE_L_K400_16x224_FT](https://github.com/innat/VideoMAE/releases/download/v1.1/TFVideoMAE_L_K400_16x224_FT)
- **Fine-tuning**
  - Model is fine-tuned on CREMA-D, eNTERFACE, MFAW, and mixed datasets.
  - Captures emotion-specific spatio-temporal patterns in audio-visual sequences.

---

## Multi-Task Model: Classification & Regression

The fine-tuned TF VideoMAE predicts:

1. **Emotion Categories** (Classification)
2. **Arousal & Valence** (Regression)


## Dependencies

Install required Python libraries:

```bash
pip install decord moviepy scipy numpy matplotlib tensorflow pandas opencv-python Pillow


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
