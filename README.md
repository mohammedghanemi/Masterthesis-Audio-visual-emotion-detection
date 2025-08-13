# Audio-Visual Emotion Detection Project

---

## ⚙️ Script Settings

Adjust the following constants in your scripts as needed:

- `input_size = 224`  
  Target resolution for video frames.

- `num_frame = 16`  
  Number of frames sampled per video clip.

- `fps = 25`  
  Frames per second for output videos.

- `OPENFACE_DIR`  
  Path to OpenFace installation directory. Change to your system path.

- `OPENFACE_PATH`  
  Path to OpenFace `FeatureExtraction.exe` executable. Change accordingly.

---

## 📖 Usage Overview

1. **Set Up Directories**  
   Create folders for `input_path` (videos) and `output_path` (results).

2. **Run Scripts**  
   Uncomment and update paths at the end of each script, then run with:  
   ```bash
   python preprocess.ipynb

## Output

- MFCC images  
- Videos with overlays  
- OpenFace CSV feature files

---

## 🧠 Multi-task Model Overview

This project supports multi-task learning combining:

### Emotion Classification

Classify discrete emotion classes (e.g., anger, happy, neutral).

### Arousal-Valence Regression

Predict continuous emotional intensity (arousal) and pleasantness (valence).

### Architecture Highlights

- Pre-trained backbone for feature extraction  
- Shared dense layers  
- Separate heads for classification and regression  
- Balanced loss functions per task

### Implementation Structure
Implementation/
├── Implementation_on_mixed_datasets/
│   └── multitask_mixeddata_with_fourDifferent_datasets.ipynb
├── Implementing_on_cremaD_dataset/
│   ├── oncrema-d-classification.ipynb
│   ├── oncrema-d-multitask.ipynb
│   └── oncrema-d-regression.ipynb
├── Implementing_on_enterface_dataset/
│   ├── enterface-classification.ipynb
│   ├── enterface-multitask.ipynb
│   └── enterface-regression.ipynb
└── Implementing_on_MAFW_dataset/
    ├── mafw-classification.ipynb
    ├── mafw-multitask.ipynb
    └── mafw-regression.ipynb

## ⚙️ How It Works

### 1. Data Preparation
- Organize train, validation, and test splits in folders.
- Use Pandas DataFrames to link video paths with classification and regression labels.
- Utilize TensorFlow `tf.data.Dataset` pipelines with `AUTOTUNE` for efficient data loading.

### 2. Model Architecture
- **Backbone:** Pre-trained CNN or transformer for video frame feature extraction.
- **Shared Layers:** Dense layers with Batch Normalization and Dropout.
- **Heads:**
  - Classification head: Dense layer with softmax activation (emotion classes).
  - Regression head: Dense layer with 2 outputs (arousal and valence).

### 3. Loss Functions and Metrics
- Classification: Sparse Categorical Crossentropy
- Regression: Concordance Correlation Coefficient (CCC) loss and metrics
- Combined loss: Weighted sum of classification and regression losses

---

## 🚀 Getting Started

### Prerequisites
- Python 3.x
- TensorFlow (GPU-enabled recommended)
- Libraries: `numpy`, `pandas`, `opencv-python`, `decord`, `moviepy`
- Pre-trained model in TensorFlow SavedModel format

### Running the Training Script
- Update dataset folder and model checkpoint paths:
  - `train_uc_set`, `val_uc_set`, `test_uc_set`
  - `output_csv_path` (arousal/valence labels)
  - `model_path` (pre-trained backbone)
    
