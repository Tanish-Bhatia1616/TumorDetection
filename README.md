# TumorDetection
MRI Tumor Detection using Deep Learning
This project implements a deep learning pipeline for automated MRI brain tumor detection using transfer learning with VGG16. It includes data preprocessing, augmentation, model training, evaluation, and a Streamlit-based user interface for real-time predictions.

 Features
- Dataset Handling: Loads and preprocesses MRI images from training and testing directories.
- Image Augmentation: Brightness and contrast adjustments for better generalization.
- Transfer Learning: Uses VGG16 (pretrained on ImageNet) as the base model.
- Model Training: Fine-tunes selected layers with dropout and dense layers.
- Evaluation: Provides accuracy, loss plots, confusion matrix, and classification report.
- Prediction Function: Detects tumor presence with confidence scores.
- Streamlit UI: User-friendly interface to upload MRI scans and get instant predictions.

Project Structure
├── Training/                # Training dataset (organized by class folders)
├── Testing/                 # Testing dataset (organized by class folders)
├── mri_model.h5             # Saved trained model
├── app.py                   # Streamlit application
├── mri_detection.py         # Core training and evaluation script
└── README.md                # Project documentation



⚙️ Workflow
- Data Loading
- Images are read from Training/ and Testing/ directories.
- Labels are encoded into integers.
- Image Augmentation
- Random brightness and contrast adjustments.
- Normalization to [0,1] range.
- Model Architecture
- Base: VGG16 (frozen layers, partial fine-tuning).
- Added layers: Flatten → Dense(128, ReLU) → Dropout → Dense(Softmax).
- Training
- Optimizer: Adam (lr=0.0001).
- Loss: Sparse categorical crossentropy.
- Metrics: Accuracy.
- Evaluation
- Accuracy/Loss plots.
- Confusion matrix visualization.
- Classification report.
- Prediction & UI
- Function to detect tumor type with confidence score.
- Streamlit app for interactive predictions.

Usage
1. Train the Model
python mri_detection.py

2. Run Streamlit App
streamlit run app.py

3. Upload MRI Image
- Upload .jpg, .jpeg, or .png files.
- The app displays prediction with confidence score.

Results
- Training Accuracy & Loss plotted across epochs.
- Confusion Matrix for test set evaluation.
- Classification Report with precision, recall, and F1-score.

Streamlit Demo
- Upload an MRI image.
- Get prediction: No Tumor or Tumor Detected (type).
- Confidence score displayed.

Technology used
- Python
- TensorFlow / Keras
- NumPy, Matplotlib, Seaborn
- Scikit-learn
- Streamlit
- PIL (Image Processing)

Author
Tanish
Final-year B.Tech (CSE) student, Guru Nanak Dev University
Focused on deep learning, deployment, and practical AI solutions.






