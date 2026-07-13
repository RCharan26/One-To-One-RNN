# 🍕 Food Category Recognition System

A complete, production-grade image classification application that categorizes food photos into 11 distinct classes. Built using **TensorFlow/Keras**, **MobileNetV2** (Transfer Learning), and **Streamlit** with a stunning glassmorphic UI.

---

## 📖 Table of Contents
- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [Dataset Information](#dataset-information)
- [Model Architecture](#model-architecture)
- [Installation & Setup](#installation--setup)
- [How to Train the Model](#how-to-train-the-model)
- [How to Run Predictions via CLI](#how-to-run-predictions-via-cli)
- [How to Start the Streamlit Web Application](#how-to-start-the-streamlit-web-application)
- [Future Enhancements](#future-enhancements)

---

## 🔍 Project Overview
This project delivers an end-to-end Deep Learning pipeline to automatically recognize categories of food images. Recognizing food categories from photos is a critical first step for automated dietary tracking, smart kitchen appliances, and food service automation. 

By leveraging pre-trained weights from **MobileNetV2** and combining them with custom regularization layers, this model achieves rapid convergence and high generalization accuracy, even on standard CPU setups. The system is designed to be fully modular, compliant with clean-code standards, and offers immediately deployable components.

---

## ✨ Key Features
- **Modern Glassmorphic Web App**: Includes visual effects, CSS neon gradients, layout responsive designs, and light/dark theme modes.
- **Top-3 Probability Visualizations**: Real-time rendering of prediction distributions using Matplotlib.
- **Batch Classification**: Upload multiple food photos at once, run inference sequentially, view previews in a layout grid, and export predictions as a CSV report.
- **Session Prediction History**: Locally caches classifications during user sessions, permitting log downloads.
- **Calorie & Nutritional Insights**: Returns energy estimations and advice cards for all 11 food classes.
- **Independent CLI Predictor**: Light CLI interface (`predict.py`) for automated background execution or fast inference checking.
- **Demo Mode**: Includes a filename-based index loader simulator fallback in Streamlit if the model has not been trained yet.

---

## 📁 Dataset Information
The project uses the standard **Food-11 dataset** containing 16,647 images partitioned into training, validation, and evaluation subsets. 

### Flat Directory Setup
In this repository, files are stored flatly inside the directories rather than class subfolders:
```
food11 dataset/
├── training/
│   ├── 0_0.jpg
│   ├── 0_1.jpg
│   ├── 10_279.jpg
│   └── ...
├── validation/
│   └── ...
└── evaluation/
    └── ...
```
Filenames start with `<class_index>_` representing the target category. The training script parses filenames, maps index integers to class names using Pandas DataFrames, and feeds them to the training pipeline using the robust `flow_from_dataframe` function.

### Category Mapping
The 11 categories in the dataset are indexed as follows:
| Index | Class Name | Funny Remark Emoji |
|---|---|---|
| **0** | Bread | "Fresh from the bakery! 🥖" |
| **1** | Dairy Product | "Calcium Power! 🥛" |
| **2** | Dessert | "Too sweet to resist! 🍰" |
| **3** | Egg | "Egg-cellent choice! 🥚" |
| **4** | Fried Food | "Crispy happiness! 🍟" |
| **5** | Meat | "Protein loaded! 🥩" |
| **6** | Noodles/Pasta | "Slurp Mode Activated! 🍜" |
| **7** | Rice | "Every meal's best friend! 🍚" |
| **8** | Seafood | "Straight from the ocean! 🦐" |
| **9** | Soup | "Warm hugs in a bowl! 🥣" |
| **10** | Vegetable/Fruit | "Healthy eating unlocked! 🥗" |

---

## 🧠 Model Architecture
We use **Transfer Learning** with **MobileNetV2** (pre-trained on ImageNet) to extract high-level feature mappings from 224x224 input images:

1. **Pre-processing Input**: Rescaling values to `[0, 1]`.
2. **Augmentation Layers (ImageDataGenerator)**:
   - Rotation Range: 20°
   - Zoom Range: 0.2
   - Horizontal Flip: Enabled
   - Width/Height Shift: 0.2
3. **Base Extractor**: MobileNetV2 (layers frozen to protect weights).
4. **Classification Head**:
   - `GlobalAveragePooling2D` (reduces features to a 1D vector).
   - `Dense(256)` layer with ReLU activation.
   - `BatchNormalization` (speeds up training and stabilizes activations).
   - `Dropout(0.5)` (prevents co-dependency/overfitting).
   - `Dense(11)` layer with Softmax activation (predicts probabilities across the 11 classes).

---

## ⚙️ Installation & Setup

1. **Clone the Repository**:
   Make sure you are inside the project workspace directory:
   ```bash
   cd FoodRecognition/
   ```

2. **Install Package Dependencies**:
   Install all required python packages:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🏋️ How to Train the Model

To train the classification neural network, run the `train.py` script. The training pipeline automatically parses the dataset, builds generators, compiles the architecture, runs fitting, evaluates on the test sets, and outputs pickled stats and graphs.

### Command options:
- Run normal training (20 epochs by default, batch size 32):
  ```bash
  python train.py
  ```
- Customize epochs and batch size:
  ```bash
  python train.py --epochs 25 --batch-size 64
  ```
- **Quick Sanity Check (Recommended for testing setup)**:
  Run a downsampled test using only a small sample of images for 2 epochs:
  ```bash
  python train.py --quick
  ```

### Generated Files after Training:
- `food_model.keras` (Keras model file containing architecture and trained weights).
- `class_names.pkl` (Pickled list of class labels in index order).
- `accuracy.pkl` (Training, validation, and evaluation final accuracies).
- `history.pkl` (Dictionary containing accuracy and loss histories per epoch).
- `graphs/accuracy.png` and `graphs/loss.png` (Training curve PNG plots).
- `graphs/confusion_matrix.png` (Confusion Matrix heatmap image).
- `graphs/classification_report.txt` (Text evaluation report per class).

---

## 🎯 How to Run Predictions via CLI
Use the `predict.py` script to run classification predictions on individual images directly from the command terminal.

### Command syntax:
```bash
python predict.py <path_to_image>
```

### Example:
```bash
python predict.py "food11 dataset/evaluation/0_5.jpg"
```

### CLI Output Example:
```
Loading model from food_model.keras...
Loaded class categories: ['Bread', 'Dairy Product', 'Dessert', 'Egg', 'Fried Food', 'Meat', 'Noodles/Pasta', 'Rice', 'Seafood', 'Soup', 'Vegetable/Fruit']
Running model prediction...

=============================================
🍕 INFERENCE RESULT 🍕
=============================================
Predicted Class: BREAD
Confidence:      92.45%
Remarks:         "Fresh from the bakery! 🥖"
---------------------------------------------
Top 3 Predictions:
 1. Bread: 92.45%
 2. Dessert: 4.10%
 3. Fried Food: 1.25%
=============================================
```

---

## ⚡ How to Start the Streamlit Web Application
To start the glassmorphic Streamlit application dashboard:

```bash
streamlit run app.py
```

### Dashboard Tabs:
1. **🏠 Home (Prediction)**: Upload individual photos, view prediction cards with remarks and progress bars, view calorie metrics, examine probability charts, and download text prediction summaries.
2. **📂 Batch Prediction**: Ingest multiple files, display outcomes sequentially, and export predictions as a `.csv` log.
3. **📊 Model Analytics**: Inspect training curves, accuracy indicators, text classification reports, and the confusion matrix.
4. **💡 Food facts & Nutrition**: Access a reference index of all food classes alongside estimated calorie counts, nutrition advice, and random food trivia generator.
5. **📜 Prediction History**: Review table log of recent classifications, clear sessions, or download logs as CSV.

---

## 🔮 Future Enhancements
- **Fine-Tuning Base Layers**: Unfreeze top layers of MobileNetV2 with low learning rate to squeeze additional accuracy percentage points.
- **Model Quantization**: Convert the Keras model to TFLite format to decrease file size and speed up local inference on mobile applications.
- **Recipes Integration**: Fetch custom recipes using public APIs based on the predicted food ingredients.
- **Multi-Label Segmentation**: Detect and segment multiple food items on a single plate instead of outputting single classifications.
