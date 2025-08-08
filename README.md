# �� CIFAR-10 Image Classification with CNN & Transfer Learning

## �� Overview

This project focuses on building and evaluating Convolutional Neural Networks (CNNs) to classify images from the CIFAR-10 dataset. We experimented with optimizers (SGD and Adam), model depth, early stopping, and transfer learning with VGG16. A web app is also provided to test the final model.

---

## �� Models and Techniques

- **Baseline CNN with SGD (1-layer and deep)**: The shallow model underfits, while the deeper model improves performance.
- **Deep CNN with Adam + EarlyStopping**: Achieved the best performance (81.6% validation accuracy).
- **Transfer Learning with VGG16**: 73.27% accuracy using frozen layers.
- **Testing**: The final model was tested on 20 unseen images, correctly predicting 15/20.

---

## �� Key Results

| Model             | Accuracy | Precision | Recall | F1-score |
|------------------|----------|-----------|--------|----------|
| Deep CNN + Adam  | 0.8076   | 0.8100    | 0.8076 | 0.8076   |
| Deep CNN + SGD   | 0.7201   | 0.7346    | 0.7201 | 0.7174   |
| VGG16 (frozen)   | 0.7327   |    —      |   —    |    —     |

---

## �� Running the Project
File explanation:
project_1_deep_learning.ipynb - Our code where we build our model.
requirements.txt. - summary of libararies needed to run the code correctly
Multiple model.h5 files - with our saved model results.
app.py - The app file that was used to build the app on HuggingFace spaces. See link below.

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Launch the notebook:
```bash
jupyter notebook project_1_deep_learning.ipynb
```

3. (Optional) Test the prediction app:
👉 https://huggingface.co/spaces/DaanBooy/Image_Predictor

---

## �� Authors

Group 1: Daan, John-Bapiste, Katy

---

## �� License

This project is for educational purposes only.
