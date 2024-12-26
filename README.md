# CAPTCHA Text Recognition using CRNN and BiGRU

This project focuses on recognizing CAPTCHA text using a deep learning approach, employing Convolutional Recurrent Neural Networks (CRNN) and Bidirectional Gated Recurrent Units (BiGRU). The implementation utilizes PyTorch for model training and evaluation, and includes data preprocessing, model design, and performance analysis.

## Features
- Preprocessing CAPTCHA images for training.
- Training a CRNN-BiGRU-based model for text recognition.
- Performance evaluation using metrics such as accuracy.
- End-to-end pipeline from data preparation to prediction.

## Technologies Used
The project uses the following Python libraries:
- PyTorch
- torchvision
- numpy
- pandas
- matplotlib
- sklearn
- tqdm
- PIL (Pillow)

## Setup Instructions

### Prerequisites
- Python 3.7+
- `pip` for installing required libraries.

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yuvvantalreja/CAPTCHA-Image-Recognition-using-CRNN-and-BiGRU.git
   cd CAPTCHA-Image-Recognition-using-CRNN-and-BiGRU
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Notebook
1. Open the Jupyter notebook:
   ```bash
   jupyter notebook CAPTCHA_Text_Recognition.ipynb
   ```
2. Follow the steps in the notebook to preprocess data, train the model, and evaluate performance.

## Key Functions
The following functions are defined in the project:
- **`weights_init`**: Initializes model weights.
- **`encode_text_batch`**: Encodes text labels into a format suitable for the model.
- **`compute_loss`**: Computes the loss for model optimization.
- **`decode_predictions`**: Decodes model outputs into human-readable text.
- **`remove_duplicates`**: Processes predictions to remove repeated characters.
- **`correct_prediction`**: Checks for accuracy of predictions.
- **`print_total_time`**: Outputs the total runtime of the training process.

## Model Architecture
The model consists of:
1. A Convolutional Neural Network (CNN) for feature extraction.
2. A Recurrent Neural Network (RNN) with BiGRU for sequence modeling.
3. A fully connected layer for final text prediction.

## Dataset
The project uses a CAPTCHA dataset containing labeled images of CAPTCHA text. The dataset is split into training and testing subsets. Data augmentation techniques are applied for better generalization.

## Results
The model achieves a high accuracy in recognizing CAPTCHA text, demonstrating its effectiveness for real-world CAPTCHA-solving applications.

## References
- [CRNN Paper](https://arxiv.org/abs/1507.05717)
- [PyTorch Documentation](https://pytorch.org/docs/)
