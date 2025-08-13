# CAPTCHA Text Recognition using CRNN and BiGRU

This project focuses on recognizing CAPTCHA text using a deep learning approach, employing Convolutional Recurrent Neural Networks and Bidirectional Gated Recurrent Units. It combines a truncated ResNet‑18 visual encoder with BiGRU, a common approach for OCR in documents and images

<img width="400" height="400" alt="Screenshot 2025-08-14 at 12 14 27 AM" src="https://github.com/user-attachments/assets/f3999dcd-e716-429a-a374-981c9215d90e" />
<img width="450" height="450" alt="Screenshot 2025-08-13 at 11 29 38 PM" src="https://github.com/user-attachments/assets/d0c8e3ac-e413-417d-b2c9-a966432893a1" />


## Key Features
- **Backbone**: `resnet18(pretrained=True)` truncated before the last 3 blocks
- **Sequence model**: Two-layer BiGRU with hidden size 256 and residual summation after the first BiGRU

## Dataset
Uses [CAPTCHA dataset] (https://www.kaggle.com/datasets/fournierp/captcha-version-2-images) containing labeled images of CAPTCHA text. The dataset is split into training and testing subsets. Data augmentation techniques are applied for better generalization.

## Results
The model achieves a high accuracy in recognizing CAPTCHA text, demonstrating its effectiveness for real-world CAPTCHA-solving applications.

## Acknowledgements
- https://github.com/EVOL-ution/Captcha-Recognition-using-CRNN
- https://github.com/GitYCC/crnn-pytorch

## References
- [CRNN Paper](https://arxiv.org/abs/1507.05717)
- [PyTorch Documentation](https://pytorch.org/docs/)

## Contributions
I'm open to any new ideas. Just open a PR!

Made with ❤️ in Pittsburgh
