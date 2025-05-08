# README.md
# Caltech-101 Fine-tuning with Pretrained ResNet-18

This project fine-tunes a pretrained ResNet-18 model on the Caltech-101 dataset.

## Steps to Run

1. **Install dependencies**
```bash
pip install torch torchvision tensorboard
```

2. **Download Caltech-101**: Place extracted folder at `caltech-101/101_ObjectCategories`

3. **Train**
```bash
python train.py
```

4. **View Training Logs**
```bash
tensorboard --logdir=./runs
```

## Output
- Model saved to `./checkpoints/resnet18_caltech101.pth`
- Training logs saved for TensorBoard
