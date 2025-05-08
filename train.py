import torch
from data_loader import get_data_loaders
from model import get_model
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os


def train_model(model, train_loader, val_loader, device, epochs=20, lr=0.001, ckpt_dir='checkpoints', log_dir='runs'):
    writer = SummaryWriter(log_dir=log_dir)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr,weight_decay=1e-4)

    best_acc = 0.0
    os.makedirs(ckpt_dir, exist_ok=True)

    for epoch in range(epochs):
        model.train()
        train_loss, correct, total = 0, 0, 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

        avg_train_loss = train_loss / total
        train_acc = correct / total
        writer.add_scalar('Loss/Train', avg_train_loss, epoch)
        writer.add_scalar('Accuracy/Train', train_acc, epoch)

        # 验证
        model.eval()
        val_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)

        avg_val_loss = val_loss / total
        val_acc = correct / total
        writer.add_scalar('Loss/Val', avg_val_loss, epoch)
        writer.add_scalar('Accuracy/Val', val_acc, epoch)

        print(f'Epoch {epoch + 1}: Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}')

        if val_acc > best_acc:
            torch.save(model.state_dict(), os.path.join(ckpt_dir, 'best_model.pth'))
            best_acc = val_acc

    writer.close()


# === 入口 ===

if __name__ == '__main__':
    data_dir = './caltech-101/101_ObjectCategories'
    batch_size = 128
    epochs = 20
    lr = 0.001

    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

    train_loader, val_loader = get_data_loaders(data_dir, batch_size)
    model = get_model(num_classes=101, dropout_rate=0.5)

    train_model(model, train_loader, val_loader, device, epochs, lr)