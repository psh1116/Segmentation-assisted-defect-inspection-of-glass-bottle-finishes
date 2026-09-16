import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms
from PIL import Image
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import cv2
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
log_dir = os.path.join(BASE_DIR, 'log', 'logs_0.0.1')
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, 'training_log.txt')
logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

checkpoint_dir = os.path.join(BASE_DIR, 'data', 'checkpoint', 'checkpoints_0.0.1')
plot_dir = os.path.join(BASE_DIR, 'plot', 'plots_0.0.1')
cache_base_dir = os.path.join(BASE_DIR, 'data', 'cache')
os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)
os.makedirs(cache_base_dir, exist_ok=True)

class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=0.2, gamma=2.0, reduction='mean'):
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        loss = (1 - p_t) ** self.gamma * bce_loss

        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

class AddSaltAndPepperNoise(object):
    def __init__(self, amount=0.01, salt_vs_pepper=0.7):
        self.amount = amount
        self.salt_vs_pepper = salt_vs_pepper

    def __call__(self, tensor):
        output = tensor.clone()
        c, h, w = tensor.shape
        num_noise = int(self.amount * h * w)
        num_salt = int(num_noise * self.salt_vs_pepper)
        salt_coords = [torch.randint(0, d, (num_salt,)) for d in (h, w)]
        output[:, salt_coords[0], salt_coords[1]] = 1.0
        num_pepper = num_noise - num_salt
        pepper_coords = [torch.randint(0, d, (num_pepper,)) for d in (h, w)]
        output[:, pepper_coords[0], pepper_coords[1]] = 0.0
        return output

    def __repr__(self):
        return f'{self.__class__.__name__}(amount={self.amount}, salt_vs_pepper={self.salt_vs_pepper})'

def update_plots(plot_dir, train_losses, val_losses, train_accs, val_accs, lr_history):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label='Train Loss', color='blue')
    plt.plot(epochs, val_losses, label='Val Loss', color='red')
    plt.title(f'Training and Validation Loss (Epoch {len(train_losses)})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'progress_loss.png'))
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_accs, label='Train Acc', color='blue')
    plt.plot(epochs, val_accs, label='Val Acc', color='red')
    plt.title(f'Training and Validation Accuracy (Epoch {len(train_accs)})')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'progress_accuracy.png'))
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, lr_history, label='Learning Rate', color='green')
    plt.title(f'Learning Rate Schedule (Epoch {len(lr_history)})')
    plt.xlabel('Epoch')
    plt.ylabel('LR')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'progress_lr.png'))
    plt.close()

def plot_confusion_matrix(y_true, y_pred, classes, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.savefig(os.path.join(plot_dir, 'confusion_matrix.svg'), format='svg', bbox_inches='tight')
    plt.close()

mean, std = [0.04297327] * 3, [0.20066201] * 3
data_transforms = {
    'train': transforms.Compose([
        transforms.ColorJitter(brightness=0.3, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.RandomApply([AddSaltAndPepperNoise(amount=0.004)], p=0.6),
        transforms.Normalize(mean=mean, std=std)
    ]),
    'val': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ]),
    'test': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
}

class EfficientCachedImageDataset(Dataset):
    def __init__(self, root_dir, transform=None, cache_base_dir=None):
        self.root_dir = root_dir
        self.classes = ['NG', 'OK']
        self.transform = transform
        self.image_paths = []
        self.labels = []

        self.cache_dir = os.path.join(cache_base_dir, os.path.basename(root_dir) + "_npy_cache")

        for label, class_name in enumerate(self.classes):
            class_dir = os.path.join(self.root_dir, class_name)
            if not os.path.isdir(class_dir):
                print(f"Warning: Directory not found {class_dir}")
                continue
            paths = [os.path.join(class_dir, f) for f in os.listdir(class_dir) if f.endswith('.bmp')]
            self.image_paths.extend(paths)
            self.labels.extend([label] * len(paths))

        if not os.path.exists(self.cache_dir) or len(os.listdir(self.cache_dir)) != len(self.image_paths):
            print(f"Cache not found or incomplete. Caching images to {self.cache_dir} as .npy files...")
            os.makedirs(self.cache_dir, exist_ok=True)
            for i, img_path in enumerate(tqdm(self.image_paths, desc=f"Caching {os.path.basename(root_dir)}")):
                try:
                    img = cv2.imread(img_path)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    np.save(os.path.join(self.cache_dir, f"{i}.npy"), img)
                except Exception as e:
                    print(f"Error caching {img_path}: {e}")
        else:
            print(f"Found complete cache for {os.path.basename(root_dir)} at {self.cache_dir}.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        cached_img_path = os.path.join(self.cache_dir, f"{idx}.npy")
        try:
            img_array = np.load(cached_img_path)
            img = Image.fromarray(img_array)
        except Exception as e:
            print(f"Error loading cached image {cached_img_path}: {e}")
            return torch.zeros(3, 480, 480), -1

        label = self.labels[idx]

        if self.transform:
            img = self.transform(img)

        return img, label

data_dir = os.path.join(BASE_DIR, 'data', 'binarized')

train_dataset = EfficientCachedImageDataset(os.path.join(data_dir, 'train'), transform=data_transforms['train'], cache_base_dir=cache_base_dir)
val_dataset = EfficientCachedImageDataset(os.path.join(data_dir, 'val'), transform=data_transforms['val'], cache_base_dir=cache_base_dir)
test_dataset = EfficientCachedImageDataset(os.path.join(data_dir, 'test'), transform=data_transforms['test'], cache_base_dir=cache_base_dir)

train_labels = train_dataset.labels
class_counts = np.bincount(train_labels)

print(f"Train set class distribution: NG={class_counts[0]}, OK={class_counts[1]}")
logging.info(f"Train set class distribution: NG={class_counts[0]}, OK={class_counts[1]}")

batch_size = 42
class_weights_sampler = 1. / torch.tensor(class_counts, dtype=torch.float)
sample_weights = class_weights_sampler[train_labels]
sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=8, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
logging.info(f"Using device: {device}")

model = models.efficientnet_v2_s(weights='IMAGENET1K_V1')
num_ftrs = model.classifier[1].in_features
model.classifier = nn.Sequential(
    nn.Dropout(p=0.2),
    nn.Linear(num_ftrs, 1)
)

model = model.to(device)

class_weights = torch.tensor([5.1, 0.9], dtype=torch.float).to(device)
criterion = BinaryFocalLoss(alpha=0.25, gamma=4.0)
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=100,
    eta_min=1e-6
)


def train_model(model, criterion, train_loader, val_loader, test_loader, optimizer, scheduler, num_epochs=50, patience=15):
    best_acc = 0.0
    best_model_wts = model.state_dict()
    trigger_times = 0

    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    lr_history = []

    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}\n' + '-' * 10)
        logging.info(f'Epoch {epoch+1}/{num_epochs}')

        model.train()
        running_loss, running_corrects = 0.0, 0
        for inputs, labels in tqdm(train_loader, desc="Training"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)

            labels_float = labels.unsqueeze(1).float()
            loss = criterion(outputs, labels_float)

            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels_float.data)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc.item())

        current_lr = optimizer.param_groups[0]['lr']
        lr_history.append(current_lr)
        print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} LR: {current_lr:.6f}')
        logging.info(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} LR: {current_lr:.6f}')

        model.eval()
        running_loss, running_corrects = 0.0, 0
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc="Validation"):
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)

                labels_float = labels.unsqueeze(1).float()
                loss = criterion(outputs, labels_float)

                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels_float.data)

        epoch_val_loss = running_loss / len(val_loader.dataset)
        epoch_val_acc = running_corrects.double() / len(val_loader.dataset)
        val_losses.append(epoch_val_loss)
        val_accs.append(epoch_val_acc.item())
        print(f'Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.4f}')
        logging.info(f'Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.4f}')

        scheduler.step()

        update_plots(plot_dir, train_losses, val_losses, train_accs, val_accs, lr_history)
        print(f"Plots updated in {plot_dir}")

        if epoch_val_acc > best_acc:
            best_acc = epoch_val_acc
            best_model_wts = model.state_dict()
            torch.save(best_model_wts, os.path.join(checkpoint_dir, 'efficientnet_ok_ng_best.pth'))
            trigger_times = 0
            print(f'Best model saved at epoch {epoch+1} with accuracy: {best_acc:.4f}')
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print(f'Early stopping triggered after {epoch+1} epochs!')
                logging.info(f'Early stopping triggered after {epoch+1} epochs!')
                break

    print(f'Best val Acc: {best_acc:.4f}')
    logging.info(f'Best val Acc: {best_acc:.4f}')

    model.load_state_dict(torch.load(os.path.join(checkpoint_dir, 'efficientnet_ok_ng_best.pth')))
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Testing"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().view(-1).numpy())

    plot_confusion_matrix(y_true, y_pred, classes=['NG', 'OK'], title='Confusion Matrix')

    return model

if __name__ == "__main__":
    model = train_model(model, criterion, train_loader, val_loader, test_loader, optimizer, scheduler, num_epochs=100, patience=15)

    final_model_path = os.path.join(checkpoint_dir, 'efficientnet_ok_ng_final.pth')
    torch.save(model.state_dict(), final_model_path)
    logging.info(f'Saved final model to {final_model_path}')
    print(f'Final model saved to {final_model_path}')}]}                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    