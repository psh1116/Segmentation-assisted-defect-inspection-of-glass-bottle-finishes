import argparse
import logging
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, random_split, Dataset
from tqdm import tqdm
import segmentation_models_pytorch as smp
import numpy as np
from PIL import Image

# --- 1. Dataset 클래스 정의 (하위 폴더 지원) ---
class BasicDataset(Dataset):
    def __init__(self, images_dir: str, mask_dir: str, scale: float = 1.0):
        self.images_dir = Path(images_dir)
        self.mask_dir = Path(mask_dir)
        self.scale = scale
        
        # 하위 폴더(OK, NG)를 포함하여 모든 이미지 파일 검색
        self.ids = []
        for ext in ['bmp']:
            for file in self.images_dir.rglob(f'*.{ext}'):
                if not file.name.startswith('.'):
                    # 확장자를 제외한 상대 경로 저장 (예: 'OK/image1')
                    relative_path = file.relative_to(self.images_dir).with_suffix('')
                    self.ids.append(str(relative_path))
        
        if not self.ids:
            raise RuntimeError(f'No input files found in {images_dir}')
        logging.info(f'데이터셋 로드 완료: {len(self.ids)} 개의 이미지를 찾았습니다.')

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def preprocess(img, scale, is_mask):
        if isinstance(img, np.ndarray) and not is_mask:
            img = Image.fromarray(img)
        elif isinstance(img, np.ndarray) and is_mask:
            # 마스크가 npy인 경우 이미 array이므로 처리 생략 가능하지만 PIL 통일 위해 변환
            img = Image.fromarray(img)
            
        w, h = img.size
        newW, newH = int(scale * w), int(scale * h)
        img = img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img_array = np.asarray(img)

        if is_mask:
            # 0은 배경, 0보다 큰 모든 값은 1로 처리
            return np.where(img_array > 0.5, 1, 0).astype(np.int64)
        else:
            if img_array.ndim == 2:
                img_array = img_array[np.newaxis, ...]
            else:
                img_array = img_array.transpose((2, 0, 1))
            if (img_array > 1).any():
                img_array = img_array / 255.0
            return img_array

    def __getitem__(self, idx):
        name = self.ids[idx]
        # 원본 이미지 찾기 (확장자 유연하게 대응)
        img_file = list(self.images_dir.glob(name + '.*'))[0]
        # 마스크 이미지 찾기 (.npy 우선 검색)
        mask_file = list(self.mask_dir.glob(name + '.npy'))
        if not mask_file:
            mask_file = list(self.mask_dir.glob(name + '.*'))
        mask_file = mask_file[0]

        img = Image.open(img_file)
        mask = np.load(mask_file) if mask_file.suffix == '.npy' else Image.open(mask_file)

        img = self.preprocess(img, self.scale, is_mask=False)
        mask = self.preprocess(mask, self.scale, is_mask=True)

        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous()
        }

# --- 2. Dice 계수 및 평가 함수 ---
def dice_coeff(input, target, epsilon=1e-6):
    inter = 2 * (input * target).sum(dim=(-1, -2))
    sets_sum = input.sum(dim=(-1, -2)) + target.sum(dim=(-1, -2))
    return ((inter + epsilon) / (sets_sum + epsilon)).mean()

@torch.inference_mode()
def evaluate(model, dataloader, device, amp):
    model.eval()
    dice_score = 0
    for batch in tqdm(dataloader, desc='평가 중', unit='batch', leave=False):
        image, mask_true = batch['image'].to(device), batch['mask'].to(device)
        with torch.autocast(device.type, enabled=amp):
            mask_pred = model(image)
            mask_pred = F.one_hot(mask_pred.argmax(dim=1), model.n_classes).permute(0, 3, 1, 2).float()
            mask_true_oh = F.one_hot(mask_true, model.n_classes).permute(0, 3, 1, 2).float()
            dice_score += dice_coeff(mask_pred[:, 1:], mask_true_oh[:, 1:])
    model.train()
    return dice_score / max(len(dataloader), 1)

# --- 3. 학습 루틴 ---
def train_model(model, device, args):
    # 데이터 경로 설정 (현재 준비된 경로)
    data_root = Path("/workspace/soo/project/nova/data/unet_train")
    dataset = BasicDataset(data_root / 'original', data_root / 'masks', args.scale)
    
    n_val = int(len(dataset) * 0.1)
    train_set, val_set = random_split(dataset, [len(dataset) - n_val, n_val], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_set, shuffle=True, batch_size=args.batch_size, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, shuffle=False, batch_size=args.batch_size, num_workers=4, pin_memory=True)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    grad_scaler = torch.amp.GradScaler('cuda', enabled=args.amp)
    criterion = nn.CrossEntropyLoss()

    logging.info(f"학습 시작: Epochs={args.epochs}, Batch={args.batch_size}, LR={args.lr}, Scale={args.scale}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        with tqdm(total=len(train_set), desc=f'에폭 {epoch}/{args.epochs}', unit='img') as pbar:
            for batch in train_loader:
                images, true_masks = batch['image'].to(device), batch['mask'].to(device)

                with torch.autocast(device.type, enabled=args.amp):
                    masks_pred = model(images)
                    loss = criterion(masks_pred, true_masks)
                    pred_soft = F.softmax(masks_pred, dim=1)
                    true_oh = F.one_hot(true_masks, args.classes).permute(0, 3, 1, 2).float()
                    loss += (1 - dice_coeff(pred_soft[:, 1:], true_oh[:, 1:]))

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                pbar.set_postfix(loss=loss.item())

        val_score = evaluate(model, val_loader, device, args.amp)
        logging.info(f'에폭 {epoch} 완료 - 검증 Dice Score: {val_score:.4f}')
        
        save_path = Path('./seg_checkpoints')
        save_path.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), str(save_path / f'checkpoint_epoch{epoch}.pth'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--learning-rate', type=float, default=1e-4, dest='lr')
    parser.add_argument('--scale', type=float, default=0.2)
    parser.add_argument('--classes', type=int, default=2)
    parser.add_argument('--amp', action='store_true')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = smp.Unet(encoder_name="resnet18", encoder_weights="imagenet", in_channels=3, classes=args.classes)
    model.n_classes = args.classes
    model.to(device)

    train_model(model, device, args)