import numpy as np
from PIL import Image
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt

class BasicDataset(torch.utils.data.Dataset):
    @staticmethod
    def preprocess(pil_img, scale, is_mask):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small'
        pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img_ndarray = np.asarray(pil_img)

        if not is_mask:
            if img_ndarray.ndim == 2:
                img_ndarray = img_ndarray[np.newaxis, ...]
            else:
                img_ndarray = img_ndarray.transpose((2, 0, 1))
            
            if img_ndarray.max() > 1:
                img_ndarray = img_ndarray / 255.0
        return img_ndarray

def shift_cam(cam_array, x_offset, y_offset):
    h, w = cam_array.shape
    shifted_cam = np.zeros_like(cam_array)
    src_x_min = max(0, -x_offset)
    src_x_max = min(w, w - x_offset)
    src_y_min = max(0, -y_offset)
    src_y_max = min(h, h - y_offset)
    dst_x_min = max(0, x_offset)
    dst_x_max = min(w, w + x_offset)
    dst_y_min = max(0, y_offset)
    dst_y_max = min(h, h + y_offset)
    if (src_x_max > src_x_min) and (src_y_max > src_y_min):
        shifted_cam[dst_y_min:dst_y_max, dst_x_min:dst_x_max] = \
            cam_array[src_y_min:src_y_max, src_x_min:src_x_max]
    return shifted_cam

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_handles = []
        self.hook_handles.append(self.target_layer.register_forward_hook(self._get_activations_hook))
        self.hook_handles.append(self.target_layer.register_backward_hook(self._get_gradients_hook))

    def _get_activations_hook(self, module, input, output):
        self.activations = output.detach()

    def _get_gradients_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, input_tensor, target_category=None):
        self.model.eval()
        output = self.model(input_tensor)
        if target_category is None:
            target_category = torch.argmax(output, dim=1).item()
        
        target_score = output[0, target_category]
        self.model.zero_grad()
        target_score.backward()

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Failed to get gradients or activations.")

        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = (cam - torch.min(cam)) / (torch.max(cam) - torch.min(cam) + 1e-8)
        return cam, target_category

    def remove_hooks(self):
        for handle in self.hook_handles:
            handle.remove()

def denormalize_image(tensor, mean, std):
    tensor = tensor.clone()
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor.permute(1, 2, 0).cpu().numpy()

def binarize_on_gpu(original_img_tensor: torch.Tensor, mask_tensor: torch.Tensor) -> Image.Image:
    binary_mask_3d = (mask_tensor > 0).unsqueeze(0).to(original_img_tensor.device)
    h, w = mask_tensor.shape
    resized_original = F.interpolate(original_img_tensor, size=(h, w), mode='bilinear', align_corners=False)
    masked_tensor = resized_original * binary_mask_3d
    
    channel = masked_tensor[0, 0, :, :]
    non_zero_pixels = channel[channel > 0]
    
    mean_val = non_zero_pixels.mean() if non_zero_pixels.numel() > 0 else 0

    final_mask = (channel > mean_val).type(torch.uint8) * 255
    final_mask_np = final_mask.cpu().numpy()
    output_img_np = np.stack([final_mask_np]*3, axis=-1)
    
    return Image.fromarray(output_img_np)

# --- 1. 설정 및 모델 로딩 ---
unet_model_path = './Pytorch-UNet/checkpoint/checkpoint_epoch10.pth'
classifier_model_path = './classification_checkpoint/efficientnet_ok_ng_best.pth'
image_path = './data/NG/2.bmp'
scale_factor = 0.5
num_classes = 2
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# PyTorch 최적화 설정
torch.backends.cudnn.benchmark = True

unet = smp.Unet(
    encoder_name="resnet34", classes=num_classes, encoder_depth=5, encoder_weights="imagenet",
    activation=None, decoder_channels=[256, 128, 64, 32, 16],
).to(memory_format=torch.channels_last).to(device)
state_dict_unet = torch.load(unet_model_path, map_location=device)
state_dict_unet.pop('mask_values', None)
unet.load_state_dict(state_dict_unet)
unet.eval()

classifier = models.efficientnet_v2_l(weights=None)
num_ftrs = classifier.classifier[1].in_features
classifier.classifier = nn.Sequential(nn.Dropout(0.4), nn.Linear(num_ftrs, 2))
classifier.load_state_dict(torch.load(classifier_model_path, map_location=device))
classifier = classifier.to(device)
classifier.eval()

mean, std = [0.04297327] * 3, [0.20066201] * 3
classify_transform = transforms.Compose([
    transforms.Resize((480, 480)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])
classes = ['NG', 'OK']
print("--- 모델 및 설정 준비 완료 ---")


# --- 2. 원본 이미지 로딩 및 시각화 ---
original_img = Image.open(image_path).convert('RGB')
plt.figure(figsize=(6, 6))
plt.imshow(original_img)
plt.title('1. Original Image')
plt.axis('off')
plt.show()

# --- 3. 세그멘테이션 및 결과 시각화 ---
seg_input_tensor = torch.from_numpy(BasicDataset.preprocess(original_img, scale_factor, is_mask=False))
seg_input_tensor = seg_input_tensor.unsqueeze(0).to(device=device, dtype=torch.float32, memory_format=torch.channels_last)

torch.cuda.synchronize()
start_segmentation = time.perf_counter()
with torch.no_grad(), torch.autocast(device_type=device.type):
    seg_output = unet(seg_input_tensor)
torch.cuda.synchronize()
time_segmentation = time.perf_counter() - start_segmentation
print(f"1. Segmentation : {time_segmentation:.4f}초")

seg_output_resized = F.interpolate(seg_output.float(), size=seg_input_tensor.shape[2:], mode='bilinear', align_corners=False)
probs = F.softmax(seg_output_resized, dim=1)
mask_gpu = torch.argmax(probs, dim=1).squeeze()

plt.figure(figsize=(6, 6))
plt.imshow(original_img.resize((mask_gpu.shape[1], mask_gpu.shape[0]), Image.BICUBIC))
plt.imshow(mask_gpu.cpu().numpy(), cmap='jet', alpha=0.5)
plt.title('2. Segmentation ')
plt.axis('off')
plt.show()

# --- 4. 이진화 및 결과 시각화 ---
start_binarization = time.perf_counter()
binarized_img = binarize_on_gpu(seg_input_tensor, mask_gpu)
time_binarization = time.perf_counter() - start_binarization
print(f"2. Binarization : {time_binarization:.4f}초")

plt.figure(figsize=(6, 6))
plt.imshow(binarized_img)
plt.title('3. Binarization')
plt.axis('off')
plt.show()



# --- 5. 분류 및 결과 출력 ---
cls_input_tensor = classify_transform(binarized_img).unsqueeze(0).to(device)
torch.cuda.synchronize()
start_classification = time.perf_counter()
with torch.no_grad(), torch.autocast(device_type=device.type):
    cls_output = classifier(cls_input_tensor)
torch.cuda.synchronize()
time_classification = time.perf_counter() - start_classification
predicted_idx = torch.argmax(cls_output, dim=1).item()
predicted_class_name = classes[predicted_idx]
print(f"3. Classification : {time_classification:.4f}초")
print("\n" + "="*40)
print(f" 판정 결과: {predicted_class_name} ")
print("="*40 + "\n")



# --- 6. Grad-CAM 생성 및 시각화 ---
target_layer = classifier.features[-2][0]
grad_cam = GradCAM(model=classifier, target_layer=target_layer)
cam, _ = grad_cam(cls_input_tensor, target_category=predicted_idx)
grad_cam.remove_hooks()

cam_resized = F.interpolate(cam, size=(480, 480), mode='bilinear', align_corners=False)
cam_np = cam_resized.squeeze().cpu().numpy()
shifted_cam_np = shift_cam(cam_np, -25, -20)
denorm_img = denormalize_image(cls_input_tensor.squeeze(), mean, std)
denorm_img = np.clip(denorm_img, 0, 1)

plt.figure(figsize=(6, 6))
plt.imshow(denorm_img)
plt.imshow(shifted_cam_np, cmap='jet', alpha=0.5)
plt.title(f'4. Grad-CAM Visualization\n(Prediction: {predicted_class_name})')
plt.axis('off')
plt.show()

total_time = time_segmentation + time_binarization + time_classification
print("\n" + "="*40)
print(" 전체 처리 시간 요약")
print("----------------------------------------")
print(f"Segmentation : {time_segmentation:.4f}초")
print(f"Binarization   : {time_binarization:.4f}초")
print(f"Classification : {time_classification:.4f}초")
print("----------------------------------------")
print(f"총 처리 시간      : {total_time:.4f}초")
print("="*40 + "\n")