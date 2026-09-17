import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from torchvision import models, transforms
import segmentation_models_pytorch as smp

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = Path(__file__).resolve().parent
SEG_MODEL_PATH = BASE_DIR / "seg_model.pth"
CLS_MODEL_PATH = BASE_DIR / "cls_model.pth"
INPUT_DIR = BASE_DIR

SEG_SCALE = 0.3
CLS_SIZE = (480, 480)


def synchronize():
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()


class FastValidator:
    STAGE_NAMES = [
        "preprocessing",
        "segmentation",
        "roi_binarization",
        "classifier_resize",
        "classification",
        "end_to_end",
    ]

    def __init__(self, seg_path, cls_path):
        print(f"Loading and preparing models... DEVICE={DEVICE}")

        self.seg_model = smp.Unet(
            encoder_name="resnet18",
            encoder_weights=None,
            in_channels=3,
            classes=2,
        )
        self.seg_model.load_state_dict(
            torch.load(seg_path, map_location=DEVICE)
        )
        self.seg_model.to(DEVICE).eval()

        self.cls_model = models.efficientnet_v2_s()
        self.cls_model.classifier[1] = nn.Linear(
            self.cls_model.classifier[1].in_features,
            1,
        )
        self.cls_model.load_state_dict(
            torch.load(cls_path, map_location=DEVICE)
        )
        self.cls_model.to(DEVICE).eval()

        self.cls_transform = transforms.Normalize(
            mean=[0.04400077, 0.04400077, 0.04400077],
            std=[0.20090826, 0.20090826, 0.20090826],
        )

        self._warm_up()

    @torch.inference_mode()
    def _warm_up(self):
        print("Warming up models...")

        seg_size = int(1024 * SEG_SCALE)

        dummy_seg = torch.zeros(
            (1, 3, seg_size, seg_size),
            device=DEVICE,
        )
        dummy_cls = torch.zeros(
            (1, 3, CLS_SIZE[0], CLS_SIZE[1]),
            device=DEVICE,
        )

        for _ in range(5):
            _ = self.seg_model(dummy_seg)
            _ = self.cls_model(dummy_cls)

        synchronize()

    def load_all_to_ram(self, input_path):
        print(f"Loading images into RAM: {input_path}")

        input_path = Path(input_path)
        image_files = sorted(
            list(input_path.rglob("*.bmp"))
            + list(input_path.rglob("*.png"))
        )

        image_data = []

        for file_path in tqdm(image_files, desc="Loading"):
            label = 1 if "NG" in file_path.parts else 0

            with Image.open(file_path) as image:
                image = image.convert("RGB").copy()

            image_data.append(
                {
                    "img": image,
                    "label": label,
                    "path": str(file_path),
                }
            )

        return image_data

    @staticmethod
    def summarize_times(stage_times, total_imgs):
        total_mean = np.mean(stage_times["end_to_end"])

        print("\n" + "=" * 95)
        print("⏱️ Processing time by stage")
        print("=" * 95)
        print(
            f"{'Stage':<24}"
            f"{'Mean(ms)':>13}"
            f"{'Median(ms)':>15}"
            f"{'P95(ms)':>13}"
            f"{'Total time(s)':>14}"
            f"{'Ratio(%)':>12}"
        )
        print("-" * 95)

        display_names = {
            "preprocessing": "1. Preprocessing",
            "segmentation": "2. Segmentation",
            "roi_binarization": "3. ROI + Binarization",
            "classifier_resize": "4. Classifier Resize",
            "classification": "5. Classification",
            "end_to_end": "Total End-to-End",
        }

        for stage_name in FastValidator.STAGE_NAMES:
            values = np.asarray(stage_times[stage_name], dtype=np.float64)

            mean_ms = values.mean() * 1000
            median_ms = np.median(values) * 1000
            p95_ms = np.percentile(values, 95) * 1000
            total_sec = values.sum()

            if stage_name == "end_to_end":
                ratio = 100.0
            else:
                ratio = (
                    values.mean() / total_mean * 100
                    if total_mean > 0
                    else 0.0
                )

            print(
                f"{display_names[stage_name]:<24}"
                f"{mean_ms:>13.3f}"
                f"{median_ms:>15.3f}"
                f"{p95_ms:>13.3f}"
                f"{total_sec:>14.3f}"
                f"{ratio:>12.2f}"
            )

        print("-" * 95)

        sum_of_stage_means = sum(
            np.mean(stage_times[name])
            for name in FastValidator.STAGE_NAMES
            if name != "end_to_end"
        )

        unmeasured_overhead = max(total_mean - sum_of_stage_means, 0.0)

        print(
            f"Sum of mean stage times: "
            f"{sum_of_stage_means * 1000:.3f} ms"
        )
        print(
            f"Other Python/timing overhead: "
            f"{unmeasured_overhead * 1000:.3f} ms"
        )
        print(f"Number of measured images: {total_imgs}")
        print("=" * 95)

    @torch.inference_mode()
    def run_benchmark(self, data_list):
        total_imgs = len(data_list)

        if total_imgs == 0:
            print("No images available for benchmarking.")
            return

        correct_count = 0
        tp = fp = tn = fn = 0

        stage_times = {
            name: []
            for name in self.STAGE_NAMES
        }

        print(f"Starting benchmark: {total_imgs} images in total")

        for item in tqdm(data_list, desc="Benchmark"):
            full_img = item["img"]
            label = item["label"]

            synchronize()
            total_start = time.perf_counter()

            stage_start = time.perf_counter()

            width, height = full_img.size
            new_width = max(1, int(SEG_SCALE * width))
            new_height = max(1, int(SEG_SCALE * height))

            resized_img = full_img.resize(
                (new_width, new_height),
                resample=Image.Resampling.BILINEAR,
            )

            img_np = np.array(resized_img, dtype=np.uint8, copy=True)

            img_tensor = (
                torch.from_numpy(img_np.transpose(2, 0, 1))
                .unsqueeze(0)
                .float()
                .to(DEVICE)
                / 255.0
            )

            synchronize()
            stage_times["preprocessing"].append(
                time.perf_counter() - stage_start
            )

            stage_start = time.perf_counter()

            seg_output = self.seg_model(img_tensor)
            seg_mask = seg_output.argmax(dim=1).squeeze(0)

            synchronize()
            stage_times["segmentation"].append(
                time.perf_counter() - stage_start
            )

            stage_start = time.perf_counter()

            roi_tensor = img_tensor.squeeze(0) * seg_mask.unsqueeze(0)
            r_channel = roi_tensor[0]

            mask_exists = r_channel > 0

            if mask_exists.any():
                mean_val = r_channel[mask_exists].mean()
                binary_map = (r_channel > mean_val).float()
            else:
                binary_map = torch.zeros_like(r_channel)

            synchronize()
            stage_times["roi_binarization"].append(
                time.perf_counter() - stage_start
            )

            stage_start = time.perf_counter()

            binary_map = binary_map.unsqueeze(0).unsqueeze(0)

            final_binary = F.interpolate(
                binary_map,
                size=CLS_SIZE,
                mode="bilinear",
                align_corners=False,
            )

            synchronize()
            stage_times["classifier_resize"].append(
                time.perf_counter() - stage_start
            )

            stage_start = time.perf_counter()

            cls_input = final_binary.repeat(1, 3, 1, 1)
            cls_input = self.cls_transform(cls_input)

            cls_output = self.cls_model(cls_input)
            prob = torch.sigmoid(cls_output).item()
            pred = int(prob > 0.5)

            synchronize()
            stage_times["classification"].append(
                time.perf_counter() - stage_start
            )

            synchronize()
            stage_times["end_to_end"].append(
                time.perf_counter() - total_start
            )

            correct_count += int(pred == label)

            if pred == 1 and label == 1:
                tp += 1
            elif pred == 1 and label == 0:
                fp += 1
            elif pred == 0 and label == 0:
                tn += 1
            else:
                fn += 1

        end_to_end = np.asarray(
            stage_times["end_to_end"],
            dtype=np.float64,
        )

        avg_time = end_to_end.mean()
        avg_time_ms = avg_time * 1000
        fps = 1.0 / avg_time if avg_time > 0 else 0.0

        accuracy = correct_count / total_imgs * 100

        print("\n" + "=" * 50)
        print("Final performance report")
        print("=" * 50)
        print(
            f"Accuracy: {accuracy:.2f}% "
            f"({correct_count}/{total_imgs})"
        )
        print(f"Mean End-to-End: {avg_time_ms:.3f} ms/image")
        print(f"FPS: {fps:.2f}")
        print(f"Total measured time: {end_to_end.sum():.3f} s")
        print("=" * 50)

        self.summarize_times(stage_times, total_imgs)


if __name__ == "__main__":
    validator = FastValidator(
        SEG_MODEL_PATH,
        CLS_MODEL_PATH,
    )

    in_memory_data = validator.load_all_to_ram(INPUT_DIR)

    if in_memory_data:
        validator.run_benchmark(in_memory_data)
    else:
        print("No data loaded.")
