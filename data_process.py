import os
from torch.utils.data import Dataset
import json
import cv2
from PIL import Image
import numpy as np
import torch
import time
from global_info import GlobalInfo


# 全局变量
SAMPLE_DIR = "samples"
os.makedirs(SAMPLE_DIR, exist_ok=True)


class GazeDataset(Dataset):
    def __init__(self, samples_dir=SAMPLE_DIR, ui_callback=None):
        self.samples_dir = samples_dir
        self.samples = []
        self.all_samples = []
        self.sample_count = 0
        self.ui_callback = ui_callback

    def load_all_samples(self):
        if not os.path.exists(self.samples_dir):
            return
        for file in os.listdir(self.samples_dir):
            if file.endswith(".json"):
                with open(os.path.join(self.samples_dir, file), 'r') as f:
                    data = json.load(f)
                    self.all_samples.append(data)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image_path = sample['image']
        coords = sample['coords']

        image = Image.open(image_path)
        image = image.convert('RGB')
        image = np.array(image)
        # image = cv2.resize(image, (128, 128))
        image = torch.from_numpy(image).float().permute(2, 0, 1) / 255.0

        coords = torch.tensor(coords, dtype=torch.float32)
        return image, coords

    def save_sample(self, image, coords):
        timestamp = int(time.time() * 1000)
        image_path = os.path.join(SAMPLE_DIR, f"sample_{timestamp}.jpg")
        coords_normalized = [coords[0] / GlobalInfo.screen_width, coords[1] / GlobalInfo.screen_height]
        print(coords[0], GlobalInfo.screen_width, coords[1] , GlobalInfo.screen_height)

        # Save image
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        pil_image.save(image_path)

        # Save metadata
        metadata = {
            "image": image_path,
            "coords": coords_normalized,
            "timestamp": timestamp
        }

        with open(os.path.join(SAMPLE_DIR, f"sample_{timestamp}.json"), 'w') as f:
            json.dump(metadata, f)

        self.samples.append(metadata)
        self.sample_count += 1

        if self.ui_callback:
            self.ui_callback("sample_saved", self.sample_count)
        print(f"Sample saved. Total samples: {self.sample_count}")

    def check_sample_upper_limit(self):
        self.load_all_samples()
        file_pairs = []
        if len(self.all_samples) <= GlobalInfo.sample_upper_limit:
            return

        for i in self.all_samples:
            image_path = i['image']
            timestamp = i['timestamp']

            file_pairs.append((image_path, f"sample_{timestamp}.json", timestamp))

        # 按时间戳排序（最旧的在前）
        file_pairs.sort(key=lambda x: x[2])

        for i in file_pairs[:len(file_pairs) - GlobalInfo.sample_upper_limit]:
            oldest_image, oldest_json, _ = i
            # 删除文件
            try:
                os.remove(oldest_image)
                os.remove(oldest_json)
                print(f"Deleted oldest file pair: {oldest_image} and {oldest_json}")
            except OSError as e:
                print(f"Error deleting files: {e}")

        self.all_samples.clear()

    def use_all_samples(self):
        self.load_all_samples()
        self.samples = self.all_samples

    def clear_samples(self):
        self.samples.clear()
