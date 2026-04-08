import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class CustomMTLDataset(Dataset):
    def __init__(self, img_dir, label_dir, transform=None):
        """
        img_dir: Folder berisi gambar (.jpg / .png)
        label_dir: Folder berisi label bounding box YOLO (.txt)
        """
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.img_names = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
        
        # Transformasi dasar ke Tensor (Ukuran 640x640 sesuai standar YOLO)
        self.transform = transform or transforms.Compose([
            transforms.Resize((640, 640)),
            transforms.ToTensor(), # Mengubah ke tensor dan normalisasi 0-1
        ])

    def __len__(self):
        return len(self.img_names)

    def _get_illumination_label(self, img_name):
        """
        Fungsi ini menentukan label cahaya berdasarkan nama file atau folder.
        Asumsi: Anda menambahkan penanda di nama file saat memodifikasi COCO.
        Misal: '000123_low.jpg', '000124_over.jpg', '000125_normal.jpg'
        """
        name_lower = img_name.lower()
        if 'low' in name_lower:
            return 0  # 0 untuk Low-Light
        elif 'normal' in name_lower:
            return 1  # 1 untuk Normal
        elif 'over' in name_lower:
            return 2  # 2 untuk Over-Exposed
        else:
            return 1  # Default ke normal jika tidak ada penanda

    def _get_bboxes(self, txt_path):
        """
        Membaca file .txt YOLO (class, x_center, y_center, width, height)
        """
        bboxes = []
        if os.path.exists(txt_path):
            with open(txt_path, 'r') as f:
                for line in f.readlines():
                    data = line.strip().split()
                    if len(data) == 5:
                        # Format: [class_id, x, y, w, h]
                        bboxes.append([float(x) for x in data])
        
        # Jika tidak ada objek di gambar, kembalikan tensor kosong
        if len(bboxes) == 0:
            return torch.zeros((0, 5))
        
        return torch.tensor(bboxes, dtype=torch.float32)

    def __getitem__(self, idx):
        # 1. Ambil Nama File
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)
        
        # Asumsi nama file label sama dengan gambar (misal: image01.jpg -> image01.txt)
        txt_name = os.path.splitext(img_name)[0] + '.txt'
        txt_path = os.path.join(self.label_dir, txt_name)
        
        # 2. Load Gambar
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
            
        # 3. Load Bounding Box Target
        bboxes = self._get_bboxes(txt_path)
        
        # 4. Tentukan Illumination Target
        illum_label = self._get_illumination_label(img_name)
        illum_tensor = torch.tensor(illum_label, dtype=torch.long)
        
        # Output 3 Elemen untuk Multi-Task Learning!
        return image, bboxes, illum_tensor

# Uji Coba Cepat (Test the Dataloader)
if __name__ == "__main__":
    # Ganti dengan path folder dummy Anda untuk mengetes
    dummy_img_dir = "./datasets/train/images"
    dummy_label_dir = "./datasets/train/labels"
    
    # Pastikan foldernya ada sebelum dijalankan
    os.makedirs(dummy_img_dir, exist_ok=True)
    os.makedirs(dummy_label_dir, exist_ok=True)
    
    dataset = CustomMTLDataset(img_dir=dummy_img_dir, label_dir=dummy_label_dir)
    print(f"Total dataset: {len(dataset)} gambar")