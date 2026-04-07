import cv2
import numpy as np
import os
import random
import shutil
from pathlib import Path

# ================= KONFIGURASI =================
# Pastikan ini mengarah ke folder GAMBAR normal Anda
INPUT_IMG_FOLDER = 'dataset_coco/normal/images'   
# Pastikan ini mengarah ke folder LABEL normal Anda
INPUT_LBL_FOLDER = 'dataset_coco/normal/labels'   

# Output Folder (Script akan otomatis membuat subfolder images & labels didalamnya)
OUTPUT_OE_BASE = 'dataset_coco/overexposed' 
OUTPUT_LL_BASE = 'dataset_coco/lowlight'    

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp']

# ================= SETUP FOLDER OUTPUT =================
# Kita buat struktur standar YOLO agar rapi
# dataset/overexposed/images
# dataset/overexposed/labels
OE_IMG_DIR = os.path.join(OUTPUT_OE_BASE, 'images')
OE_LBL_DIR = os.path.join(OUTPUT_OE_BASE, 'labels')
LL_IMG_DIR = os.path.join(OUTPUT_LL_BASE, 'images')
LL_LBL_DIR = os.path.join(OUTPUT_LL_BASE, 'labels')

for d in [OE_IMG_DIR, OE_LBL_DIR, LL_IMG_DIR, LL_LBL_DIR]:
    Path(d).mkdir(parents=True, exist_ok=True)

class LightAugmentor:
    def __init__(self):
        pass

    def add_noise(self, image, intensity_range=(10, 40)):
        """Menambahkan Gaussian Noise"""
        row, col, ch = image.shape
        mean = 0
        sigma = random.uniform(intensity_range[0], intensity_range[1]) ** 0.5
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        gauss = gauss.reshape(row, col, ch)
        noisy = image.astype('float32') + gauss
        return np.clip(noisy, 0, 255).astype('uint8')

    def adjust_gamma(self, image, gamma):
        """
        Rumus Gamma yang SUDAH DIPERBAIKI:
        Output = Input ^ gamma
        """
        table = np.array([((i / 255.0) ** gamma) * 255 
                          for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(image, table)

    def generate_overexposed(self, image):
        method = random.choice(['gamma', 'linear'])
        if method == 'gamma':
            gamma = random.uniform(0.3, 0.6) 
            return self.adjust_gamma(image, gamma), f"gamma_{gamma:.2f}"
        else:
            alpha = random.uniform(1.2, 1.8)
            beta = random.randint(30, 80)
            return cv2.convertScaleAbs(image, alpha=alpha, beta=beta), f"linear_{alpha:.2f}"

    def generate_lowlight(self, image):
        method = random.choice(['gamma', 'linear'])
        if method == 'gamma':
            gamma = random.uniform(2.5, 4.0)
            dark_img = self.adjust_gamma(image, gamma)
            suffix = f"gamma_{gamma:.2f}"
        else:
            alpha = random.uniform(0.1, 0.4) 
            beta = random.randint(-30, -10)
            dark_img = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
            suffix = f"linear_{alpha:.2f}"
        
        final_img = self.add_noise(dark_img)
        return final_img, suffix

def main():
    augmentor = LightAugmentor()
    
    # Cek folder input
    if not os.path.exists(INPUT_IMG_FOLDER):
        print(f"Error: Folder Gambar '{INPUT_IMG_FOLDER}' tidak ditemukan.")
        return
    if not os.path.exists(INPUT_LBL_FOLDER):
        print(f"Warning: Folder Label '{INPUT_LBL_FOLDER}' tidak ditemukan. Augmentasi akan berjalan tanpa label.")

    # Ambil semua file gambar
    files = [f for f in os.listdir(INPUT_IMG_FOLDER) 
             if os.path.splitext(f)[1].lower() in IMG_EXTENSIONS]
    
    print(f"Memproses {len(files)} gambar normal beserta labelnya...")
    
    count = 0
    count_lbl = 0

    for filename in files:
        img_path = os.path.join(INPUT_IMG_FOLDER, filename)
        img = cv2.imread(img_path)
        
        if img is None: continue
            
        name, ext = os.path.splitext(filename)

        # Cari Label Asli (.txt)
        lbl_filename = name + ".txt"
        src_lbl_path = os.path.join(INPUT_LBL_FOLDER, lbl_filename)
        has_label = os.path.exists(src_lbl_path)

        # === 1. PROSES OVER-EXPOSED ===
        oe_img, oe_param = augmentor.generate_overexposed(img)
        # Nama file baru
        oe_name_base = f"{name}_OE_{oe_param}"
        oe_img_name = oe_name_base + ext
        oe_lbl_name = oe_name_base + ".txt"
        
        # Simpan Gambar
        cv2.imwrite(os.path.join(OE_IMG_DIR, oe_img_name), oe_img)
        # Simpan Label (Copy dari asli)
        if has_label:
            shutil.copy(src_lbl_path, os.path.join(OE_LBL_DIR, oe_lbl_name))

        # === 2. PROSES LOW-LIGHT ===
        ll_img, ll_param = augmentor.generate_lowlight(img)
        # Nama file baru
        ll_name_base = f"{name}_LL_{ll_param}"
        ll_img_name = ll_name_base + ext
        ll_lbl_name = ll_name_base + ".txt"

        # Simpan Gambar
        cv2.imwrite(os.path.join(LL_IMG_DIR, ll_img_name), ll_img)
        # Simpan Label (Copy dari asli)
        if has_label:
            shutil.copy(src_lbl_path, os.path.join(LL_LBL_DIR, ll_lbl_name))
            count_lbl += 1

        count += 1
        if count % 50 == 0:
            print(f"Selesai memproses {count} gambar...")

    print(f"\nSelesai Total!")
    print(f"Output Over-Exposed : {OUTPUT_OE_BASE} (Images & Labels)")
    print(f"Output Low-Light    : {OUTPUT_LL_BASE} (Images & Labels)")
    print(f"Total Gambar : {count * 2} file baru")
    print(f"Total Label  : {count_lbl * 2} file baru")

if __name__ == "__main__":
    main()