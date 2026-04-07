import os
import shutil
import random
from pathlib import Path
from tqdm import tqdm  # Library untuk loading bar

# ================= KONFIGURASI (UBAH DISINI) =================
# 1. Lokasi folder dataset MENTAH Anda (yang berisi normal, lowlight, dll)
#    Gunakan r'' agar aman di Windows
SOURCE_DIR = r'D:\Python\Re_Train_YOLO\dataset_coco' 

# 2. Lokasi folder tujuan (Dataset SIAP TRAIN)
#    Script akan membuat folder ini otomatis
DEST_DIR = r'D:\Python\Re_Train_YOLO\dataset_coco_siap_train'

# 3. Kategori folder sumber yang mau diambil
CATEGORIES = ['normal', 'lowlight', 'overexposed']

# 4. Persentase Validasi (0.2 artinya 20% Val, 80% Train)
VAL_RATIO = 0.2

# =============================================================

def split_data():
    # Buat struktur folder tujuan
    for split in ['train', 'val']:
        for dtype in ['images', 'labels']:
            os.makedirs(os.path.join(DEST_DIR, split, dtype), exist_ok=True)

    print(f"🚀 Memulai proses pemindahan data...")
    print(f"📂 Sumber: {SOURCE_DIR}")
    print(f"📂 Tujuan: {DEST_DIR}")

    # List untuk menyimpan semua pasangan (gambar, label) yang valid
    all_data_pairs = []

    # 1. SCANNING DATA
    print("\n🔍 Sedang memindai file...")
    source_path = Path(SOURCE_DIR)
    
    # Mencari gambar di semua subfolder kategori
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    
    for category in CATEGORIES:
        cat_path = source_path / category
        if not cat_path.exists():
            print(f"⚠️ Peringatan: Folder kategori '{category}' tidak ditemukan, dilewati.")
            continue
            
        # Cari semua file secara rekursif (termasuk di dalam subfolder)
        images = [f for f in cat_path.rglob('*') if f.suffix.lower() in image_extensions]
        
        print(f"   - {category}: ditemukan {len(images)} gambar.")
        
        for img_path in images:
            # Cari label pasangannya (.txt)
            # Logika: Cek di folder yg sama, atau folder labels di dekatnya
            lbl_path = img_path.with_suffix('.txt')
            
            # Jika tidak ada di sebelah gambar, coba trik replace 'images' -> 'labels'
            if not lbl_path.exists():
                str_path = str(img_path)
                if 'images' in str_path:
                    try_lbl = Path(str_path.replace('images', 'labels').replace(img_path.suffix, '.txt'))
                    if try_lbl.exists():
                        lbl_path = try_lbl
            
            # Jika label ketemu, masukkan ke daftar antrian
            if lbl_path.exists():
                all_data_pairs.append((img_path, lbl_path))

    total_files = len(all_data_pairs)
    if total_files == 0:
        print("❌ ERROR: Tidak ada pasangan Gambar+Label yang ditemukan!")
        return

    print(f"✅ Total pasangan valid (Gambar + Label): {total_files}")
    
    # 2. NGACAK & SPLIT
    random.shuffle(all_data_pairs)
    
    num_val = int(total_files * VAL_RATIO)
    num_train = total_files - num_val
    
    train_set = all_data_pairs[:num_train]
    val_set = all_data_pairs[num_train:]
    
    print(f"📊 Pembagian: {num_train} Train | {num_val} Validation")

    # 3. COPY FILE (Eksekusi)
    def copy_files(data_list, split_name):
        print(f"\n🚚 Menyalin ke folder {split_name.upper()}...")
        for img, lbl in tqdm(data_list):
            # Tentukan tujuan
            dest_img = os.path.join(DEST_DIR, split_name, 'images', img.name)
            dest_lbl = os.path.join(DEST_DIR, split_name, 'labels', lbl.name)
            
            # Copy file (gunakan copy2 agar metadata terjaga)
            shutil.copy2(img, dest_img)
            shutil.copy2(lbl, dest_lbl)

    copy_files(train_set, 'train')
    copy_files(val_set, 'val')

    print("\n🎉 SELESAI! Struktur dataset Anda sudah siap.")
    print(f"Silakan ZIP folder '{DEST_DIR}' dan upload ke Google Drive.")

if __name__ == "__main__":
    split_data()