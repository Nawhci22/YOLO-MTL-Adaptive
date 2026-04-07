import os
import glob
from PIL import Image
from tqdm import tqdm

# Daftar folder root dataset Anda
FOLDERS = [
    "dataset_coco/normal", 
    "dataset_coco/lowlight", 
    "dataset_coco/overexposed"
]

def get_label_path(image_path):
    """
    Mencari lokasi file label (.txt) berdasarkan lokasi gambar.
    Mendukung 2 struktur folder:
    1. Mixed: Gambar dan Label di folder yang sama.
    2. Split: Gambar di folder 'images', Label di folder 'labels'.
    """
    # Ganti ekstensi gambar jadi .txt
    base_txt_path = os.path.splitext(image_path)[0] + ".txt"
    
    # KEMUNGKINAN 1: Label ada di folder yang sama
    if os.path.exists(base_txt_path):
        return base_txt_path
    
    # KEMUNGKINAN 2: Label ada di folder 'labels' sejajar dengan 'images'
    # Contoh: dataset/normal/images/foto.jpg -> dataset/normal/labels/foto.txt
    
    # Cek variasi path separator (Windows '\' atau Linux '/')
    if "/images/" in base_txt_path:
        parallel_txt_path = base_txt_path.replace("/images/", "/labels/")
    elif "\\images\\" in base_txt_path:
        parallel_txt_path = base_txt_path.replace("\\images\\", "\\labels\\")
    else:
        return None

    if os.path.exists(parallel_txt_path):
        return parallel_txt_path
    
    return None

def check_and_clean():
    print("Memulai pemindaian file corrupt & label yatim piatu...")
    count_img_deleted = 0
    count_lbl_deleted = 0
    
    for folder in FOLDERS:
        # Gunakan recursive=True untuk masuk ke subfolder 'images'
        # Mencari semua file di dalam folder tersebut
        files = glob.glob(os.path.join(folder, "**", "*.*"), recursive=True)
        
        # Filter hanya file gambar
        img_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        print(f"Checking folder '{folder}' ({len(img_files)} images)...")
        
        for file_path in tqdm(img_files):
            is_corrupt = False
            
            try:
                # Cek integritas gambar
                img = Image.open(file_path)
                img.verify() 
                
                # Double check dengan load penuh
                img = Image.open(file_path) 
                img.load()
                
            except (IOError, SyntaxError, OSError) as e:
                is_corrupt = True
                print(f"\n[CORRUPT] Ditemukan: {file_path}")

            if is_corrupt:
                # 1. Hapus Gambar
                try:
                    os.remove(file_path)
                    count_img_deleted += 1
                    print(f" - Image Deleted: OK")
                except Exception as e:
                    print(f" - Image Delete GAGAL: {e}")

                # 2. Cari dan Hapus Label
                label_path = get_label_path(file_path)
                if label_path:
                    try:
                        os.remove(label_path)
                        count_lbl_deleted += 1
                        print(f" - Label Deleted: {label_path}")
                    except Exception as e:
                        print(f" - Label Delete GAGAL: {e}")
                else:
                    print(f" - Label tidak ditemukan (Aman).")

    print("\n" + "="*40)
    print("PEMBERSIHAN SELESAI")
    print("="*40)
    print(f"Gambar Rusak Dihapus : {count_img_deleted}")
    print(f"Label Terkait Dihapus: {count_lbl_deleted}")
    print("="*40)

if __name__ == "__main__":
    check_and_clean()