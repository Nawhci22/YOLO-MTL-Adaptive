import os
import requests
import zipfile
import shutil
import cv2
import numpy as np
from pycocotools.coco import COCO
from tqdm import tqdm
from pathlib import Path

# ================= KONFIGURASI =================
# 1. Kelas yang diinginkan
#    Pastikan nama kelas persis dengan nama di COCO (bhs Inggris, lowercase)
TARGET_CLASSES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
        'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
        'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
        'hair drier', 'toothbrush']

# 2. Tipe Dataset ('train' atau 'val')
#    'val' = file anotasi kecil (~20MB), total 5k gambar.
#    'train' = file anotasi besar (~250MB), total 118k gambar.
DATA_TYPE = "val"  
YEAR = "2017"

# 3. Jumlah maksimum gambar yang ingin didownload
MAX_SAMPLES = 5000

# 4. Filter Kualitas
MIN_BRIGHTNESS = 75
MAX_BRIGHTNESS = 180
MIN_CONTRAST = 30

# 5. Path Output
BASE_DIR = "dataset_coco/normal"
IMG_DIR = os.path.join(BASE_DIR, "images")
LBL_DIR = os.path.join(BASE_DIR, "labels")
# ===============================================

def download_file(url, save_path):
    """Download file dengan progress bar"""
    if os.path.exists(save_path):
        print(f"File {save_path} sudah ada, melewati download.")
        return

    print(f"Mendownload {os.path.basename(save_path)}...")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(save_path, 'wb') as file, tqdm(
        desc=os.path.basename(save_path),
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)

def is_good_image(img_arr):
    """Cek kualitas gambar dari array numpy"""
    if img_arr is None: return False
    if len(img_arr.shape) < 3: return False

    gray = cv2.cvtColor(img_arr, cv2.COLOR_BGR2GRAY)
    mean_val = np.mean(gray)
    std_val = np.std(gray)

    if mean_val < MIN_BRIGHTNESS: return False
    if mean_val > MAX_BRIGHTNESS: return False
    if std_val < MIN_CONTRAST: return False
    return True

def convert_coco_bbox_to_yolo(bbox, img_w, img_h):
    """
    COCO: [x_min, y_min, width, height]
    YOLO: [x_center, y_center, width, height] (Normalized)
    """
    x_min, y_min, w, h = bbox
    
    x_center = (x_min + w / 2) / img_w
    y_center = (y_min + h / 2) / img_h
    w_norm = w / img_w
    h_norm = h / img_h
    
    return [x_center, y_center, w_norm, h_norm]

def main():
    os.makedirs(IMG_DIR, exist_ok=True)
    os.makedirs(LBL_DIR, exist_ok=True)

    # 1. Download File Anotasi COCO (JSON)
    #    Kita hanya butuh JSON-nya saja untuk tahu URL gambar.
    ann_zip_name = f"annotations_trainval{YEAR}.zip"
    ann_url = f"http://images.cocodataset.org/annotations/{ann_zip_name}"
    ann_zip_path = os.path.join(BASE_DIR, ann_zip_name)
    
    download_file(ann_url, ann_zip_path)

    # Ekstrak JSON
    ann_json_file = f"instances_{DATA_TYPE}{YEAR}.json"
    ann_json_path = os.path.join(BASE_DIR, "annotations", ann_json_file)
    
    if not os.path.exists(ann_json_path):
        print("Mengekstrak anotasi...")
        with zipfile.ZipFile(ann_zip_path, 'r') as zip_ref:
            zip_ref.extractall(BASE_DIR)

    # 2. Inisialisasi COCO API
    print("Memuat data anotasi (ini mungkin memakan waktu)...")
    coco = COCO(ann_json_path)

    # 3. Dapatkan ID Gambar yang mengandung target kelas
    #    Kita cari gambar yang memuat SALAH SATU dari target class
    print(f"Mencari gambar dengan kelas: {TARGET_CLASSES}")
    
    # Ambil ID kategori COCO untuk target kelas kita
    # Contoh: 'person' di COCO id-nya 1, 'car' id-nya 3
    catIds = coco.getCatIds(catNms=TARGET_CLASSES)
    
    # Mapping ID COCO ke ID YOLO (0, 1, 2...)
    # Agar di file txt nanti urut: person=0, car=1, dst.
    coco_id_to_yolo_idx = {coco_id: idx for idx, coco_id in enumerate(catIds)}
    
    # Ambil semua Image ID yang mengandung kategori tersebut
    imgIds = []
    for catId in catIds:
        imgIds.extend(coco.getImgIds(catIds=[catId]))
    
    # Hapus duplikat (satu gambar bisa punya person DAN car)
    imgIds = list(set(imgIds))
    
    # Batasi jumlah sampel
    if MAX_SAMPLES and len(imgIds) > MAX_SAMPLES:
        np.random.shuffle(imgIds)
        imgIds = imgIds[:MAX_SAMPLES]
        
    print(f"Total gambar target ditemukan: {len(imgIds)}")

    # 4. Loop Download & Processing
    count_saved = 0
    count_rejected = 0

    pbar = tqdm(imgIds, desc="Processing Images")
    for img_id in pbar:
        # Load metadata gambar
        img_info = coco.loadImgs(img_id)[0]
        file_name = img_info['file_name']
        img_url = img_info['coco_url']
        
        save_img_path = os.path.join(IMG_DIR, file_name)
        save_lbl_path = os.path.join(LBL_DIR, os.path.splitext(file_name)[0] + ".txt")

        # Skip jika sudah ada
        if os.path.exists(save_img_path) and os.path.exists(save_lbl_path):
            count_saved += 1
            continue

        try:
            # A. Download Gambar ke Memory (tanpa simpan dulu)
            #    Menggunakan requests.get().content lalu decode cv2
            resp = requests.get(img_url, timeout=10)
            if resp.status_code != 200:
                continue
                
            img_array = np.asarray(bytearray(resp.content), dtype=np.uint8)
            img = cv2.imdecode(img_array, -1) # -1 = keep alpha/color

            # B. Cek Kualitas
            if not is_good_image(img):
                count_rejected += 1
                continue

            # C. Ambil Anotasi untuk gambar ini
            #    Hanya ambil anotasi yang termasuk dalam TARGET_CLASSES kita
            annIds = coco.getAnnIds(imgIds=img_info['id'], catIds=catIds, iscrowd=None)
            anns = coco.loadAnns(annIds)

            yolo_labels = []
            h, w, _ = img.shape # Tinggi, Lebar gambar aktual

            for ann in anns:
                # Dapatkan YOLO class ID (0, 1, 2...)
                cls_idx = coco_id_to_yolo_idx[ann['category_id']]
                
                # Konversi Bbox
                bbox = ann['bbox'] # [x, y, w, h]
                yolo_bbox = convert_coco_bbox_to_yolo(bbox, w, h)
                
                # Format baris: class_id x_center y_center w h
                label_line = f"{cls_idx} {' '.join(map(str, yolo_bbox))}"
                yolo_labels.append(label_line)

            # D. Simpan jika ada label valid
            if yolo_labels:
                # Simpan Gambar
                cv2.imwrite(save_img_path, img)
                
                # Simpan Label
                with open(save_lbl_path, 'w') as f:
                    f.write('\n'.join(yolo_labels))
                
                count_saved += 1
            else:
                count_rejected += 1 # Gambar bagus, tapi label target tidak valid (jarang terjadi)

        except Exception as e:
            # print(f"Error processing {file_name}: {e}")
            continue
            
    # 5. Buat classes.txt
    classes_path = os.path.join(BASE_DIR, "classes.txt")
    with open(classes_path, 'w') as f:
        for cls_name in TARGET_CLASSES:
            f.write(cls_name + "\n")

    print("\n" + "="*40)
    print("SELESAI")
    print(f"Disimpan : {count_saved}")
    print(f"Ditolak  : {count_rejected}")
    print(f"Output   : {BASE_DIR}")
    print("="*40)

if __name__ == "__main__":
    main()