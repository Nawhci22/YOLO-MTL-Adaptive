from ultralytics import YOLO

def main():
    # 1. Inisialisasi Model
    # Menggunakan YOLOv8 nano (paling ringan dan cepat)
    model = YOLO('yolov8n.pt') 

    print("Memulai proses training dengan custom hyperparameters...")

    # 2. Menjalankan Training
    results = model.train(
        # === KONFIGURASI DASAR ===
        data='dataset.yaml',     # Path ke file konfigurasi dataset
        epochs=100,              # Jumlah total iterasi training
        imgsz=640,               # Ukuran resolusi gambar untuk training
        batch=16,                # Jumlah gambar per batch (turunkan ke 8 jika VRAM GPU kurang)
        device=0,                # Gunakan 0 untuk GPU, atau 'cpu' jika tidak ada GPU
        project='YOLO_Training', # Nama folder utama untuk menyimpan hasil
        name='yolov8n_custom',   # Nama sub-folder untuk eksperimen ini
        
        # === CUSTOM HYPERPARAMETERS (TRAINING & OPTIMIZATION) ===
        optimizer='AdamW',       # Optimizer (bisa 'SGD', 'Adam', 'AdamW')
        lr0=0.001,               # Initial learning rate (standar AdamW)
        lrf=0.01,                # Final learning rate fraction (lr0 * lrf)
        momentum=0.937,          # Momentum untuk pergerakan optimizer
        weight_decay=0.0005,     # Weight decay untuk mencegah overfitting
        warmup_epochs=3.0,       # Epoch pemanasan agar loss tidak langsung meledak
        warmup_momentum=0.8,     # Momentum saat fase warmup
        box=7.5,                 # Bobot loss untuk akurasi kotak (bounding box)
        cls=0.5,                 # Bobot loss untuk akurasi klasifikasi kelas
        dfl=1.5,                 # Bobot loss untuk Distribution Focal Loss (deteksi halus)
        
        # === MEMATIKAN AUGMENTASI BAWAAN YOLO ===
        # Karena data Anda SUDAH diaugmentasi secara eksternal,
        # kita harus mematikan augmentasi otomatis dari YOLOv8
        # agar gambar tidak menjadi terlalu rusak.
        hsv_h=0.0,               # Variasi Hue (warna)
        hsv_s=0.0,               # Variasi Saturation (saturasi)
        hsv_v=0.0,               # Variasi Value (kecerahan)
        degrees=0.0,             # Rotasi gambar
        translate=0.0,           # Pergeseran posisi
        scale=0.0,               # Perubahan ukuran zoom
        shear=0.0,               # Distorsi miring
        perspective=0.0,         # Distorsi perspektif
        flipud=0.0,              # Flip atas-bawah
        fliplr=0.0,              # Flip kiri-kanan
        mosaic=0.0,              # Menggabungkan 4 gambar menjadi 1 (Matikan!)
        mixup=0.0,               # Mencampur 2 gambar transparan (Matikan!)
        copy_paste=0.0           # Menempel objek dari gambar lain (Matikan!)
    )

    print("Training Selesai! Hasil disimpan di folder 'YOLOv8n_Training/yolov8n_custom'")

if __name__ == '__main__':
    # Di Windows, multiprocessing (yang digunakan dataloader YOLO) 
    # harus dibungkus dalam blok if __name__ == '__main__':
    main()