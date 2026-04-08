import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 1. KOMPONEN BACKBONE (Efisiensi & Adaptasi)
# ==========================================

class SCINet_Block(nn.Module):
    """Self-Calibrated Illumination Network (Simplified SOTA)"""
    def __init__(self, channels):
        super().__init__()
        # Mengkalibrasi fitur secara spasial tanpa mengubah dimensi
        self.conv1 = nn.Conv2d(channels, channels // 2, kernel_size=1)
        self.conv2 = nn.Conv2d(channels // 2, channels, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        calibrated_weight = self.sigmoid(self.conv2(F.relu(self.conv1(x))))
        return x * calibrated_weight # Fitur dikalibrasi (diperkuat di area gelap)

class PConv(nn.Module):
    """Partial Convolution untuk C2f-faster (Sangat ringan)"""
    def __init__(self, dim, n_div=4):
        super().__init__()
        self.dim_conv3 = dim // n_div
        self.conv = nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)

    def forward(self, x):
        # Memproses hanya sebagian channel, sisanya diteruskan (hemat komputasi)
        x1, x2 = torch.split(x, [self.dim_conv3, x.size(1) - self.dim_conv3], dim=1)
        x1 = self.conv(x1)
        return torch.cat((x1, x2), dim=1)

class C2f_faster(nn.Module):
    """Modul C2f menggunakan PConv untuk efisiensi FPS mobile"""
    def __init__(self, c1, c2, n=1):
        super().__init__()
        self.cv1 = nn.Conv2d(c1, c2, 1, 1)
        self.cv2 = nn.Conv2d(c2 + (n * c2), c2, 1, 1)
        self.m = nn.ModuleList([PConv(c2) for _ in range(n)])

    def forward(self, x):
        y = [self.cv1(x)]
        for m in self.m:
            y.append(m(y[-1]))
        return self.cv2(torch.cat(y, dim=1))

# ==========================================
# 2. KOMPONEN NECK (Fusi & Pemisahan Fitur)
# ==========================================

class MCAM(nn.Module):
    """Multi-Dimensional Collaborative Attention Module"""
    def __init__(self, channels):
        super().__init__()
        # Channel Attention
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 8, 1),
            nn.ReLU(),
            nn.Conv2d(channels // 8, channels, 1),
            nn.Sigmoid()
        )
        # Spatial Attention
        self.sa = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x * self.ca(x) # Apply Channel Attention
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        spatial_map = torch.cat([max_pool, avg_pool], dim=1)
        return x * self.sa(spatial_map) # Apply Spatial Attention

class FDPN(nn.Module):
    """Feature Decoupling Pyramid Network"""
    def __init__(self, in_channels_list, out_channels):
        super().__init__()
        # Menerima fitur dari multi-skala backbone
        self.reduce_convs = nn.ModuleList([
            nn.Conv2d(c, out_channels, 1) for c in in_channels_list
        ])
        self.mcam = MCAM(out_channels) # Atensi di Neck

    def forward(self, features):
        # features: [P3, P4, P5] dari backbone
        decoupled_features = []
        for i, f in enumerate(features):
            reduced = self.reduce_convs[i](f)
            attended = self.mcam(reduced) # Filter fitur noise/cahaya buruk
            decoupled_features.append(attended)
        
        # Output: Fitur bersih untuk deteksi, dan fitur global untuk klasifikasi cahaya
        return decoupled_features 

# ==========================================
# 3. KOMPONEN HEAD (Eksekusi Multi-Task)
# ==========================================

class DyHead_Block(nn.Module):
    """Dynamic Head (Simplified) untuk Bounding Box Regression & Classification"""
    def __init__(self, channels):
        super().__init__()
        self.task_aware_attention = nn.Sequential(
            nn.Linear(channels, channels),
            nn.ReLU(),
            nn.Linear(channels, channels),
            nn.Sigmoid()
        )
        # Cabang regresi (Bbox) dan klasifikasi (Objek) - Memisahkan task
        self.bbox_conv = nn.Conv2d(channels, 4, 3, padding=1) # 4 koordinat
        self.cls_conv = nn.Conv2d(channels, 80, 3, padding=1) # Asumsi 80 kelas COCO

    def forward(self, x):
        # Implementasi sederhana: task-awareness membantu membedakan mana 
        # fitur lokalisasi dan mana fitur klasifikasi
        b, c, h, w = x.size()
        attn = self.task_aware_attention(x.mean((2,3))).view(b, c, 1, 1)
        x = x * attn
        
        bbox_pred = self.bbox_conv(x)
        cls_pred = self.cls_conv(x)
        return bbox_pred, cls_pred

class IllumHead(nn.Module):
    """Illumination Classification Head"""
    def __init__(self, in_channels, num_classes=3):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(in_channels, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.pool(x).flatten(1)
        return self.fc(x)

# ==========================================
# 4. ORKESTRASI: YOLO-MTL-ADAPTIVE
# ==========================================

class YOLO_MTL_Adaptive(nn.Module):
    def __init__(self):
        super().__init__()
        
        # --- A. BACKBONE ---
        self.stem = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1)
        self.scinet = SCINet_Block(32) # Adaptasi awal
        
        # Ekstraksi fitur (P3, P4, P5) menggunakan C2f-faster
        self.layer1 = C2f_faster(32, 64)   # Output P3
        self.layer2 = C2f_faster(64, 128)  # Output P4
        self.layer3 = C2f_faster(128, 256) # Output P5 (Global)

        # --- B. NECK ---
        # Mengambil P3, P4, P5 dan melakukan decoupling + MCAM
        self.fdpn = FDPN(in_channels_list=[64, 128, 256], out_channels=128)

        # --- C. HEADS ---
        # 1. Detect Head (Bawaan modifikasi DyHead) - diaplikasikan ke tiap skala
        self.dyheads = nn.ModuleList([DyHead_Block(128) for _ in range(3)])
        
        # 2. IllumHead - Hanya menggunakan P5 (skala terdalam/konteks global)
        self.illum_head = IllumHead(in_channels=128)

    def forward(self, x):
        # 1. Jalankan Backbone
        x = self.stem(x)
        x = self.scinet(x) # Bersihkan dari noise low-light/over-exposed
        
        p3 = self.layer1(x)
        p4 = self.layer2(p3)
        p5 = self.layer3(p4)
        
        # 2. Jalankan Neck (Pemisahan Fitur)
        decoupled_features = self.fdpn([p3, p4, p5])
        
        # 3. Jalankan Heads (Multi-Task)
        detect_outputs = []
        for i, feature in enumerate(decoupled_features):
            bbox, cls = self.dyheads[i](feature)
            detect_outputs.append((bbox, cls))
            
        # Prediksi kondisi cahaya dari fitur P5 (fitur terakhir di decoupled list)
        illum_logits = self.illum_head(decoupled_features[-1])
        
        return detect_outputs, illum_logits

# Uji Coba Model
if __name__ == "__main__":
    model = YOLO_MTL_Adaptive()
    dummy_image = torch.randn(1, 3, 640, 640) # Batch 1, RGB, 640x640
    
    detect_out, illum_out = model(dummy_image)
    
    print("Berhasil! Bentuk Output:")
    print(f"IllumHead Output (Low, Normal, Over): {illum_out.shape} -> (Batch, Classes)")
    print(f"DetectHead Skala 1 Bbox: {detect_out[0][0].shape}")
