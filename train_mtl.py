import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision

# Import modul custom Anda
from YOLO_MTL_Adaptive import YOLO_MTL_Adaptive
from loss import MTLLoss, xywh2xyxy
from my_dataset import CustomMTLDataset

# Import metrik
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchmetrics.classification import MulticlassAccuracy

# ==========================================
# FUNGSI PENGGABUNG BATCH (COLLATE FUNCTION)
# ==========================================
def custom_collate_fn(batch):
    """
    Fungsi khusus agar DataLoader tidak memaksa menumpuk (stack) 
    target bounding box yang ukurannya berbeda-beda per gambar.
    """
    images = []
    targets_bbox = []
    targets_illum = []

    for img, bbox, illum in batch:
        images.append(img)
        targets_bbox.append(bbox)
        targets_illum.append(illum)

    # 1. Tumpuk gambar menjadi satu tensor [Batch, Channel, Height, Width]
    images = torch.stack(images, dim=0)
    
    # 2. BIARKAN targets_bbox TETAP SEBAGAI LIST (Jangan di-stack!)
    # targets_bbox = [tensor(2, 5), tensor(3, 5), ...]
    
    # 3. Tumpuk label iluminasi menjadi tensor 1D [Batch]
    targets_illum = torch.tensor(targets_illum) 
    
    return images, targets_bbox, targets_illum

# ==========================================
# 1. FUNGSI BANTUAN (NMS Asli & Konversi Box)
# ==========================================
def xywh2xyxy_val(x):
    """Konversi format [cx, cy, w, h] ke [x1, y1, x2, y2]"""
    y = torch.zeros_like(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # x1
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # y1
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # x2
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # y2
    return y

def apply_nms(all_pred_boxes, all_pred_logits, conf_thres=0.25, iou_thres=0.45):
    """Fungsi NMS Asli (Pengganti Dummy)"""
    results = []
    batch_size = all_pred_boxes.size(0)
    
    for i in range(batch_size):
        boxes = all_pred_boxes[i]
        
        # Konversi logits ke probabilitas
        class_probs = all_pred_logits[i].softmax(dim=-1)
        
        # Cari probabilitas kelas tertinggi
        class_conf, class_pred = torch.max(class_probs, 1, keepdim=True)
        
        # Buang tebakan yang skornya di bawah threshold (0.25)
        conf_mask = (class_conf.squeeze() > conf_thres)
        
        boxes = boxes[conf_mask]
        class_conf = class_conf[conf_mask]
        class_pred = class_pred[conf_mask]
        
        if boxes.shape[0] == 0:
            results.append({
                "boxes": torch.empty((0,4), device=boxes.device), 
                "scores": torch.empty(0, device=boxes.device), 
                "labels": torch.empty(0, dtype=torch.long, device=boxes.device)
            })
            continue

        # Konversi kotak ke xyxy untuk algoritma NMS PyTorch
        boxes_xyxy = xywh2xyxy_val(boxes)
        nms_indices = torchvision.ops.nms(boxes_xyxy, class_conf.squeeze(), iou_thres)
        
        # Simpan kotak final yang selamat dari NMS
        results.append({
            "boxes": boxes_xyxy[nms_indices],
            "scores": class_conf[nms_indices].squeeze(1),
            "labels": class_pred[nms_indices].squeeze(1)
        })
    return results

# ==========================================
# 2. FUNGSI TRAINING (Versi Asli - Tanpa Dummy)
# ==========================================
def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch):
    model.train()
    total_loss_epoch = 0.0
    
    for batch_idx, (images, targets_bbox, targets_illum) in enumerate(dataloader):
        images, targets_illum = images.to(device), targets_illum.to(device)
        
        optimizer.zero_grad()
        
        # 1. Forward pass (Prediksi)
        # detect_preds berisi prediksi Bounding Box, illum_preds berisi prediksi pencahayaan
        detect_preds, illum_preds = model(images)
        
        # 2. Hitung Loss Asli (Menggunakan MPDIoU, Bipartite Matching, & Uncertainty)
        # PERHATIKAN BARIS INI: Harus menggunakan 4 parameter yang sesungguhnya!
        loss, l_det, l_illum = criterion(detect_preds, targets_bbox, illum_preds, targets_illum)
        
        # 3. Backward pass & Update bobot
        loss.backward()
        optimizer.step()
        
        total_loss_epoch += loss.item()
        
        # Print progress per beberapa batch agar Anda tahu model tidak hang
        if batch_idx % 10 == 0:
            print(f"   Batch {batch_idx}/{len(dataloader)} | Loss: {loss.item():.4f} (Det: {l_det.item():.4f}, Illum: {l_illum.item():.4f})")
        
    print(f"Epoch {epoch} [TRAIN Selesai] | Avg Total Loss: {total_loss_epoch/len(dataloader):.4f}")
    print(f"Bobot Otomatis -> W_Det: {torch.exp(-criterion.log_vars[0]).item():.4f}, W_Illum: {torch.exp(-criterion.log_vars[1]).item():.4f}")
    
# ==========================================
# 3. FUNGSI VALIDASI (Versi Asli - Tanpa Dummy)
# ==========================================
def validate(model, dataloader, device, num_classes=80): # Ubah 80 jika dataset Anda berbeda
    model.eval()
    map_metric = MeanAveragePrecision(box_format='xyxy', iou_type='bbox')
    illum_acc_metric = MulticlassAccuracy(num_classes=3).to(device)
    
    with torch.no_grad():
        for images, targets_bbox, targets_illum in dataloader:
            images, targets_illum = images.to(device), targets_illum.to(device)
            
            # Forward pass
            detect_outputs, illum_logits = model(images)
            
            # --- EVALUASI DETEKSI ---
            # A. Ratakan output DyHead (seperti di loss.py)
            all_pred_boxes = []
            all_pred_logits = []
            for bbox_out, cls_out in detect_outputs:
                b, c, h, w = bbox_out.size()
                bbox_flat = bbox_out.view(b, c, -1).permute(0, 2, 1) 
                cls_flat = cls_out.view(b, num_classes, -1).permute(0, 2, 1) 
                all_pred_boxes.append(bbox_flat)
                all_pred_logits.append(cls_flat)
                
            all_pred_boxes = torch.cat(all_pred_boxes, dim=1)
            all_pred_logits = torch.cat(all_pred_logits, dim=1)
            
            # B. Terapkan NMS Asli ke Prediksi
            preds_det = apply_nms(all_pred_boxes, all_pred_logits, conf_thres=0.25, iou_thres=0.45)
            
            # C. Format Target Ground Truth Asli (Bukan Simulasi Lagi)
            targets_det = []
            for i in range(len(targets_bbox)):
                target = targets_bbox[i].to(device) # [Num_Obj, 5] -> [class, cx, cy, w, h]
                if len(target) == 0:
                    targets_det.append({
                        "boxes": torch.empty((0,4), device=device),
                        "labels": torch.empty(0, dtype=torch.long, device=device)
                    })
                else:
                    gt_classes = target[:, 0].long()
                    gt_boxes_xywh = target[:, 1:5]
                    gt_boxes_xyxy = xywh2xyxy_val(gt_boxes_xywh) # Wajib xyxy untuk mAP
                    targets_det.append({
                        "boxes": gt_boxes_xyxy,
                        "labels": gt_classes
                    })
            
            # Masukkan ke Kalkulator mAP
            map_metric.update(preds_det, targets_det)
            
            # --- EVALUASI ILUMINASI ---
            preds_illum = torch.argmax(illum_logits, dim=1)
            illum_acc_metric.update(preds_illum, targets_illum)

    # Hitung metrik akhir
    final_map = map_metric.compute()
    final_acc = illum_acc_metric.compute()
    
    print(f"Epoch [VAL]   | mAP@0.5:0.95: {final_map['map'].item():.4f} | mAP@0.5: {final_map['map_50'].item():.4f} | Illum Acc: {final_acc.item()*100:.2f}%")
    
    map_metric.reset()
    illum_acc_metric.reset()
    
    return final_map['map'].item(), final_acc.item()

# ==========================================
# 4. ORKESTRASI UTAMA (Main Loop)
# ==========================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Menjalankan training di: {device}")
    
    # 1. Inisialisasi Model & Loss
    model = YOLO_MTL_Adaptive().to(device)
    criterion = MTLLoss().to(device)
    
    # Optimizer mencakup parameter model DAN parameter uncertainty dari loss
    optimizer = optim.AdamW(list(model.parameters()) + list(criterion.parameters()), lr=1e-3)
    
    # 2. Setup Dataloaders (Ganti path dengan folder dataset Anda)
    train_dataset = CustomMTLDataset(img_dir="./datasets/train/images", label_dir="./datasets/train/labels")
    val_dataset = CustomMTLDataset(img_dir="./datasets/val/images", label_dir="./datasets/val/labels")
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=custom_collate_fn)
    
    # 3. Training Loop
    num_epochs = 1
    best_map = 0.0 # Lacak mAP terbaik
    
    print("\n=== Memulai Training YOLO-MTL-Adaptive ===")
    for epoch in range(1, num_epochs + 1):
        print(f"\n--- Epoch {epoch}/{num_epochs} ---")
        
        # Langkah A: Belajar (Train)
        train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        
        # Langkah B: Ujian (Validasi)
        current_map, current_acc = validate(model, val_loader, device)
        
        # Langkah C: Simpan Model Terbaik (Berdasarkan mAP)
        if current_map > best_map:
            best_map = current_map
            print(f"*** mAP Meningkat! Menyimpan best_mtl_model.pt ***")
            
            # Simpan state dictionary model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_map': best_map,
            }, "best_mtl_model.pt")

    print("\n=== Training Selesai ===")
    print(f"Model terbaik tersimpan dengan mAP: {best_map:.4f}")

if __name__ == "__main__":
    main()