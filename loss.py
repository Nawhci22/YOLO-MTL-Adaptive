import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops.boxes as box_ops
from scipy.optimize import linear_sum_assignment

# ==========================================
# 1. FUNGSI BANTUAN (UTILITIES)
# ==========================================
def xywh2xyxy(x):
    """Konversi format [cx, cy, w, h] ke [x1, y1, x2, y2]"""
    y = torch.zeros_like(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # x1
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # y1
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # x2
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # y2
    return y

def simple_bipartite_matching(pred_boxes_xyxy, pred_logits, gt_boxes_xyxy, gt_labels):
    """Fungsi Penjodohan (Hungarian Algorithm)"""
    if len(gt_boxes_xyxy) == 0:
        return torch.empty(0, dtype=torch.int64), torch.empty(0, dtype=torch.int64)

    with torch.no_grad():
        iou_matrix = box_ops.box_iou(pred_boxes_xyxy, gt_boxes_xyxy)
        cost_bbox = 1.0 - iou_matrix 
        
        pred_probs = pred_logits.softmax(dim=-1)
        cost_class = -pred_probs[:, gt_labels]
        
        cost_matrix = cost_bbox + cost_class
        cost_matrix_np = cost_matrix.cpu().numpy()
        
        pred_indices, gt_indices = linear_sum_assignment(cost_matrix_np)
        
    return torch.as_tensor(pred_indices, dtype=torch.int64), torch.as_tensor(gt_indices, dtype=torch.int64)

# ==========================================
# 2. KOMPONEN LOSS: MPDIoU
# ==========================================
class MPDIoULoss(nn.Module):
    def __init__(self, eps=1e-7):
        super().__init__()
        self.eps = eps

    def forward(self, pred_boxes, target_boxes):
        if len(pred_boxes) == 0:
            return torch.tensor(0.0, device=pred_boxes.device, requires_grad=True)
            
        b1_x1, b1_y1, b1_x2, b1_y2 = pred_boxes.unbind(dim=-1)
        b2_x1, b2_y1, b2_x2, b2_y2 = target_boxes.unbind(dim=-1)

        area_pred = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
        area_target = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)

        inter_x1 = torch.max(b1_x1, b2_x1)
        inter_y1 = torch.max(b1_y1, b2_y1)
        inter_x2 = torch.min(b1_x2, b2_x2)
        inter_y2 = torch.min(b1_y2, b2_y2)
        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)

        union_area = area_pred + area_target - inter_area + self.eps
        iou = inter_area / union_area

        cw_x1 = torch.min(b1_x1, b2_x1)
        cw_y1 = torch.min(b1_y1, b2_y1)
        cw_x2 = torch.max(b1_x2, b2_x2)
        cw_y2 = torch.max(b1_y2, b2_y2)
        
        c_diagonal_sq = ((cw_x2 - cw_x1) ** 2) + ((cw_y2 - cw_y1) ** 2) + self.eps
        d1_sq = (b1_x1 - b2_x1)**2 + (b1_y1 - b2_y1)**2
        d2_sq = (b1_x2 - b2_x2)**2 + (b1_y2 - b2_y2)**2

        mpdiou = iou - (d1_sq / c_diagonal_sq) - (d2_sq / c_diagonal_sq)
        return (1.0 - mpdiou).mean()

# ==========================================
# 3. KOMPONEN LOSS: DETEKSI (DyHead)
# ==========================================
class DetectionLoss(nn.Module):
    def __init__(self, num_classes=80):
        super().__init__()
        self.num_classes = num_classes
        self.mpdiou_criterion = MPDIoULoss()

    def forward(self, detect_outputs, targets_bbox):
        """
        detect_outputs: List dari tuple [(bbox_p3, cls_p3), (bbox_p4, cls_p4), ...]
        targets_bbox: List of tensors untuk setiap gambar [Num_Objects, 5] -> format [class, cx, cy, w, h]
        """
        device = detect_outputs[0][0].device
        batch_size = len(targets_bbox)
        
        # 1. FLATTEN OUTPUT DYHEAD
        # Model memprediksi dalam format matriks [Batch, Channel, H, W]. 
        # Kita harus meratakannya menjadi daftar ribuan kotak [Batch, Total_Anchors, Channel]
        all_pred_boxes = []
        all_pred_logits = []
        
        for bbox_out, cls_out in detect_outputs:
            b, c, h, w = bbox_out.size()
            # Reshape dan gabungkan semua anchor point
            bbox_flat = bbox_out.view(b, c, -1).permute(0, 2, 1) # [Batch, H*W, 4]
            cls_flat = cls_out.view(b, self.num_classes, -1).permute(0, 2, 1) # [Batch, H*W, 80]
            
            all_pred_boxes.append(bbox_flat)
            all_pred_logits.append(cls_flat)
            
        all_pred_boxes = torch.cat(all_pred_boxes, dim=1)   # Gabung P3, P4, P5
        all_pred_logits = torch.cat(all_pred_logits, dim=1) # Gabung P3, P4, P5
        
        total_det_loss = 0.0
        
        # 2. PROSES PER GAMBAR (Karena jumlah objek tiap gambar berbeda)
        for i in range(batch_size):
            pred_boxes = all_pred_boxes[i]       # [Ribuan_Anchor, 4] format cx,cy,w,h
            pred_logits = all_pred_logits[i]     # [Ribuan_Anchor, 80]
            target = targets_bbox[i].to(device)  # [Num_Obj, 5]
            
            # Ubah format prediksi ke xyxy untuk proses matching
            pred_boxes_xyxy = xywh2xyxy(pred_boxes)
            
            if len(target) == 0:
                # Jika tidak ada objek (Background Image), hukum semua prediksi sebagai "Bukan Objek"
                target_classes_bg = torch.zeros_like(pred_logits)
                loss_cls = F.binary_cross_entropy_with_logits(pred_logits, target_classes_bg)
                total_det_loss += loss_cls
                continue
                
            gt_classes = target[:, 0].long()
            gt_boxes_xywh = target[:, 1:5]
            gt_boxes_xyxy = xywh2xyxy(gt_boxes_xywh)
            
            # ---------------------------------------------------------
            # PEMANGGILAN FUNGSI BIPARTITE MATCHING DI SINI!
            # ---------------------------------------------------------
            matched_pred_idx, matched_gt_idx = simple_bipartite_matching(
                pred_boxes_xyxy, pred_logits, gt_boxes_xyxy, gt_classes
            )
            
            # 3. HITUNG LOSS KOTAK (HANYA UNTUK YANG BERJODOH)
            matched_pred_boxes = pred_boxes_xyxy[matched_pred_idx]
            matched_gt_boxes = gt_boxes_xyxy[matched_gt_idx]
            loss_bbox = self.mpdiou_criterion(matched_pred_boxes, matched_gt_boxes)
            
            # 4. HITUNG LOSS KLASIFIKASI (UNTUK SEMUA KOTAK)
            # Buat target kosong (semua anchor dianggap background/0)
            target_classes_full = torch.zeros_like(pred_logits)
            # Isi indeks yang 'berjodoh' dengan nilai 1 pada kolom kelas objek yang tepat
            target_classes_full[matched_pred_idx, gt_classes[matched_gt_idx]] = 1.0
            
            # Hitung Loss Klasifikasi menggunakan Binary Cross Entropy
            loss_cls = F.binary_cross_entropy_with_logits(pred_logits, target_classes_full)
            
            # Tambahkan ke total loss gambar ini
            total_det_loss += (loss_bbox + loss_cls)
            
        # Rata-rata loss untuk 1 Batch
        return total_det_loss / batch_size

# ==========================================
# 4. ORKESTRASI LOSS: MTL LOSS UTAMA
# ==========================================
class MTLLoss(nn.Module):
    """Combined Loss dengan Uncertainty Weighting"""
    def __init__(self, num_classes=80):
        super().__init__()
        # Parameter sigma (log variance) yang dipelajari model. Dimulai dari 0.
        self.log_vars = nn.Parameter(torch.zeros(2)) 
        
        # Panggil modul Detection Loss di atas
        self.det_criterion = DetectionLoss(num_classes=num_classes)

    def forward(self, detect_outputs, targets_bbox, pred_illum, target_illum):
        # 1. Hitung Loss Deteksi Asli (MPDIoU + Bipartite Matching)
        loss_det = self.det_criterion(detect_outputs, targets_bbox)
        
        # 2. Hitung Loss Klasifikasi Cahaya (Cross Entropy)
        loss_illum = F.cross_entropy(pred_illum, target_illum)
        
        # 3. Uncertainty Weighting (Penyeimbang Otomatis)
        precision_det = torch.exp(-self.log_vars[0])
        loss_1 = precision_det * loss_det + self.log_vars[0]
        
        precision_illum = torch.exp(-self.log_vars[1])
        loss_2 = precision_illum * loss_illum + self.log_vars[1]
        
        total_loss = loss_1 + loss_2
        return total_loss, loss_det, loss_illum