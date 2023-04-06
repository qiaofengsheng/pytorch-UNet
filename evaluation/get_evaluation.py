
import numpy as np
from sklearn.metrics import confusion_matrix
import cv2

def keep_image_size_open_label(path, size=(256, 256)):
    img = Image.open(path)
    temp = max(img.size)
    mask = Image.new('P', (temp, temp))
    mask.paste(img, (0, 0))
    mask = mask.resize(size)
    mask = np.array(mask)
    mask[mask!=255]=0
    mask[mask==255]=1
    mask = Image.fromarray(mask)
    return mask

def keep_image_size_open_predict(path, size=(256, 256)):
    img = Image.open(path)
    temp = max(img.size)
    mask = Image.new('P', (temp, temp))
    mask.paste(img, (0, 0))
    mask = mask.resize(size)
    mask = np.array(mask)
    mask = Image.fromarray(mask)
    return mask

def compute_iou(seg_pred, seg_gt, num_classes):
    ious = []
    for c in range(num_classes):
        pred_inds = seg_pred == c
        target_inds = seg_gt == c
        intersection = np.logical_and(pred_inds, target_inds).sum()
        union = np.logical_or(pred_inds, target_inds).sum()
        if union == 0:
            ious.append(float('nan'))
        else:
            ious.append(float(intersection) / float(union))
    return ious

def compute_miou(seg_preds, seg_gts, num_classes):
    ious = []
    for i in range(len(seg_preds)):
        ious.append(compute_iou(seg_preds[i], seg_gts[i], num_classes))
    ious = np.array(ious, dtype=np.float32)
    miou = np.nanmean(ious, axis=0)
    return miou

if __name__ == '__main__':
    from PIL import Image
    import os

    label_path = "data/val/SegmentationClass" # 标签的文件夹位置

    predict_path = "data/val/predict" # 预测结果的文件夹位置

    res_miou = []
    for pred_im in os.listdir(predict_path):
        label = keep_image_size_open_label(os.path.join(label_path,pred_im))
        pred = keep_image_size_open_predict(os.path.join(predict_path,pred_im))
        l, p = np.array(label).astype(int), np.array(pred).astype(int)
        print(set(l.reshape(-1).tolist()),set(p.reshape(-1).tolist()))
        miou = compute_miou(p,l,2)
        res_miou.append(miou)
    print(np.array(res_miou).mean(axis=0))


    