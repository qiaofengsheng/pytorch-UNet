'''
    混淆矩阵
    Recall、Precision、MIOU计算
'''
import numpy as np
from sklearn.metrics import confusion_matrix
import cv2

# 输入必须为灰度图
# labels为你的像素值的类别
from utils import keep_image_size_open


def get_miou_recall_precision(label_image, pred_image, labels):
    label = label_image.reshape(-1)
    pred = pred_image.reshape(-1)
    out = confusion_matrix(label, pred, labels=labels)
    # print(out)
    # TP = out[0][0]
    # FN = out[0][1] + out[0][2]
    # FP = out[1][0] + out[2][0]
    # TN = out[1][1] + out[1][2] + out[2][1] + out[2][2]
    # print(TP / (TP + FP + FN))
    r, l = out.shape
    iou_temp = 0
    recall = {}
    precision = {}
    for i in range(r):
        TP = out[i][i]
        temp = np.concatenate((out[0:i, :], out[i + 1:, :]), axis=0)
        sum_one = np.sum(temp, axis=0)
        FP = sum_one[i]
        temp2 = np.concatenate((out[:, 0:i], out[:, i + 1:]), axis=1)
        FN = np.sum(temp2, axis=1)[i]
        TN = temp2.reshape(-1).sum() - FN
        iou_temp += (TP / (TP + FP + FN))
        recall[i] = TP / (TP + FN)
        precision[i] = TP / (TP + FP)
    MIOU = iou_temp / len(labels)
    return MIOU, recall, precision


if __name__ == '__main__':
    from PIL import Image
    label = keep_image_size_open(r'D:\pythonSpace\teach_demo\pytorch-unet\data\SegmentationClass\000799.png')
    pred = Image.open(r'D:\pythonSpace\teach_demo\pytorch-unet\result\result.png')
    l, p = np.array(label).astype(int), np.array(pred).astype(int)
    print(get_miou_recall_precision(l, p, [0, 1, 2]))
