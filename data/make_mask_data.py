'''
 ==================板块功能描述====================
           @Time     :2022/4/9   15:34
           @Author   : qiaofengsheng
           @File     :make_mask_data.py
           @Software :PyCharm
           @description:
 ================================================
 '''
import os

import cv2
import numpy as np
from PIL import Image, ImageDraw
import json

CLASS_NAMES = ['dog', 'cat']


def make_mask(image_dir, save_dir):
    data = os.listdir(image_dir)
    temp_data = []
    for i in data:
        if i.split('.')[1] == 'json':
            temp_data.append(i)
        else:
            continue
    for js in temp_data:
        json_data = json.load(open(os.path.join(image_dir, js), 'r'))
        shapes_ = json_data['shapes']
        mask = Image.new('P', Image.open(os.path.join(image_dir, js.replace('json', 'png'))).size)
        for shape_ in shapes_:
            label = shape_['label']
            points = shape_['points']
            points = tuple(tuple(i) for i in points)
            mask_draw = ImageDraw.Draw(mask)
            mask_draw.polygon(points, fill=CLASS_NAMES.index(label) + 1)
        mask.save(os.path.join(save_dir, js.replace('json', 'png')))


if __name__ == '__main__':
    make_mask('image', 'SegmentationClass')
    # img=Image.open('SegmentationClass/000019.png')
    # print(np.array(img).shape)
    # out=np.array(img).reshape(-1)
    # print(set(out.tolist()))
