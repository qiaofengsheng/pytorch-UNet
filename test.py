import os

import cv2
import numpy as np
import torch

from net import *
from utils import *
from data import *
from torchvision.utils import save_image
from PIL import Image
net=UNet().cuda()

weights='params/unet.pth'
if os.path.exists(weights):
    net.load_state_dict(torch.load(weights))
    print('successfully')
else:
    print('no loading')

_input=input('please input JPEGImages path:')

img=keep_image_size_open_rgb(_input)
img_data=transform(img).cuda()
print(img_data.shape)
img_data=torch.unsqueeze(img_data,dim=0)
net.eval()
out=net(img_data)
out=torch.squeeze(out,dim=0)
out=(out*255).permute((1,2,0)).cpu().detach().numpy()
print(set((out*255).reshape(-1).tolist()))

cv2.imwrite('result/result.png',out)


