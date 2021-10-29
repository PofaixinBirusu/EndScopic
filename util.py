import cv2
import numpy as np
from PIL import Image
from copy import deepcopy
from torch import nn
from torch.nn import functional as F


def extract_big_pic_roi(big_img, w_thresh=700):
    gray = cv2.cvtColor(big_img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    roi = None
    for c in contours[1:]:
        x, y, w, h = cv2.boundingRect(c)
        # 过滤没用的地方
        if w < w_thresh or h < w_thresh:
            continue
        roi = big_img[y:y+h, x:x+w, :]
    if roi is None:
        # print("don't find roi, please check !!!")
        pass
    return roi


def pil2cv(pil_img):
    img = cv2.cvtColor(np.asarray(pil_img), cv2.COLOR_RGB2BGR)
    return img


def cv2pil(cv_img):
    return Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))


def img_pad(img, tw, th, pad=128):
    width, height = img.shape[1], img.shape[0]
    new_w, new_h = int(width*min(tw/width, th/height)), int(height*min(tw/width, th/height))
    canvas = np.full((th, tw, 3), pad, dtype=np.uint8)
    canvas[(th-new_h)//2:(th-new_h)//2+new_h, (tw-new_w)//2:(tw-new_w)//2+new_w, :] = cv2.resize(
        img, (new_w, new_h), interpolation=cv2.INTER_CUBIC
    )
    return canvas


class SpatialGatingUnit(nn.Module):
    def __init__(self, d_ffn, seq_len):
        super().__init__()
        self.norm = nn.LayerNorm(d_ffn)
        self.spatial_proj = nn.Conv1d(seq_len, seq_len, kernel_size=1)
        nn.init.constant_(self.spatial_proj.bias, 1.0)  # 偏差

    def forward(self, x):
        u, v = x.chunk(2, dim=-1)
        v = self.norm(v)
        v = self.spatial_proj(v)
        out = u * v
        return out


class gMLPBlock(nn.Module):
    def __init__(self, d_model, d_ffn, seq_len):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.channel_proj1 = nn.Linear(d_model, d_ffn * 2)  # (256, d_ffn * 2=1024)  [-1,256,1024]
        self.sgu = SpatialGatingUnit(d_ffn, seq_len)  #
        self.channel_proj2 = nn.Linear(d_ffn, d_model)

    def forward(self, x):
        residual = x
        x = self.norm(x)  # [-1,256,256]
        x = F.gelu(self.channel_proj1(x))  # GELU激活函数 [-1,256,256]
        x = self.sgu(x)  # [-1,256,256]
        x = self.channel_proj2(x)
        out = x + residual
        return out


class gMLP(nn.Module):
    def __init__(self, d_model=256, d_ffn=512, seq_len=256, num_layers=6):
        super().__init__()
        self.model = nn.Sequential(
            *[gMLPBlock(d_model, d_ffn, seq_len) for _ in range(num_layers)]
        )
        # [gMLPBlock(d_model=256, d_ffn=512, seq_len=256) for _ in range(num_layers)]

    def forward(self, x):
        return self.model(x)


if __name__ == '__main__':
    img = cv2.imread("1.png")
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # ret, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    # cv2.imshow("", thresh)
    # cv2.waitKey(0)
    # contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # roi = None
    # img_draw = deepcopy(img)
    # for c in contours[1:]:
    #     x, y, w, h = cv2.boundingRect(c)
    #     # 过滤没用的地方
    #     if w < 100 or h < 100:
    #         continue
    #     roi = img[y:y+h, x:x+w, :]
    #     img_draw = cv2.rectangle(img_draw, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # cv2.imshow("", img_draw)
    # cv2.waitKey(0)
    roi = extract_big_pic_roi(img)
    roi = img_pad(roi, 640, 640, 0)
    cv2.imshow("", roi)
    cv2.waitKey(0)