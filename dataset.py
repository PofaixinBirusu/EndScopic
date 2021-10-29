import torch
from torch.utils import data
from util import extract_big_pic_roi, pil2cv, img_pad, cv2pil
import os
import cv2
from PIL import Image
from torchvision import transforms
import numpy as np


class Imgs(data.Dataset):
    def __init__(self, imgs, label, transform=None):
        # 里面存一些PIL图
        self.imgs = imgs
        self.transform = transform
        self.label = label

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img = self.imgs[index]
        if self.transform is not None:
            img = self.transform(img)
        return img


class Endoscope(data.Dataset):
    def __init__(self, roots, labels):
        super(Endoscope, self).__init__()
        self.imgs, self.labels = [], []
        for i, root in enumerate(roots):
            path = root
            for _, y, filelist in os.walk(root):
                # print(y[0], filelist)
                names = []
                if len(y) > 0:
                    if len(y) == 1:
                        path += "/"+y[0]
                    else:
                        names += [path+"/"+name for name in y]
                if len(names) > 0:
                    # print(names)
                    for name in names:
                        for _, _, x in os.walk(name+"/"+"图像/"):
                            self.imgs.append([name+"/"+"图像/"+img for img in x if img.startswith("IMG") and img.endswith(".png")])
                            self.labels.append(labels[i])
                            break
        # print(self.imgs)
        # 随机取100个病人吧，电脑跑不动那么多
        # idx = np.random.permutation(len(self.imgs))[:100]
        # np.save("idx100.npy", idx)
        # idx = np.load("idx100.npy")
        # self.imgs = [self.imgs[idx[i]] for i in range(idx.shape[0])]
        # self.labels = [self.labels[idx[i]] for i in range(idx.shape[0])]
        cnt = [0, 0]
        for i in range(len(self.labels)):
            cnt[self.labels[i]] += 1
        print(cnt)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        imgs_of_patient = self.imgs[index]
        # imgs_of_patient = imgs_of_patient[(len(imgs_of_patient)-32)//2:(len(imgs_of_patient)-32)//2+32]
        # print(len(imgs_of_patient))
        imgs = []
        for path in imgs_of_patient:
            img = Image.open(path)
            img = pil2cv(img)
            h, w = img.shape[0], img.shape[1]
            if h > 1000 or w > 1000:
                img = extract_big_pic_roi(img)
            if img is not None:
                imgs.append(cv2pil(img_pad(img, 640, 640, 0)))
        # 这里取16,32,48都没什么问题，说的是文件夹里最多有几张图。每个文件夹中的图片数量可以不相等，如果多于这个限制，就最多取这个数，取多了显存扛不住
        get_num = 16
        start_idx = (len(imgs)-get_num)//2
        imgs = imgs[start_idx:start_idx+get_num]
        ds = Imgs(imgs, self.labels[index], transform=transforms.Compose([
            transforms.ToTensor()
        ]))
        # print(imgs_of_patient[0])
        return ds


if __name__ == '__main__':
    # endscope_positive_dataset = Endoscope(roots=["E:/内镜图片资料/2020pcr阳性"])
    endscope_dataset = Endoscope(roots=["E:/内镜图片资料/2020pcr阳性", "E:/内镜图片资料/2021PCR阳性", "E:/内镜图片资料/2020pcr阴性"], labels=[1, 1, 0])
    # endscope_negative_dataset = Endoscope(roots=["E:/内镜图片资料/2020pcr阴性"])
    # print(len(endscope_positive_dataset))
    # print(len(endscope_negative_dataset))
    # for i in range(len(endscope_positive_dataset)):
    #     img, path = endscope_positive_dataset[i]
    #     w, h = img.shape[1], img.shape[0]
    #     if w < 512 and h < 512:
    #         print(img.shape)
    #         print("error")
    #     print("\r%d / %d" % (i+1, len(endscope_positive_dataset)), end="")
    #     print(img)
    for i in range(0, len(endscope_dataset)):
        ds = endscope_dataset[i]
        # for j in range(len(ds)):
        #     img = pil2cv(ds[j])
        #     cv2.imshow("", img)
        #     cv2.waitKey(0)
        print(ds.label)