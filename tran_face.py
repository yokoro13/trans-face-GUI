import cv2
from util import pil_to_tensor, cv2_to_pil, tensor_to_pil, pil_to_cv2
import numpy as np
import torch
from model import Generator, Classification
from PIL import Image
from deeolab_v3_plus.segmentation import Segmentation


class TransFace(object):
    def __init__(self):
        self.cascade_path = "haarcascade_frontalface_alt.xml"
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.c_dim = 5
        self.deeplab = Segmentation()
        self.G = Generator(c_dim=self.c_dim)
        self.G.to(self.device)
        #self.C = Classification(c_dim=self.c_dim)
        #self.C.to(self.device)
        self.restore_model()

    def classification(self, image):
        image = pil_to_tensor(image)
        with torch.no_grad():
            out_cls = self.C(image)
            out_cls = out_cls.view(out_cls.size(1))

            out_cls[out_cls < 0.5] = 0
            out_cls[out_cls >= 0.5] = 1
            return out_cls

    def merge_img(self, img_org, img_trans, img_mask):
        img_mask = np.where(img_mask == 0, img_org, img_trans)
        return cv2_to_pil(img_mask)

    def trans(self, img, c_trg=None):
        if c_trg is None:
            c_trg = [0, 1, 0, 1, 1]
        c_trg = np.asarray(c_trg, dtype=np.float32)
        c_trg = torch.from_numpy(c_trg)
        c_trg = c_trg.view(1, c_trg.size(0)).to(self.device)
        img = img.convert("RGB")
        img_gray = np.asarray(img)
        img_gray = cv2.cvtColor(img_gray, cv2.COLOR_RGB2GRAY)

        cascade = cv2.CascadeClassifier(self.cascade_path)
        facerect = cascade.detectMultiScale(img_gray, scaleFactor=1.2, minNeighbors=2, minSize=(10, 10))

        for rect in facerect:
            # 顔画像のcrop
            start_x = rect[0] - rect[2]//2
            start_y = rect[1] - rect[3]//2
            img3 = img.crop((start_x, start_y, rect[0] + rect[2] + rect[2]//2, rect[1] + rect[3] + rect[3]//2))
            # 顔の変換
            face_img = self.trans_face(pil_to_tensor(img3), c_trg).resize((2 * rect[2], 2 * rect[3]), Image.LANCZOS)

            # 背景切り取り
            cut_img = self.deeplab.validation(img3)
            cut_img = cv2.resize(cut_img, (2 * rect[2], 2 * rect[3]), interpolation=cv2.INTER_NEAREST)
            # 合成
            face_img = self.merge_img(pil_to_cv2(img3.resize((2 * rect[2], 2 * rect[3]))), pil_to_cv2(face_img), cut_img)
            img.paste(face_img, (start_x, start_y))

        return pil_to_cv2(img)

    def restore_model(self):
        self.G.load_state_dict(torch.load("./models/generator_5.ckpt", map_location=lambda storage, loc: storage))
        #self.C.load_state_dict(torch.load("./models/classification.ckpt", map_location=lambda storage, loc: storage))

    def trans_face(self, x_real, c):
        with torch.no_grad():
            x_real = self.G(x_real, c)
            return tensor_to_pil(x_real.data.cpu(), nrow=1, padding=0)


if __name__ == '__main__':
    img_path = "images/175397.jpg"
    input_img = Image.open(img_path)
    trans = TransFace()
    c_trg = [0, 1, 0, 1, 1]
    # print(trans.classification(input_img))
    cv2.imwrite("cascade.jpg", trans.trans(input_img))
