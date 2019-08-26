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
        self.restore_model()
        self.face_data = []

    def merge_img(self, img_org, img_trans, img_mask):
        img_mask = np.where(img_mask == 0, img_org, img_trans)
        return cv2_to_pil(img_mask)

    def cut_face(self, img):
        self.clear_faces()
        img = img.convert("RGB")
        img_gray = np.asarray(img)
        img_gray = cv2.cvtColor(img_gray, cv2.COLOR_RGB2GRAY)

        cascade = cv2.CascadeClassifier(self.cascade_path)
        facerect = cascade.detectMultiScale(img_gray, scaleFactor=1.2, minNeighbors=2, minSize=(10, 10))

        for rect in facerect:
            # 顔画像のcrop
            start_x = rect[0] - rect[2]//2
            start_y = rect[1] - rect[3]//2
            end_x = rect[0] + rect[2] + rect[2]//2
            end_y = rect[1] + rect[3] + rect[3]//2
            img3 = img.crop((start_x, start_y, end_x, end_y))
            self.face_data.append([start_x, start_y, end_x, end_y, img3])

    def translation(self, img, c_trg=None):
        if c_trg is None:
            c_trg = [0, 1, 0, 1, 1]
        c_trg = np.asarray(c_trg, dtype=np.float32)
        c_trg = torch.from_numpy(c_trg)
        c_trg = c_trg.view(1, c_trg.size(0)).to(self.device)
        h, w = img.size
        img_size = (2 * h, 2 * w)

        # 顔の変換
        face_img = self.trans_face(pil_to_tensor(img), c_trg).resize(img_size, Image.LANCZOS)

        # 背景切り取り
        cut_img = self.deeplab.validation(img)
        cut_img = cv2.resize(cut_img, img_size, interpolation=cv2.INTER_NEAREST)
        # 合成
        return self.merge_img(pil_to_cv2(img.resize(img_size)), pil_to_cv2(face_img), cut_img)

    def merge_bg(self, img_bg, face_img, face_data):
        img_bg.paste(face_img, (face_data[0], face_data[1]))
        return img_bg

    def get_faces(self):
        return self.face_data

    def clear_faces(self):
        self.face_data.clear()

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
