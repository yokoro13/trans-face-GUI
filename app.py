import tkinter.filedialog as tkFiledDialog
import tkinter as tk
from tkinter import *
from PIL import Image, ImageTk
from tran_face import TransFace
from util import cv2_to_pil

global file_name


def f():
    print_right_image()


def print_left_image():
    global img
    img = Image.open(file_name)
    img_org = img.resize((128, 156))
    canvas = tk.Canvas(frame_img, bg="purple", width=128, height=156)
    canvas.place(x=100, y=350)
    img_org = ImageTk.PhotoImage(img_org)
    imgs.append(img_org)
    canvas.create_image(3, 3, image=img_org, anchor=tk.NW)
    canvas.grid(row=0, column=0)


def print_right_image():
    img_tr = cv2_to_pil(trans_face.trans(img=img, c_trg=c_trg)).resize((128, 156))
    canvas = tk.Canvas(frame_img, bg="purple", width=128, height=156)
    canvas.place(x=300, y=350)
    img_tr = ImageTk.PhotoImage(img_tr)
    imgs.append(img_tr)
    canvas.create_image(3, 3, image=img_tr, anchor=tk.NW)
    canvas.grid(row=0, column=1)


def load_file():
    imgs.clear()
    global file_name
    file_name = tkFiledDialog.askopenfilename(filetype=fType)
    print_left_image()


def change_c_trg(index):
    def x():
        c_trg[index] = 1 ^ c_trg[index]
        label_button[index]["text"] = c_trg_labels[index] + "\n{}".format(c_trg[index])
        if file_name is not None:
            print_right_image()
    return x


def main():
    # loader
    load_button = tk.Button(frame1, text="load file", command=load_file)
    load_button.grid(row=0, column=4)

    for i, label in enumerate(c_trg_labels):
        label_button.append(tk.Button(frame2, text=label + "\n{}".format(c_trg[i]), command=change_c_trg(i)))
        label_button[i].pack(side=LEFT)

    root.mainloop()


if __name__ == "__main__":
    c_trg_labels = ["black hair", "blonde hair", "brown hair", "male", "age"]
    c_trg = [0, 1, 0, 1, 1]

    root = tk.Tk()
    root.geometry('400x400')
    frame1 = tk.Frame(root)
    frame1.pack(side=TOP)
    frame2 = tk.Frame(root)
    frame2.pack(side=BOTTOM)
    frame_img = tk.Frame(root)
    frame_img.pack(side=BOTTOM)

    trans_face = TransFace()
    imgs = []
    label_button = []

    fType = [("画像ファイル", "*.jpg"), ("画像ファイル", "*.png")]
    file_name = None
    img = None
    main()