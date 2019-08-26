import tkinter.filedialog as tkFiledDialog
import tkinter as tk
from tkinter import *
from PIL import Image, ImageTk
from tran_face import TransFace

class MainWindow(object):
    def __init__(self):
        root = tk.Tk()
        root.geometry('400x400')
        self.frame1 = tk.Frame(root)
        self.frame1.pack(side=TOP)
        self.frame2 = tk.Frame(root)
        self.frame2.pack(side=BOTTOM)
        self.frame_img = tk.Frame(root)
        self.frame_img.pack(side=BOTTOM)

        self.c_trg_labels = ["black hair", "blonde hair", "brown hair", "male", "age"]
        self.c_trg = [0, 1, 0, 1, 1]
        self.trans_face = TransFace()
        self.imgs = {}
        self.fType = [("画像ファイル", "*.jpg"), ("画像ファイル", "*.png")]

        # loader
        self.label_button = []
        self.load_button = tk.Button(self.frame1, text="load file", command=self.load_file)
        self.load_button.grid(row=0, column=4)
        for i, label in enumerate(self.c_trg_labels):
            self.label_button.append(tk.Button(self.frame2, text=label + "\n{}".format(self.c_trg[i]), command=self.change_c_trg(i)))
            self.label_button[i].pack(side=LEFT)
        root.mainloop()

    def load_file(self):
        self.imgs.clear()
        self.file_name = tkFiledDialog.askopenfilename(filetype=self.fType)
        self.print_load_image()

    def destroy_child(self, frame):
        children = frame.winfo_children()
        for child in children:
            child.destroy()

    def print_load_image(self):
        self.img = Image.open(self.file_name)
        self.trans_face.cut_face(self.img)
        self.destroy_child(self.frame_img)
        w, h = self.img.size
        canvas = tk.Canvas(self.frame_img, width=w, height=h)
        canvas.place(x=100, y=350)
        self.imgs[0] = ImageTk.PhotoImage(self.img)
        canvas.create_image(3, 3, image=self.imgs[0], anchor=tk.NW)
        canvas.grid(row=0, column=0)
        canvas.bind("<1>", self.select_face)

    def print_transed_image(self):
        print(self.select_img)
        img_tr = self.trans_face.translation(img=self.select_img[4], c_trg=self.c_trg).resize(self.select_img[4].size)
        img_tr.save("tesat.png")
        img_tr = self.trans_face.merge_bg(self.img, img_tr, self.select_img)
        self.destroy_child(self.frame_img)
        w, h = self.img.size
        canvas = tk.Canvas(self.frame_img, width=w, height=h)
        canvas.place(x=300, y=350)
        self.imgs[0] = ImageTk.PhotoImage(img_tr)
        canvas.create_image(3, 3, image=self.imgs[0], anchor=tk.NW)
        canvas.grid(row=0, column=1)
        canvas.bind("<1>", self.select_face)

    def select_face(self, event):
        face_data = self.trans_face.get_faces()
        x, y = event.x, event.y
        for face in face_data:
            print((x, y))
            print(face)
            if face[0] <= x <= face[2]:
                if face[1] <= y <= face[3]:
                    self.select_img = face
                    print("True")
                    break
                else:
                    print("yFalse")
            else:
                print("xFalse")
        self.print_transed_image()

    def change_c_trg(self, index):
        def x():
            self.c_trg[index] = 1 ^ self.c_trg[index]
            self.label_button[index]["text"] = self.c_trg_labels[index] + "\n{}".format(self.c_trg[index])
            if self.file_name is not None:
                self.print_transed_image()
        return x


if __name__ == "__main__":
    MainWindow()
