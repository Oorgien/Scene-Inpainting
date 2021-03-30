import matplotlib
matplotlib.use('Agg')
from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk, ImageGrab, ImageDraw
import io
import os
import copy
from utils import get_config
from models.RaGAN import test
import torchvision.transforms as transforms


class App():
    def __init__(self):
        if not os.path.isdir('gui/images'):
            os.makedirs('gui/images')
        self.image_dir = 'gui/images'

        self.root = Tk()

        self.frame = Frame(self.root)
        self.frame.grid()

        # Thickness of the brush
        self.brush_thickness = IntVar(self.root)
        self.brush_thickness.set(10)  # default value

        self.brush_options = OptionMenu(self.root, self.brush_thickness, *list(range(1, 25)))
        self.brush_options.grid()

        # Set up canvas
        self.label = Label(self.root, text='Upload image', font=(
            'Verdana', 15), bg="white")
        self.root.title("Painting")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        self.canvas = Canvas(self.root, bg='black', width=256, height=512)
        self.canvas.bind("<Button-1>", self.savePosn)
        self.canvas.bind("<B1-Motion>", self.addLine)
        self.canvas.grid(row=2, column=3)

        # Drawing canvas for white mask capturing
        self.loaded_image = None
        self.clear_mask()

        # Button 1 - open image
        self.b1 = Button(self.root, text="Open image",
                    width=10, height=2, command=self.open_img)
        self.b1.grid(row=1, column=1)
        # Button 2 - clear drawn mask
        self.b2 = Button(self.root, text="Clear mask",
                    width=10, height=2, command=self.clear_mask)
        self.b2.grid(row=2, column=1)

        # Button 3 - Process inpainting
        self.b3 = Button(self.root, text="Inpainting",
                         width=10, height=2, command=self.inpaint)
        self.b3.grid(row=3, column=1)

        # Button 3 - Clear result
        self.b4 = Button(self.root, text="Clear result",
                         width=10, height=2, command=self.clear_result)
        self.b4.grid(row=2, column=0)

        self.root.mainloop()

    def savePosn(self, event):
        """
        Save position of mouse

        :param event: - click of the mouse
        """
        self.lastx, self.lasty = event.x, event.y

    def addLine(self, event):
        """
        :param event: - click of the mouse
        """
        thickness_val = self.brush_thickness.get()
        self.canvas.create_oval(self.lastx-thickness_val,
                                self.lasty-thickness_val,
                                self.lastx+thickness_val,
                                self.lasty+thickness_val,
                                fill="white", outline='white')
        self.drawing.ellipse([self.lastx-thickness_val,
                              self.lasty-thickness_val,
                              self.lastx+thickness_val,
                              self.lasty+thickness_val],
                             fill=1, outline=1)
        self.savePosn(event)

    def save_mask(self):
        """
        Save only mask
        """
        self.canvas.update()
        self.mask.save(os.path.join(self.image_dir, 'mask_3.tiff'))

    def clear_mask(self):
        if self.loaded_image != None:
            self.canvas.create_image(0, 0, image=self.render, anchor=NW)
        self.mask = Image.new("1", (256, 256), color=0)
        self.drawing = ImageDraw.Draw(self.mask)

    def clear_result(self):
        if self.loaded_image != None:
            self.canvas.create_image(0, 256, image=self.render, anchor=NW)

    def open_img(self):
        self.filename = filedialog.askopenfilename()
        self.loaded_image = Image.open(self.filename)
        self.render = ImageTk.PhotoImage(self.loaded_image)
        self.canvas.create_image(0, 0, image=self.render, anchor=NW)
        self.canvas.update()

        self.mask = Image.new("1", (256, 256), color=0)
        self.drawing = ImageDraw.Draw(self.mask)

    def inpaint(self):
        """
        Process inpainting
        """
        self.save_mask()
        self.config = "configs/configs_RaGAN/config_model_2_bce_afterlearn.yaml"
        args = get_config(self.config)
        to_tensor = transforms.ToTensor()
        self.predicted = test(args,
                              self.loaded_image,
                              to_tensor(self.mask),
                              self.image_dir)

        self.result = ImageTk.PhotoImage(self.predicted)

        self.canvas.create_image(0, 256, image=self.result, anchor=NW)
        self.canvas.update()

    def save_mask_2(self):
        self.canvas.update()
        # self.ps = self.canvas.postscript(height=256, width=256, colormode='color')
        self.ps = self.canvas.postscript(file=os.path.join(self.image_dir, 'ps.eps'),
                                         height=256, width=256, colormode='color')
        # self.mask = Image.open(io.BytesIO(self.ps.encode('utf-8')))
        self.mask = Image.open(os.path.join(self.image_dir, 'ps.eps'))
        # print (self.mask.shape())
        self.mask.save(os.path.join(self.image_dir, 'mask.jpg'))

    def save_mask_3(self):
        self.canvas.update()
        x = self.root.winfo_rootx() + self.canvas.winfo_x()
        y = self.root.winfo_rooty() + self.canvas.winfo_y()
        x1 = x + self.canvas.winfo_width()
        y1 = y + self.canvas.winfo_height()
        box = (x, y, x1, y1)
        # box = (0, 0, 500, 500)
        ImageGrab.grab().crop((box)).save(os.path.join(self.image_dir,"mask_2.tiff"))
        self.mask = Image.open(os.path.join(self.image_dir, 'mask_2.tiff'))

if __name__ == "__main__":
    app = App()
