import tkinter as tk
import cv2
from PIL import Image, ImageTk
import os
import numpy as np
from skimage import io
import matplotlib.pyplot as plt
from scipy import stats

# utils for depth value visualization with on mouse pointer from depth image 
class app():

    def __init__(self, main, color, pred_depth, gt_depth, measures = None):
        self.measures = measures
        self.main = main
        self.rect1 = None 
        self.x = self.y = 0      
        self.start_x = None
        self.start_y = None
        
        assert pred_depth.shape == gt_depth.shape 
        self.color = color
        self.pred_depth = pred_depth
        self.gt_depth = gt_depth
        
        self.height, self.width = self.pred_depth.shape
        self.pred_photo = Depth_to_Tk(self.pred_depth)
        self.gt_photo = Depth_to_Tk(self.gt_depth)
        self.color_photo = ImageTk.PhotoImage(Image.fromarray(self.color))
        
        self.color_canvas = self.create_image_canvas(self.color_photo, 0, 0, anchor= 'nw')
        self.pred_canvas = self.create_image_canvas(self.pred_photo, 1, 0, anchor= 'nw')
        self.gt_canvas = self.create_image_canvas(self.gt_photo, 1,  1 , anchor= 'nw') 
        self.text_canvas(self.width, 0)
        
    def create_image_canvas(self, image, row, col, anchor = 'nw'):
        
        canvas = tk.Canvas(self.main, width = self.width, height = self.height, cursor ='cross')
        canvas.create_image(0, 0, anchor = anchor, image= image)
        canvas.grid(row = row, column = col)
        self.mouse_bind(canvas)
        return canvas
    
    def mouse_bind(self, canvas):
        
        canvas.bind("<ButtonPress-1>", self.on_button_press)
        canvas.bind("<B1-Motion>", self.on_move_press)
        canvas.bind("<ButtonRelease-1>", self.on_button_release)
        canvas.bind('<Button-3>', self.delete)
        canvas.bind("<Motion>", self.onMouseOver)
        
    def on_button_press(self, event):
        
        self.start_x, self.start_y = event.x, event.y
        
        self.rect1 = self.color_canvas.create_rectangle(self.x, self.y, 1, 1, fill="", outline ='red', width = 2, tags="color" )
        self.rect2 = self.pred_canvas.create_rectangle(self.x, self.y, 1, 1, fill="", outline ='green', width = 2, tags="pred")
        self.rect3 = self.gt_canvas.create_rectangle(self.x, self.y, 1, 1, fill="", outline ='green', width = 2, tags="gt")
        
    def on_move_press(self, event):
        
        self.curX , self.curY = event.x, event.y
        self.color_canvas.coords(self.rect1, self.start_x, self.start_y, self.curX, self.curY)
        self.pred_canvas.coords(self.rect2, self.start_x, self.start_y, self.curX, self.curY)
        self.gt_canvas.coords(self.rect3, self.start_x, self.start_y, self.curX, self.curY)
    
    def on_button_release(self, event):
        info = self.object_details()
        if info: 
            abs_rel =info['abs_rel']
            self.pred_Mode.config(text=f"Object (mode) : {int(info['pred_mode']['value'])} mm")
            self.gt_Mode.config(text=f"Object (mode) : {int(info['gt_mode']['value'])} mm")
            self.pred_mean.config(text=f"Object (mean) : {int(info['pred_mean'])} mm")
            self.gt_mean.config(text=f"Object (mean) : {int(info['gt_mean'])} mm")
            self.obj_eval.config(text="Object (abs_rel) : '%.4f'"%abs_rel)
            self.pred_min_max.config(text=f"min,max : {int(info['pred_min'])} , {int(info['pred_max'])}")
            self.gt_min_max.config(text=f"min,max : {int(info['gt_min'])} , {int(info['gt_max'])}")
            
        else: 
            pass
    
    def text_canvas(self, x, y):
        
        self.canvas = tk.Canvas(self.main , width = self.width, height = self.height,  highlightthickness=0)
        self.canvas.place(x = x , y = y) 
        
        self.abs_rel = eval_depth(self.gt_depth[self.gt_depth != 0] , self.pred_depth[self.gt_depth != 0])
         
        self.label_eval = tk.Label(self.main , text= f"Absolute Relative Error = {self.abs_rel}", fg="white", bg= 'gray1', font=("'serif 10'", 15))
        self.label_eval.place(x = x+10, y=y+5)
 
        self.pred_header = tk.Label(self.main , text='Predicted Depth', fg="white", bg= 'gray1', font=("'serif 10'", 15))
        self.pred_header.place(x = x+10, y=y+70)   
 
        self.gt_header = tk.Label(self.main , text='Ground Truth Depth', fg="white", bg= 'gray1', font=("'serif 10'", 15))
        self.gt_header.place(x = x+320, y=y+70)
        
        self.pred_Mode = tk.Label(self.main , text='Object (mode) =', fg="blue", bg= 'green1', font=("'serif 10'", 15))
        self.pred_Mode.place(x = x+10, y=y+100)
        
        self.gt_Mode = tk.Label(self.main , text='Object (mode) =', fg="blue", bg= 'green1', font=("'serif 10'", 15))
        self.gt_Mode.place(x = x+320, y=y+100)
        
        self.pred_mean = tk.Label(self.main , text='Object (mean) =', fg="blue", bg= 'green1', font=("'serif 10'", 15))
        self.pred_mean.place(x = x+10, y=y+125)
        
        self.gt_mean = tk.Label(self.main , text='Object (mean) =', fg="blue", bg= 'green1', font=("'serif 10'", 15))
        self.gt_mean.place(x = x+320, y=y+125)
        
        self.pred_min_max = tk.Label(self.main , text='min,max = ', fg="blue", bg= 'green1', font=("'serif 10'", 15))
        self.pred_min_max.place(x = x+10, y=y+150)
        
        self.gt_min_max = tk.Label(self.main , text='max,max = ', fg="blue", bg= 'green1', font=("'serif 10'", 15))
        self.gt_min_max.place(x = x+320, y=y+150)
         
        self.obj_eval = tk.Label(self.main , text= 'Object (abs_rel) =', fg="blue", bg= 'green1', font=("'serif 10'", 15))
        self.obj_eval.place(x = x+10, y=y+175)
        
        self.pred_distance = tk.Label(self.main , text='', fg="blue", bg= 'green1', font=("'serif 10'", 15))
        self.pred_distance.place(x = x+10, y=y+225)
        
        self.gt_distance = tk.Label(self.main , text='', fg="blue", bg= 'green1', font=("'serif 10'", 15))
        self.gt_distance.place(x = x+320, y=y+225)
        
        self.labelDiff = tk.Label(self.main , text='', fg="blue", bg = 'green1', font=("'serif 10'", 15))
        self.labelDiff.place(x = x+10, y=y+250)
 
        self.labelCoord = tk.Label(self.main , text='', fg="gray1", font=("'serif 10'", 15))
        self.labelCoord.place(x = x+10, y=y+275)  
        
    def onMouseOver(self, event):    
        
        x, y = event.x, event.y
        try:
            pred_text= 'distance : ' + str(self.pred_depth[y][x]) +' mm'
            gt_text =  'distance : ' + str(self.gt_depth[y][x]) +' mm'
            text_diff = 'Diff (gt-pred) : ' + str(int(self.pred_depth[y][x] - self.gt_depth[y][x])) +' mm'
        
            self.labelDiff.config (text= text_diff )
            self.pred_distance.config(text= pred_text)
            self.gt_distance.config(text= gt_text)
            self.labelCoord.config(text='Coordinate = (x: %s, y: %s)'%(str(x),str(y)))
        
        except:
            pass
     
    def delete(self, event):
        
        self.color_canvas.delete('color')
        self.pred_canvas.delete('pred')
        self.gt_canvas.delete('gt')
        # self.canvas.delete('all')
    
    def object_details(self):

        if self.start_x < self.curX and self.start_y < self.curY:
            x1,y1,x2,y2 = self.start_x, self.start_y, self.curX, self.curY
            
            pred_crop = self.pred_depth[ y1:y2, x1:x2]
            gt_crop = self.gt_depth[ y1:y2, x1:x2]
            
            gt_obj = gt_crop[gt_crop != 0]
            pred_obj = pred_crop[gt_crop !=0]
            
            info = {}
            #whole image eval   
            info['gt_mode'] = {'value': stats.mode(gt_obj.flatten())[0][0] , 'count' :stats.mode(gt_obj.flatten())[1][0]}
            info['gt_mean'] = gt_obj.mean()
            info['gt_max'] = gt_crop.max()
            info['gt_min'] = gt_crop.min()
            info['pred_mode'] = {'value': stats.mode(pred_obj.flatten())[0][0] , 'count' :stats.mode(pred_obj.flatten())[1][0]}
            info['pred_mean'] = pred_obj.mean()
            info['pred_max'] = pred_crop.max()
            info['pred_min'] = pred_crop.min()
            info['abs_rel'] = eval_depth(gt_obj, pred_obj)
            return info
        
        else:
            info = None
            return info
        
def eval_depth(gt, pred):
    abs_rel = np.mean(np.abs(gt - pred) / gt) 
    
    return abs_rel

def Depth_to_Tk(img):
    color = cv2.applyColorMap(cv2.convertScaleAbs(img, alpha=0.0355), cv2.COLORMAP_JET)[:,:,::-1]    
    tk_im = ImageTk.PhotoImage(Image.fromarray(color))
    return tk_im 

def compute_eval(gt, pred):
    print('[INFO] Computing Error\n')
    valid_mask = gt > 0
    pred_eval, gt_eval = pred[valid_mask], gt[valid_mask]

    threshold = np.maximum((gt_eval / pred_eval), (pred_eval / gt_eval))

    delta1 = (threshold < 1.25).mean()
    delta2 = (threshold < 1.25 ** 2).mean()
    delta3 = (threshold < 1.25 ** 3).mean()

    abs_diff = np.abs(pred_eval - gt_eval)

    # mae = np.mean(abs_diff)
    # rmse = (gt_eval - pred_eval) ** 2
    # rmse = np.sqrt(rmse.mean())
    # rmse = np.sqrt(np.mean(np.power(abs_diff, 2)))
    abs_rel = np.mean(abs_diff / gt_eval)

    err = np.log(pred_eval) - np.log(gt_eval)
    silog = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2) * 100
    
    measures = {}
    measures['silog'] = silog.round(4)
    # measures['mae'] = mae.round(4)
    # measures['rmse'] = rmse.round(4)
    measures['abs_rel'] = abs_rel.round(4)
    measures['d1'] = delta1.round(4)
    measures['d2'] = delta2.round(4)
    measures['d3'] = delta3.round(4)
    
    for k in measures:
        print(f'{k}\t: {measures[k]}')
    return measures
def add_mask(depth, validity_mask): 
    MIN_DEPTH = 0.0
    MAX_DEPTH = min(300, np.percentile(depth, 99))
    depth = np.clip(depth, MIN_DEPTH, MAX_DEPTH) 
    depth [validity_mask== 0] = 0 
    return depth

def show(img, mode = 'cv2'):
    if mode == 'cv2':
        cv2.imshow('test', img[:, :, ::-1]); cv2.waitKey(0); cv2.destroyAllWindows()
    elif mode == 'plt':
        if img.ndim == 2:
            cmap = input('Type cmap')
            plt.imshow(img, cmap = cmap); plt.show()
        else: 
            plt.imshow(img); plt.show()
        
    elif mode == 'sk':
        io.imshow(img); io.show()

def compare_plot(figsize= (25,25), cmap = 'gray', **imgs):

    if len(imgs)> 1:
        _, axes = plt.subplots(1, len(imgs), figsize = figsize)
        axes = axes.flatten()
        
        for (name,img), ax in zip(imgs.items(), axes):
            if img.ndim ==2:
                ax.imshow(img,cmap=cmap)
            else:
                ax.imshow(img)
            ax.set_title(name)
        plt.show()

    else:
        name, img = list(imgs.items())[0]
        plt.figure(figsize= figsize)
        plt.title(name)
        if img.ndim == 2:
            plt.imshow(img, cmap= cmap); plt.show()
        else:
            plt.imshow(img); plt.show()    