#!/usr/bin/env python3
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

class PreviewPanel:
    """分栏预览面板"""
    def __init__(self, parent):
        self.parent = parent
        self.frame = ttk.LabelFrame(parent, text="预览区域")
        self.frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # 分割面板
        self.paned = ttk.PanedWindow(self.frame, orient=tk.HORIZONTAL)
        self.paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 左侧原始预览
        self.left_frame = ttk.LabelFrame(self.paned, text="原始内容")
        self.paned.add(self.left_frame, weight=1)
        self.left_label = ttk.Label(self.left_frame, text="暂无原始内容", anchor="center")
        self.left_label.pack(fill=tk.BOTH, expand=True)
        
        # 右侧预测结果预览
        self.right_frame = ttk.LabelFrame(self.paned, text="预测结果")
        self.paned.add(self.right_frame, weight=1)
        self.right_label = ttk.Label(self.right_frame, text="暂无预测结果", anchor="center")
        self.right_label.pack(fill=tk.BOTH, expand=True)
        
        # 缓存
        self.left_img = None
        self.right_img = None

    def show_original_image(self, image_path):
        """显示原始图片"""
        try:
            img = Image.open(image_path)
            img = self._resize_img_to_label(img, self.left_label)
            self.left_img = ImageTk.PhotoImage(img)
            self.left_label.config(image=self.left_img, text="")
        except Exception as e:
            self.left_label.config(text=f"加载失败：{str(e)}", image="")

    def show_result_image(self, image_path):
        """显示预测结果图片"""
        try:
            img = Image.open(image_path)
            img = self._resize_img_to_label(img, self.right_label)
            self.right_img = ImageTk.PhotoImage(img)
            self.right_label.config(image=self.right_img, text="")
        except Exception as e:
            self.right_label.config(text=f"加载失败：{str(e)}", image="")

    def _resize_img_to_label(self, img, label):
        """缩放图片到标签大小"""
        label_w = label.winfo_width() or 400
        label_h = label.winfo_height() or 300
        
        img_w, img_h = img.size
        scale = min(label_w/img_w, label_h/img_h, 1.0)
        new_w = int(img_w * scale)
        new_h = int(img_h * scale)
        
        return img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    def clear(self):
        """清空预览区域"""
        self.left_label.config(text="暂无原始内容", image="")
        self.right_label.config(text="暂无预测结果", image="")
        self.left_img = None
        self.right_img = None