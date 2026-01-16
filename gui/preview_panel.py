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
        """缩放图片到标签大小（优化鲁棒性，避免标签未渲染导致的尺寸异常）"""
        # 优化：避免标签未完成渲染时，winfo_width()/winfo_height()返回0
        label_w = label.winfo_width()
        label_h = label.winfo_height()
        
        # 兜底尺寸：若标签未渲染完成，使用默认宽高
        if label_w <= 1 or label_h <= 1:
            label_w, label_h = 400, 300
        
        img_w, img_h = img.size
        scale = min(label_w/img_w, label_h/img_h, 1.0)
        new_w = int(img_w * scale)
        new_h = int(img_h * scale)
        
        return img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    def clear(self):
        """清空预览区域（增加鲁棒性，避免组件已销毁导致的报错）"""
        try:
            # 1. 先判断组件是否存在且未被销毁（winfo_exists()：返回1表示组件有效，0表示已销毁）
            if hasattr(self, 'left_label') and self.left_label.winfo_exists():
                self.left_label.config(text="暂无原始内容", image="")
            if hasattr(self, 'right_label') and self.right_label.winfo_exists():
                self.right_label.config(text="暂无预测结果", image="")
        except tk.TclError:
            # 2. 捕获Tkinter组件操作异常（兜底处理，避免程序崩溃）
            pass
        finally:
            # 3. 无论是否报错，都清空图片缓存，释放内存
            self.left_img = None
            self.right_img = None