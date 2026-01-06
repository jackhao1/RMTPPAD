#!/usr/bin/env python3
import tkinter as tk
from tkinter import ttk

class LoadingWindow:
    """模型加载等待窗口类"""
    def __init__(self, parent):
        self.parent = parent
        self.window = None
        self.label = None
        self.index = 0
        self.states = ["|", "/", "-", "\\"]
        
        self._create_window()
        self._update_animation()

    def _create_window(self):
        """创建加载窗口"""
        self.window = tk.Toplevel(self.parent)
        self.window.title("模型加载中")
        self.window.geometry("300x100")
        self.window.resizable(False, False)
        self.window.transient(self.parent)
        self.window.grab_set()
        
        # 窗口居中
        parent_x = self.parent.winfo_x()
        parent_y = self.parent.winfo_y()
        parent_w = self.parent.winfo_width()
        parent_h = self.parent.winfo_height()
        x = parent_x + (parent_w - 300) // 2
        y = parent_y + (parent_h - 100) // 2
        self.window.geometry(f"300x100+{x}+{y}")
        
        # UI元素
        ttk.Label(self.window, text="正在加载模型，请稍候...", font=("Arial", 12)).pack(pady=10)
        self.label = ttk.Label(self.window, text="", font=("Arial", 16))
        self.label.pack()

    def _update_animation(self):
        """更新加载动画"""
        if self.window and self.label:
            self.label.config(text=self.states[self.index])
            self.index = (self.index + 1) % len(self.states)
            self.window.after(100, self._update_animation)

    def close(self):
        """关闭加载窗口"""
        if self.window:
            self.window.grab_release()
            self.window.destroy()
            self.window = None