#!/usr/bin/env python3
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import sys
import threading
import time
from core.config import Config
from core.predictors import ImagePredictor, VideoPredictor, CameraPredictor
from core.video_player import IndependentVideoPlayer
from gui.preview_panel import PreviewPanel

# 设置模块路径
sys.path.insert(0, Config.MTDETR_PATH)
sys.path.insert(0, Config.ULTRALYTICS_ROOT)
if "PYTHONPATH" in os.environ:
    os.environ["PYTHONPATH"] = f"{Config.MTDETR_PATH};{Config.ULTRALYTICS_ROOT};{os.environ['PYTHONPATH']}"
else:
    os.environ["PYTHONPATH"] = f"{Config.MTDETR_PATH};{Config.ULTRALYTICS_ROOT}"

class MTDETRApp:
    """主应用类（打包前稳定版，可直接上传GitHub）"""
    def __init__(self, root, model):
        self.root = root
        self.model = model
        self.root.title("RMT-PPAD 预测工具")
        self.root.geometry("1200x800")
        self.root.resizable(True, True)
        
        # 全局变量（仅核心属性，无线程锁）
        self.predictor = None
        
        # 构建UI
        self._create_ui()
        
        # 初始化独立视频播放器
        self.video_player = IndependentVideoPlayer(self.preview_panel, self.logger)
        
        # 初始化日志
        self.logger("📢 RMT-PPAD预测工具已启动")
        self.logger(f"🖼️ 结果保存路径：{Config.SAVE_ROOT}")
        self.logger("💡 操作步骤：1.选数据源 → 2.预测 → 查看实时预览")

    def _create_ui(self):
        """构建精简UI（打包前稳定版）"""
        # 1. 预测配置区域
        frame_source = ttk.LabelFrame(self.root, text="预测配置")
        frame_source.pack(fill=tk.X, padx=10, pady=5)
        
        # 预测类型
        ttk.Label(frame_source, text="预测类型：").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.combo_predict_type = ttk.Combobox(frame_source, values=["图片", "视频", "摄像头"], width=10, state="readonly")
        self.combo_predict_type.current(0)
        self.combo_predict_type.grid(row=0, column=1, padx=5, pady=5)
        
        # 数据源
        ttk.Label(frame_source, text="数据源：").grid(row=0, column=2, padx=5, pady=5, sticky=tk.W)
        self.entry_source = ttk.Entry(frame_source, width=60)
        self.entry_source.grid(row=0, column=3, padx=5, pady=5)
        ttk.Button(frame_source, text="选择图片", command=self.select_image).grid(row=0, column=4, padx=5, pady=5)
        ttk.Button(frame_source, text="选择视频", command=self.select_video).grid(row=0, column=5, padx=5, pady=5)
        
        # 摄像头ID
        ttk.Label(frame_source, text="摄像头ID：").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        self.entry_camera_id = ttk.Entry(frame_source, width=10)
        self.entry_camera_id.insert(0, "0")
        self.entry_camera_id.grid(row=1, column=1, padx=5, pady=5)

        # 2. 控制按钮区域
        frame_ctrl = ttk.Frame(self.root)
        frame_ctrl.pack(padx=10, pady=5)
        ttk.Button(frame_ctrl, text="启动预测", command=self.start_predict).grid(row=0, column=0, padx=5, pady=5)
        ttk.Button(frame_ctrl, text="停止预测", command=self.stop_predict).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(frame_ctrl, text="清空预览", command=self.clear_preview).grid(row=0, column=2, padx=5, pady=5)

        # 3. 分栏预览区域
        self.preview_panel = PreviewPanel(self.root)

    def select_image(self):
        """选择图片（稳定版）"""
        path = filedialog.askopenfilename(
            title="选择预测图片",
            filetypes=[("图片文件", "*.jpg;*.jpeg;*.png;*.bmp"), ("所有文件", "*.*")]
        )
        if path:
            self.entry_source.delete(0, tk.END)
            self.entry_source.insert(0, path)
            self.preview_panel.show_original_image(path)
            self.logger(f"📁 已选择图片：{path}")

    def select_video(self):
        """选择视频（稳定版）"""
        path = filedialog.askopenfilename(
            title="选择预测视频",
            filetypes=[("视频文件", "*.mp4;*.avi;*.mov;*.mkv"), ("所有文件", "*.*")]
        )
        if path:
            self.entry_source.delete(0, tk.END)
            self.entry_source.insert(0, path)
            self.logger(f"📁 已选择视频：{path}")

    def start_predict(self):
        """启动预测（稳定版，无线程锁）"""
        # 检查是否已有预测在运行
        if self.predictor and hasattr(self.predictor, 'is_running') and self.predictor.is_running:
            self.logger("⚠️ 预测已在进行中")
            return
        
        predict_type = self.combo_predict_type.get()
        source = self.entry_source.get()
        
        # 验证数据源
        if predict_type == "图片" and not os.path.exists(source):
            self.logger("❌ 图片文件不存在")
            messagebox.showwarning("警告", "图片文件不存在！")
            return
        if predict_type == "视频" and not os.path.exists(source):
            self.logger("❌ 视频文件不存在")
            messagebox.showwarning("警告", "视频文件不存在！")
            return
        
        # 创建预测器并启动
        if predict_type == "图片":
            self.predictor = ImagePredictor(self.model, self.preview_panel, self.logger)
            threading.Thread(target=self.predictor.start, args=(source,), daemon=True).start()
        elif predict_type == "视频":
            self.predictor = VideoPredictor(self.model, self.preview_panel, self.logger, self.video_player)
            threading.Thread(target=self.predictor.start, args=(source, Config.VIDEO_SAVE_ROOT), daemon=True).start()
        elif predict_type == "摄像头":
            self.predictor = CameraPredictor(self.model, self.preview_panel, self.logger)
            threading.Thread(target=self.predictor.start, args=(self.entry_camera_id.get(),), daemon=True).start()
        
        self.logger(f"🚀 开始{predict_type}预测")

    def stop_predict(self):
        """停止预测（稳定版，无线程锁）"""
        try:
            if self.predictor:
                self.predictor.stop()
                self.predictor = None
                self.logger("🛑 预测已停止")
        except Exception as e:
            self.logger(f"❌ 停止预测时出错：{str(e)}")
            messagebox.showwarning("警告", f"停止预测时出现异常：\n{str(e)}")

    def clear_preview(self):
        """清空预览（稳定版）"""
        self.stop_predict()
        self.preview_panel.clear()
        self.logger("🗑️ 已清空预览区域")

    def logger(self, content):
        """日志输出（稳定版）"""
        log_msg = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {content}"
        print(log_msg)

# ========== 后台加载模型 + 启动GUI（稳定版） ==========
def load_model_background():
    """后台加载模型（稳定版）"""
    print("📌 正在后台加载模型...")
    Config.init_dirs()
    try:
        Config.create_symlinks()
    except Exception as e:
        print(f"⚠️ 软链接创建失败（可选）：{e}")
    
    # 加载MTDETR模型
    model = None
    try:
        from ultralytics import MTDETR
        model = MTDETR(Config.MODEL_WEIGHT_PATH)
        print("✅ 模型加载完成！")
    except Exception as e:
        print(f"❌ 模型加载失败：{str(e)}")
        messagebox.showerror("错误", f"模型加载失败：\n{str(e)}")
        sys.exit(1)
    
    return model

def main():
    """程序主入口（稳定版，可直接上传GitHub）"""
    # 1. 加载模型
    model = load_model_background()
    
    # 2. 创建GUI并启动
    root = tk.Tk()
    app = MTDETRApp(root, model)
    
    # 退出清理逻辑
    def on_closing():
        try:
            app.stop_predict()
            root.destroy()
        except Exception as e:
            print(f"⚠️ 退出时清理资源失败：{str(e)}")
            root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()
    
    # 3. 最终清理
    if app.predictor:
        app.predictor.stop()
    app.preview_panel.clear()

if __name__ == "__main__":
    main()