#!/usr/bin/env python3
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import sys
import threading
import time

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

try:
    from core.config import Config
    from core.predictors import ImagePredictor, VideoPredictor, CameraPredictor
    from core.video_player import IndependentVideoPlayer
    from gui.preview_panel import PreviewPanel
except ImportError as e:
    print(f"❌ 核心模块导入失败：{e}")
    messagebox.showerror("导入错误", f"无法导入核心模块：\n{str(e)}\n请检查模块路径是否正确")
    sys.exit(1)

sys.path.insert(0, Config.MTDETR_PATH)
sys.path.insert(0, Config.ULTRALYTICS_ROOT)
if "PYTHONPATH" in os.environ:
    os.environ["PYTHONPATH"] = f"{Config.MTDETR_PATH};{Config.ULTRALYTICS_ROOT};{os.environ['PYTHONPATH']}"
else:
    os.environ["PYTHONPATH"] = f"{Config.MTDETR_PATH};{Config.ULTRALYTICS_ROOT}"

class MTDETRApp:
    """
    主应用类：MTDETR预测工具GUI，强制要求先选择/输入保存文件夹，再启动预测
    核心功能：提供图片/视频/摄像头三种预测入口，按需创建专属保存子目录，避免冗余路径
    核心修改：移除主动创建目录逻辑，仅依赖预测器创建目录，杜绝额外runs目录
    """
    def __init__(self, root, model):
        self.root = root
        self.model = model
        
        # ========== 修改1：修改窗口标题（左上角文字） ==========
        self.root.title("RMTPPAD预测工具")  # 可替换为你想要的任意标题
        
        # ========== 修改2：设置窗口图标（左上角图标，支持.ico格式） ==========
        try:
            # 1. 若有.ico图标文件，放在程序同级目录，替换下面的"app_icon.ico"为你的图标文件名
            # 2. 若无图标文件，注释掉下面这行即可，不影响程序运行
            self.root.iconbitmap("app_icon.ico")  
        except Exception as e:
            self.logger(f"⚠️ 窗口图标加载失败（若无.ico文件可忽略此提示）：{str(e)}")
        
        self.root.geometry("1200x800")
        self.root.resizable(True, True)
        
        self.predictor = None
        self.custom_save_root = None
        
        self._create_ui()
        self.video_player = IndependentVideoPlayer(self.preview_panel, self.logger)
        self._update_save_dir_log()
        self.logger("💡 操作步骤：1.选保存文件夹 → 2.选数据源 → 3.调整置信度 → 4.预测 → 查看实时预览")
        self._on_predict_type_changed(None)

    def _create_ui(self):
        """构建应用GUI界面，包含预测配置、保存目录配置、控制按钮和预览区域"""

        frame_source = ttk.LabelFrame(self.root, text="预测配置")
        frame_source.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(frame_source, text="预测类型：").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.combo_predict_type = ttk.Combobox(frame_source, values=["图片", "视频", "摄像头"], width=10, state="readonly")
        self.combo_predict_type.current(0)
        self.combo_predict_type.grid(row=0, column=1, padx=5, pady=5)
        self.combo_predict_type.bind("<<ComboboxSelected>>", self._on_predict_type_changed)
        
        ttk.Label(frame_source, text="数据源：").grid(row=0, column=2, padx=5, pady=5, sticky=tk.W)
        self.entry_source = ttk.Entry(frame_source, width=40)
        self.entry_source.grid(row=0, column=3, padx=5, pady=5)
        
        self.btn_select_image = ttk.Button(frame_source, text="选择图片", command=self.select_image)
        self.btn_select_video = ttk.Button(frame_source, text="选择视频", command=self.select_video)
        
        ttk.Label(frame_source, text="摄像头ID：").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        self.entry_camera_id = ttk.Entry(frame_source, width=10)
        self.entry_camera_id.insert(0, "0")
        self.entry_camera_id.grid(row=1, column=1, padx=5, pady=5)

        ttk.Label(frame_source, text="检测置信度：").grid(row=1, column=2, padx=5, pady=5, sticky=tk.W)
        self.entry_conf = ttk.Entry(frame_source, width=10)
        self.entry_conf.insert(0, str(Config.CONF_THRESHOLD))
        self.entry_conf.grid(row=1, column=3, padx=5, pady=5)
        ttk.Button(frame_source, text="应用置信度", command=self.apply_conf_threshold).grid(row=1, column=4, padx=5, pady=5)
        ttk.Label(frame_source, text="（范围：0.0~1.0，值越小检测越灵敏）").grid(row=1, column=5, padx=5, pady=5, sticky=tk.W)

        # 保存目录配置区域
        frame_save = ttk.LabelFrame(self.root, text="结果保存配置（必填）")
        frame_save.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(frame_save, text="保存根目录：").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.entry_save_root = ttk.Entry(frame_save, width=50)
        self.entry_save_root.insert(0, "")
        self.entry_save_root.grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(frame_save, text="选择保存根目录", command=self.select_save_dir).grid(row=0, column=2, padx=5, pady=5)
        
        # ========== 修改3：删除指定提示文本“（必须先设置此目录，否则无法启动预测）” ==========
        # 原该行代码已直接删除，不再显示该提示

        # 控制按钮区域
        frame_ctrl = ttk.Frame(self.root)
        frame_ctrl.pack(fill=tk.X, padx=10, pady=5)
        ttk.Button(frame_ctrl, text="启动预测", command=self.start_predict).grid(row=0, column=0, padx=5, pady=5)
        ttk.Button(frame_ctrl, text="停止预测", command=self.stop_predict).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(frame_ctrl, text="清空预览", command=self.clear_preview).grid(row=0, column=2, padx=5, pady=5)

        # 分栏预览区域
        self.preview_panel = PreviewPanel(self.root)
    
    def select_save_dir(self):
        """弹出目录选择对话框，记录用户选中的保存根目录并更新输入框"""
        selected_dir = filedialog.askdirectory(title="选择保存根目录")
        
        if selected_dir:
            self.custom_save_root = os.path.abspath(selected_dir)
            self.entry_save_root.delete(0, tk.END)
            self.entry_save_root.insert(0, self.custom_save_root)
            self.logger(f"📂 已选择保存根目录：{self.custom_save_root}")
            self.logger(f"ℹ️ 预测时将自动创建对应子目录（图片→images，视频→videos，摄像头→camera）")
            self._update_save_dir_log()
    
    def _update_save_dir_log(self):
        """更新并打印当前生效的保存目录状态，未设置时给出提示（移除冗余警告文本）"""
        current_save_root = self.custom_save_root or self.entry_save_root.get().strip()
        if current_save_root:
            self.logger("📢 预测工具当前生效配置")
            self.logger(f"🖼️ 保存根路径：{current_save_root}（子目录将在预测时按需创建）")
        else:
            self.logger("📢 预测工具当前生效配置")
            # ========== 修改4：删除日志中的“⚠️ 请先选择或输入，否则无法启动预测”提示 ==========
            self.logger(f"🖼️ 暂未设置保存根目录")
    
    def _on_predict_type_changed(self, event):
        """
        下拉框切换事件处理：
        1. 清空数据源输入框
        2. 停止当前运行的预测
        3. 根据预测类型显隐对应的数据选择按钮
        """
        self.entry_source.delete(0, tk.END)
        self.stop_predict()
        
        current_type = self.combo_predict_type.get() or "图片"
        self.btn_select_image.grid_remove()
        self.btn_select_video.grid_remove()
        
        if current_type == "图片":
            self.btn_select_image.grid(row=0, column=4, padx=5, pady=5)
            self.entry_source.config(state="normal")
        elif current_type == "视频":
            self.btn_select_video.grid(row=0, column=4, padx=5, pady=5)
            self.entry_source.config(state="normal")
        elif current_type == "摄像头":
            self.entry_source.config(state="disabled")

    def apply_conf_threshold(self):
        """应用置信度阈值，支持预测过程中动态调整，同时做合法性校验"""
        try:
            new_conf = float(self.entry_conf.get().strip())
            if not (0.0 <= new_conf <= 1.0):
                raise ValueError("置信度需在0.0~1.0之间")
            
            if self.predictor and hasattr(self.predictor, 'set_conf_threshold'):
                self.predictor.set_conf_threshold(new_conf)
                self.logger(f"ℹ️ 置信度已动态调整为{new_conf}（当前预测生效）")
            else:
                Config.CONF_THRESHOLD = new_conf
                self.entry_conf.delete(0, tk.END)
                self.entry_conf.insert(0, str(new_conf))
                self.logger(f"ℹ️ 置信度已设为{new_conf}（预测启动后生效）")
        except ValueError as e:
            self.logger(f"❌ 置信度设置失败：{str(e)}")
            messagebox.showwarning("警告", f"无效的置信度值：\n{str(e)}")

    def select_image(self):
        """弹出图片选择对话框，选择预测用图片并更新数据源输入框"""
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
        """弹出视频选择对话框，选择预测用视频并更新数据源输入框"""
        path = filedialog.askopenfilename(
            title="选择预测视频",
            filetypes=[("视频文件", "*.mp4;*.avi;*.mov;*.mkv"), ("所有文件", "*.*")]
        )
        if path:
            self.entry_source.delete(0, tk.END)
            self.entry_source.insert(0, path)
            self.logger(f"📁 已选择视频：{path}")
    
    def _video_predict_complete_callback(self, pred_mp4_path):
        """
        视频预测完成回调函数：弹出提示弹窗
        :param pred_mp4_path: 视频预测结果的实际保存路径
        """
        # 必须使用after方法，确保在GUI主线程中弹出弹窗（避免线程安全问题）
        self.root.after(0, lambda: messagebox.showinfo(
            "预测完成",
            f"🎉 视频预测已全部完成！\n\n预测结果已保存至：\n{pred_mp4_path}\n\n请前往查看。"
        ))
        self.logger(f"✅ 视频预测完成，结果保存至：{pred_mp4_path}")

    def start_predict(self):
        """
        启动预测核心方法：
        核心修改：1. 移除主动创建根目录逻辑  2. 确保仅使用用户自定义目录  3. 优化视频回调路径传递
        """
        if self.predictor and hasattr(self.predictor, 'is_running') and self.predictor.is_running:
            self.logger("⚠️ 预测已在进行中，请勿重复启动")
            return
        
        predict_type = self.combo_predict_type.get()
        source = self.entry_source.get()
        current_save_root = self.custom_save_root or self.entry_save_root.get().strip()
        
        if not current_save_root:
            err_msg = "❌ 请先通过「选择保存根目录」按钮选择文件夹，或手动输入保存目录！"
            self.logger(err_msg)
            messagebox.showwarning("操作禁止", err_msg)
            return
        
        if predict_type in ["图片", "视频"] and not os.path.exists(source):
            err_msg = f"❌ {predict_type}不存在：{source}"
            self.logger(err_msg)
            messagebox.showwarning("警告", err_msg)
            return
        
        # ========== 核心修改1：删除原有的 os.makedirs(current_save_root, exist_ok=True) ==========
        
        sub_dir = "images" if predict_type == "图片" else "videos" if predict_type == "视频" else "camera"
        result_save_path = os.path.join(current_save_root, sub_dir)
        
        if predict_type == "图片":
            self.predictor = ImagePredictor(self.model, self.preview_panel, self.logger)
            if hasattr(self.predictor, 'set_save_dir'):
                self.predictor.set_save_dir(current_save_root)
        elif predict_type == "视频":
            self.predictor = VideoPredictor(self.model, self.preview_panel, self.logger, self.video_player)
            if hasattr(self.predictor, 'set_save_dir'):
                self.predictor.set_save_dir(current_save_root)
            # ========== 核心修改2：优化回调函数，直接绑定预测器的实际保存路径 ==========
            if hasattr(self.predictor, 'set_complete_callback'):
                def callback():
                    if hasattr(self.predictor, 'pred_mp4_path') and self.predictor.pred_mp4_path:
                        self._video_predict_complete_callback(self.predictor.pred_mp4_path)
                self.predictor.set_complete_callback(callback)
        elif predict_type == "摄像头":
            self.predictor = CameraPredictor(self.model, self.preview_panel, self.logger)
            if hasattr(self.predictor, 'set_save_dir'):
                self.predictor.set_save_dir(current_save_root)
        
        try:
            new_conf = float(self.entry_conf.get().strip())
            if 0.0 <= new_conf <= 1.0:
                Config.CONF_THRESHOLD = new_conf
                if hasattr(self.predictor, 'set_conf_threshold'):
                    self.predictor.set_conf_threshold(new_conf)
        except Exception as e:
            self.logger(f"ℹ️ 置信度使用默认值{Config.CONF_THRESHOLD}：{str(e)}")
        
        if predict_type == "图片":
            threading.Thread(target=self.predictor.start, args=(source,), daemon=True).start()
        elif predict_type == "视频":
            threading.Thread(target=self.predictor.start, args=(source,), daemon=True).start()
        elif predict_type == "摄像头":
            threading.Thread(target=self.predictor.start, args=(self.entry_camera_id.get(),), daemon=True).start()
        
        self.logger(f"🚀 开始{predict_type}预测（置信度：{Config.CONF_THRESHOLD}，保存根目录：{current_save_root}）")
        self.logger(f"ℹ️ 正在创建{predict_type}专属子目录：{result_save_path}")

    def stop_predict(self):
        """停止当前运行的预测，释放相关资源并更新日志状态"""
        try:
            if self.predictor:
                self.predictor.stop()
                self.predictor = None
                self.logger("🛑 预测已停止，资源已释放")
        except Exception as e:
            self.logger(f"❌ 停止预测时出错：{str(e)}")
            messagebox.showwarning("警告", f"停止预测时出现异常：\n{str(e)}")

    def clear_preview(self):
        """清空预览区域和数据源输入框，同时停止当前预测"""
        self.stop_predict()
        self.preview_panel.clear()
        self.entry_source.delete(0, tk.END)
        self.logger("🗑️ 已清空预览区域和GUI数据源路径")

    def logger(self, content):
        """带时间戳的日志输出方法，便于调试和运行状态追溯"""
        log_msg = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {content}"
        print(log_msg)

def load_model_background():
    """后台加载MTDETR模型，不提前创建任何目录，加载失败时弹出错误提示并退出"""
    print("📌 正在后台加载模型...")
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
    """程序主入口：加载模型、初始化GUI、配置退出清理逻辑"""
    model = load_model_background()
    root = tk.Tk()
    app = MTDETRApp(root, model)
    
    def on_closing():
        try:
            app.stop_predict()
            app.preview_panel.clear()
            root.destroy()
        except Exception as e:
            print(f"⚠️ 退出时清理资源失败：{str(e)}")
            root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()