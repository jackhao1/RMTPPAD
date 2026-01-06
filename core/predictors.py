#!/usr/bin/env python3
import os
import time
import cv2
import torch
import threading
from core.config import Config
from PIL import Image, ImageTk

class BasePredictor:
    """预测器基类（稳定版，无stop_event）"""
    def __init__(self, model, preview_panel, logger):
        self.model = model
        self.preview_panel = preview_panel
        self.logger = logger
        self.is_running = False
        self.cap = None
        self.result_path = ""
        self.lock = threading.Lock()

    def start(self, *args, **kwargs):
        raise NotImplementedError("子类必须实现start方法")

    def stop(self):
        """通用停止方法（稳定版）"""
        with self.lock:
            self.is_running = False
        
        if self.cap and isinstance(self.cap, cv2.VideoCapture):
            try:
                self.cap.release()
            except Exception as e:
                self.logger(f"⚠️ 释放资源失败：{str(e)}")
            self.cap = None
        
        self.logger("🛑 预测已停止")

    def _get_device(self):
        """获取推理设备（优先GPU）"""
        return 0 if torch.cuda.is_available() else "cpu"

    def _safe_update_preview_frame(self, frame, is_original):
        """安全更新预览帧（稳定版，无is_valid检查）"""
        try:
            # 颜色空间转换
            if len(frame.shape) == 2:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            else:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 调整尺寸并更新UI
            target_label = self.preview_panel.left_label if is_original else self.preview_panel.right_label
            img = Image.fromarray(frame_rgb)
            img = self.preview_panel._resize_img_to_label(img, target_label)
            img_tk = ImageTk.PhotoImage(img)
            
            # 主线程更新（保存引用防止图像消失）
            def update_ui():
                if is_original:
                    self.preview_panel.left_img = img_tk
                    target_label.config(image=img_tk, text="")
                else:
                    self.preview_panel.right_img = img_tk
                    target_label.config(image=img_tk, text="")
            
            target_label.after(0, update_ui)
        except Exception as e:
            self.logger(f"⚠️ 预览更新失败：{str(e)}")

# ===================== 图片预测器 =====================
class ImagePredictor(BasePredictor):
    """图片预测器（稳定版）"""
    def start(self, image_path):
        if not self.model:
            self.logger("❌ 模型未加载")
            return
        
        if not os.path.exists(image_path):
            self.logger(f"❌ 图片不存在：{image_path}")
            return
        
        try:
            # 读取并显示原始图片
            img = cv2.imread(image_path)
            self._safe_update_preview_frame(img, is_original=True)
            
            # 执行预测
            results = self.model.predict(
                source=image_path,
                save=False,
                device=self._get_device(),
                imgsz=Config.IMGSZ,
                conf=Config.CONF_THRESHOLD
            )
            
            # 保存并显示预测结果
            pred_img = results[0].plot()
            os.makedirs(Config.IMAGE_SAVE_ROOT, exist_ok=True)
            save_name = f"pred_{os.path.basename(image_path)}"
            self.result_path = os.path.join(Config.IMAGE_SAVE_ROOT, save_name)
            cv2.imwrite(self.result_path, cv2.cvtColor(pred_img, cv2.COLOR_RGB2BGR))
            
            self._safe_update_preview_frame(pred_img, is_original=False)
            self.logger(f"✅ 图片预测完成：{self.result_path}")
        except Exception as e:
            self.logger(f"❌ 图片预测失败：{str(e)}")

# ===================== 视频预测器 =====================
class VideoPredictor(BasePredictor):
    """视频预测器（稳定版，无帧数限制）"""
    def __init__(self, model, preview_panel, logger, video_player):
        super().__init__(model, preview_panel, logger)
        self.video_player = video_player
        self.out = None
        self.predict_thread = None

    def start(self, video_path, output_dir):
        if not self.model:
            self.logger("❌ 模型未加载")
            return
        
        if not os.path.exists(video_path):
            self.logger(f"❌ 视频不存在：{video_path}")
            return
        
        # 初始化播放器
        if not self.video_player.load_video(video_path):
            return
        self.video_player.start_play()
        
        # 初始化视频保存
        fps = self.video_player.fps
        width = int(self.video_player.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.video_player.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        os.makedirs(output_dir, exist_ok=True)
        save_name = f"pred_{os.path.basename(video_path)}"
        self.result_path = os.path.join(output_dir, save_name)
        
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.out = cv2.VideoWriter(self.result_path, fourcc, fps, (width, height))
        
        # 启动推理线程
        with self.lock:
            self.is_running = True
        self.predict_thread = threading.Thread(target=self._predict_loop, daemon=True)
        self.predict_thread.start()
        self.logger(f"🎬 开始视频预测：{video_path}")

    def _predict_loop(self):
        """视频推理循环（稳定版，无帧数限制）"""
        frame_count = 0
        while True:
            # 仅检查运行状态
            with self.lock:
                if not self.is_running:
                    break
            
            # 获取帧
            frame = self.video_player.get_latest_frame()
            if frame is None:
                time.sleep(0.001)
                continue
            
            try:
                # 执行推理并更新预览
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.model.predict(
                    source=frame_rgb,
                    device=self._get_device(),
                    imgsz=Config.IMGSZ,
                    conf=Config.CONF_THRESHOLD,
                    verbose=False
                )
                
                pred_frame = results[0].plot()
                self._safe_update_preview_frame(pred_frame, is_original=False)
                
                # 保存视频帧
                if self.out and self.out.isOpened():
                    self.out.write(cv2.cvtColor(pred_frame, cv2.COLOR_RGB2BGR))
                
                frame_count += 1
                if frame_count % 50 == 0:
                    self.logger(f"📊 进度：{frame_count}帧")
            except Exception as e:
                self.logger(f"⚠️ 帧{frame_count}出错：{str(e)}")

    def stop(self):
        """停止视频预测（稳定版）"""
        super().stop()
        
        # 释放视频写入器
        if self.out:
            try:
                self.out.release()
            except Exception as e:
                self.logger(f"⚠️ 释放写入器失败：{str(e)}")
            self.out = None
        
        # 停止播放器
        if self.video_player:
            self.video_player.stop()
        
        self.logger("✅ 视频预测已停止")

# ===================== 摄像头预测器 =====================
class CameraPredictor(BasePredictor):
    """摄像头预测器（稳定版）"""
    def start(self, camera_id=0):
        if not self.model:
            self.logger("❌ 模型未加载")
            return
        
        # 初始化摄像头
        with self.lock:
            self.is_running = True
        self.cap = cv2.VideoCapture(int(camera_id))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        self.logger("📹 摄像头预测已启动（按Q退出）")
        
        while True:
            # 仅检查运行状态
            with self.lock:
                if not self.is_running:
                    break
            
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            
            try:
                # 实时预测并更新预览
                self._safe_update_preview_frame(frame, is_original=True)
                results = self.model.predict(
                    source=frame,
                    device=self._get_device(),
                    imgsz=Config.IMGSZ,
                    conf=Config.CONF_THRESHOLD,
                    verbose=False
                )
                
                pred_frame = results[0].plot()
                self._safe_update_preview_frame(pred_frame, is_original=False)
                
                # 按Q退出
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            except Exception as e:
                self.logger(f"⚠️ 摄像头帧出错：{str(e)}")
        
        self.stop()
        cv2.destroyAllWindows()
        self.logger("✅ 摄像头预测已停止")