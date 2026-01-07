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
    """视频预测器：GUI循环播放 + 后台单次预测 + 正常保存"""
    def __init__(self, model, preview_panel, logger, video_player):
        super().__init__(model, preview_panel, logger)
        self.video_player = video_player
        self.out = None
        self.predict_thread = None
        self.current_frame = 0  # 当前推理帧数
        self.total_frames = 0   # 视频总帧数
        # 新增：独立的预测用视频读取器（和GUI播放器解耦）
        self.pred_cap = None

    def start(self, video_path, output_dir):
        if not self.model:
            self.logger("❌ 模型未加载")
            return
        
        if not os.path.exists(video_path):
            self.logger(f"❌ 视频不存在：{video_path}")
            return
        
        # 1. GUI播放器：循环播放原视频（不受预测影响）
        if not self.video_player.load_video(video_path):
            return
        self.video_player.allow_loop = True  # GUI强制循环
        self.video_player.start_play()
        self.logger("🎨 GUI已开始循环播放原视频")
        
        # 2. 初始化预测用视频读取器（后台单次读取）
        self.pred_cap = cv2.VideoCapture(video_path)
        if not self.pred_cap or not self.pred_cap.isOpened():
            self.logger("❌ 预测用视频读取器初始化失败")
            return
        self.total_frames = int(self.pred_cap.get(cv2.CAP_PROP_FRAME_COUNT)) if self.pred_cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0 else 0
        self.current_frame = 0
        fps = int(self.pred_cap.get(cv2.CAP_PROP_FPS)) if self.pred_cap.get(cv2.CAP_PROP_FPS) > 0 else 30
        width = int(self.pred_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.pred_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # 3. 初始化视频写入器（保存正常非循环视频）
        os.makedirs(output_dir, exist_ok=True)
        save_name = f"pred_{os.path.basename(video_path)}"
        self.result_path = os.path.join(output_dir, save_name)
        
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.out = cv2.VideoWriter(self.result_path, fourcc, fps, (width, height))
        if not self.out.isOpened():
            self.logger("❌ 视频写入器初始化失败")
            return
        
        # 4. 启动后台预测线程（单次完整推理）
        with self.lock:
            self.is_running = True
        self.predict_thread = threading.Thread(target=self._predict_loop, daemon=True)
        self.predict_thread.start()
        self.logger(f"🎬 开始后台视频预测：{video_path} | 总帧数：{self.total_frames} | 帧率：{fps}fps")

    def _predict_loop(self):
        """后台预测循环：单次完整推理（非循环）"""
        while True:
            # 检查停止信号 或 预测完成
            with self.lock:
                if not self.is_running or self.current_frame >= self.total_frames:
                    break
            
            # 后台读取原视频帧（单次，非循环）
            ret, frame = self.pred_cap.read()
            if not ret:
                break
            
            try:
                # 执行推理
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.model.predict(
                    source=frame_rgb,
                    device=self._get_device(),
                    imgsz=Config.IMGSZ,
                    conf=Config.CONF_THRESHOLD,
                    verbose=False,
                    stream=False
                )
                
                pred_frame = results[0].plot()
                # 更新GUI右侧预览（无需和左侧同步）
                self._safe_update_preview_frame(pred_frame, is_original=False)
                
                # 写入预测帧（生成正常非循环视频）
                if self.out and self.out.isOpened():
                    self.out.write(cv2.cvtColor(pred_frame, cv2.COLOR_RGB2BGR))
                
                # 更新进度
                self.current_frame += 1
                if self.current_frame % 50 == 0:
                    self.logger(f"📊 预测进度：{self.current_frame}/{self.total_frames} 帧")
            except Exception as e:
                self.logger(f"⚠️ 帧{self.current_frame}出错：{str(e)}")
                self.current_frame += 1

        # 预测完成：保存文件，GUI继续循环播放
        self.logger(f"✅ 后台视频预测完成！总处理帧数：{self.current_frame}")
        self.stop()

    def stop(self):
        """停止预测：同时结束GUI播放 + 保存预测文件"""
        super().stop()
        
        # 1. 停止GUI循环播放
        if self.video_player:
            self.video_player.stop()
            self.logger("🛑 GUI视频播放已停止")
        
        # 2. 关闭预测用视频读取器
        if self.pred_cap:
            try:
                self.pred_cap.release()
            except Exception as e:
                self.logger(f"⚠️ 释放预测读取器失败：{str(e)}")
            self.pred_cap = None
        
        # 3. 关闭视频写入器（关键：生成正常视频）
        if self.out:
            try:
                self.out.release()
                self.logger(f"💾 预测视频已保存：{self.result_path}（非循环）")
            except Exception as e:
                self.logger(f"⚠️ 释放写入器失败：{str(e)}")
            self.out = None
        
        # 重置计数
        self.current_frame = 0
        self.total_frames = 0
        self.logger("🛑 视频预测已完全停止")

# ===================== 摄像头预测器 =====================
class CameraPredictor(BasePredictor):
    """摄像头预测器（修改：移除Q键 + 保存到runs/camera）"""
    def __init__(self, model, preview_panel, logger):
        super().__init__(model, preview_panel, logger)
        self.out = None  # 新增：视频写入器
        self.predict_thread = None  # 新增：预测线程

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
        
        # 新增：初始化视频写入器（保存到Config.CAMERA_SAVE_ROOT）
        time.sleep(0.1)  # 等待摄像头参数生效
        fps = int(self.cap.get(cv2.CAP_PROP_FPS)) or 30
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # 创建保存目录（从配置读取）
        os.makedirs(Config.CAMERA_SAVE_ROOT, exist_ok=True)
        save_name = f"camera_pred_{int(time.time())}.mp4"
        self.result_path = os.path.join(Config.CAMERA_SAVE_ROOT, save_name)
        
        # 初始化视频写入器
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.out = cv2.VideoWriter(self.result_path, fourcc, fps, (width, height))
        if not self.out.isOpened():
            self.logger("⚠️ mp4v编码器失败，尝试XVID格式...")
            save_name = f"camera_pred_{int(time.time())}.avi"
            self.result_path = os.path.join(Config.CAMERA_SAVE_ROOT, save_name)
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            self.out = cv2.VideoWriter(self.result_path, fourcc, fps, (width, height))
        
        self.logger(f"📹 摄像头预测已启动（点击停止按钮结束）")
        self.logger(f"💾 摄像头视频将保存至：{self.result_path}")
        
        # 新增：启动线程执行预测（避免阻塞GUI）
        self.predict_thread = threading.Thread(target=self._predict_loop, daemon=True)
        self.predict_thread.start()

    def _predict_loop(self):
        """摄像头预测循环（移除Q键，仅通过is_running控制停止）"""
        while True:
            # 仅检查运行状态，移除Q键逻辑
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
                
                # 新增：写入预测视频帧
                if self.out and self.out.isOpened():
                    self.out.write(pred_frame)
                    
            except Exception as e:
                self.logger(f"⚠️ 摄像头帧出错：{str(e)}")
        
        # 释放视频写入器
        if self.out:
            self.out.release()
            self.logger(f"✅ 摄像头预测视频已保存：{self.result_path}")
        
        self.stop()
        cv2.destroyAllWindows()
        self.logger("✅ 摄像头预测已停止")

    def stop(self):
        """重写stop方法：确保释放视频写入器"""
        super().stop()
        
        # 等待线程结束
        if self.predict_thread and self.predict_thread.is_alive():
            self.predict_thread.join(timeout=1)
        
        # 释放视频写入器
        if self.out:
            try:
                self.out.release()
            except Exception as e:
                self.logger(f"⚠️ 释放摄像头视频写入器失败：{str(e)}")
            self.out = None