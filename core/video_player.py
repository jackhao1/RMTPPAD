#!/usr/bin/env python3
import cv2
import threading
import time
import queue
from PIL import Image, ImageTk

class IndependentVideoPlayer:
    """独立原始视频播放器（稳定版，无循环导入）"""
    def __init__(self, preview_panel, logger):
        self.preview_panel = preview_panel
        self.logger = logger
        self.cap = None
        self.is_playing = False
        self.is_paused = False
        self.play_thread = None
        self.frame_queue = queue.Queue(maxsize=5)
        self.fps = 30
        self.frame_delay = 33
        self.lock = threading.Lock()

    def load_video(self, video_path):
        """加载视频并初始化参数"""
        self.stop()
        
        with self.lock:
            self.cap = cv2.VideoCapture(video_path)
            if not self.cap or not self.cap.isOpened():
                self.logger(f"❌ 无法打开视频：{video_path}")
                return False
            
            # 获取视频参数
            self.fps = int(self.cap.get(cv2.CAP_PROP_FPS)) if self.cap.get(cv2.CAP_PROP_FPS) > 0 else 30
            self.frame_delay = int(1000 / self.fps)
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) if self.cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0 else 0
            self.logger(f"📽️ 加载视频成功：帧率={self.fps}fps，总帧数={self.total_frames}")
        
        # 清空帧队列
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                pass
        
        return True

    def start_play(self):
        """启动播放线程"""
        with self.lock:
            if not self.cap or self.is_playing:
                return
            self.is_playing = True
            self.is_paused = False
        
        self.play_thread = threading.Thread(target=self._play_loop, daemon=True)
        self.play_thread.start()

    def _play_loop(self):
        """播放循环（稳定版，保留循环播放）"""
        while True:
            with self.lock:
                if not self.is_playing:
                    break
                if self.is_paused:
                    time.sleep(0.01)
                    continue
            
            start_time = time.time()
            
            # 读取帧
            ret, frame = None, None
            with self.lock:
                if self.cap and self.cap.isOpened():
                    ret, frame = self.cap.read()
            
            # 循环播放
            if not ret:
                with self.lock:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                time.sleep(0.01)
                continue
            
            # 放入帧队列
            try:
                if self.frame_queue.full():
                    self.frame_queue.get_nowait()
                self.frame_queue.put(frame.copy(), timeout=0.01)
            except queue.Full:
                pass
            
            # 更新预览
            self._safe_update_left_preview(frame)
    
            # 控制播放速度
            elapsed = int((time.time() - start_time) * 1000)
            sleep_time = max(0, self.frame_delay - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time / 1000)

    def _safe_update_left_preview(self, frame):
        """更新左侧预览"""
        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            img = self.preview_panel._resize_img_to_label(img, self.preview_panel.left_label)
            img_tk = ImageTk.PhotoImage(img)
            
            def update_ui():
                self.preview_panel.left_img = img_tk
                self.preview_panel.left_label.config(image=img_tk, text="")
            
            if self.preview_panel and self.preview_panel.left_label:
                self.preview_panel.left_label.after(0, update_ui)
        except Exception as e:
            self.logger(f"⚠️ 左侧预览更新失败：{str(e)}")

    def get_latest_frame(self):
        """获取最新帧"""
        try:
            return self.frame_queue.get_nowait()
        except queue.Empty:
            return None

    def pause(self):
        """暂停播放"""
        with self.lock:
            self.is_paused = True

    def resume(self):
        """恢复播放"""
        with self.lock:
            self.is_paused = False

    def stop(self):
        """停止播放"""
        with self.lock:
            self.is_playing = False
            self.is_paused = False
        
        if self.play_thread and self.play_thread.is_alive():
            self.play_thread.join(timeout=2)
        
        with self.lock:
            if self.cap:
                try:
                    self.cap.release()
                except Exception as e:
                    self.logger(f"⚠️ 释放视频资源失败：{str(e)}")
                self.cap = None
        
        # 清空队列
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                pass
        
        # 清空预览
        if self.preview_panel and self.preview_panel.left_label:
            def clear_ui():
                self.preview_panel.left_label.config(text="暂无原始内容", image="")
                self.preview_panel.left_img = None
            self.preview_panel.left_label.after(0, clear_ui)
        
        self.logger("🛑 视频播放已停止")