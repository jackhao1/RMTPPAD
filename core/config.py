#!/usr/bin/env python3
import os

# 获取项目根目录
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class Config:
    """全局配置（拆分保存路径）"""
    # 模块路径
    MTDETR_PATH = os.path.join(BASE_DIR, "ultralytics", "models", "mtdetr")
    ULTRALYTICS_ROOT = os.path.join(BASE_DIR, "ultralytics")
    
    # 模型权重固定路径
    MODEL_WEIGHT_PATH = os.path.join(BASE_DIR, "best.pt")
    
    # 拆分保存路径（视频/图片/摄像头）
    SAVE_ROOT = os.path.join(BASE_DIR, "runs")
    IMAGE_SAVE_ROOT = os.path.join(SAVE_ROOT, "images")       # 图片结果
    VIDEO_SAVE_ROOT = os.path.join(SAVE_ROOT, "videos")       # 视频结果
    CAMERA_SAVE_ROOT = os.path.join(SAVE_ROOT, "camera")      # 摄像头结果
    
    # 预测参数
    IMGSZ = 640
    CONF_THRESHOLD = 0.25
    
    @classmethod
    def init_dirs(cls):
        """初始化所有保存目录"""
        os.makedirs(cls.IMAGE_SAVE_ROOT, exist_ok=True)
        os.makedirs(cls.VIDEO_SAVE_ROOT, exist_ok=True)
        os.makedirs(cls.CAMERA_SAVE_ROOT, exist_ok=True)
    
    @classmethod
    def create_symlinks(cls):
        """创建软链接到原D盘路径"""
        original_root = r"D:\RMTDRIVE\ultralytics"
        original_runs = os.path.join(original_root, "runs")
        
        # 创建总runs软链接
        if not os.path.exists(cls.SAVE_ROOT):
            os.symlink(original_runs, cls.SAVE_ROOT)
        
        # 创建模型软链接
        if not os.path.exists(cls.MODEL_WEIGHT_PATH):
            original_best_pt = os.path.join(original_root, "best.pt")
            os.symlink(original_best_pt, cls.MODEL_WEIGHT_PATH)