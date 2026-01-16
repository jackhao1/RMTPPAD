#!/usr/bin/env python3
import os
import sys

def get_base_dir():
    """
    获取基准目录（无兜底：仅返回明确的运行环境目录，不做异常兜底）
    1. EXE打包运行：返回EXE所在目录
    2. 普通Python运行：返回项目根目录
    """
    if getattr(sys, 'frozen', False):
        # EXE打包环境：直接返回EXE所在目录（无异常兜底，明确依赖环境配置）
        exe_dir = os.path.dirname(os.path.abspath(sys.executable))
        return os.path.abspath(exe_dir)
    else:
        # 普通Python环境：直接返回项目根目录（无异常兜底，明确目录结构）
        current_file = os.path.abspath(__file__)
        project_parent = os.path.dirname(current_file)
        project_root = os.path.dirname(project_parent)
        return os.path.abspath(project_root)

# 全局基准目录（无兜底：直接基于明确逻辑构建，不做额外容错）
BASE_DIR = get_base_dir()

# 兼容PyInstaller单文件模式的临时目录（无兜底：直接映射BASE_DIR，不做额外 fallback）
MEIPASS_DIR = getattr(sys, '_MEIPASS', BASE_DIR)

class Config:
    """全局配置（无兜底：仅保留明确配置，移除所有容错和 fallback 逻辑）"""
    # 模块路径：明确基于临时目录构建，无额外兜底，依赖打包/目录结构的正确性
    MTDETR_PATH = os.path.join(MEIPASS_DIR, "ultralytics", "models", "mtdetr")
    ULTRALYTICS_ROOT = os.path.join(MEIPASS_DIR, "ultralytics")
    
    # 模型权重路径（无兜底：仅保留最高优先级，移除「找不到则回退」的兜底逻辑）
    # 明确依赖：PyInstaller临时目录/基准目录下的 ultralytics/best.pt 必须存在
    MODEL_WEIGHT_PATH = os.path.join(MEIPASS_DIR, "ultralytics", "best.pt")
    
    # 结果保存路径：明确基于基准目录构建，无兜底，直接创建 runs 及其子目录
    SAVE_ROOT = os.path.join(BASE_DIR, "runs")
    IMAGE_SAVE_ROOT = os.path.join(SAVE_ROOT, "images")
    VIDEO_SAVE_ROOT = os.path.join(SAVE_ROOT, "videos")
    CAMERA_SAVE_ROOT = os.path.join(SAVE_ROOT, "camera")
    
    # 预测参数：明确配置，无动态兜底
    IMGSZ = 640
    CONF_THRESHOLD = 0.4
    
    @classmethod
    def init_dirs(cls):
        """初始化所有保存目录（无兜底：移除所有 try/except 异常兜底，直接创建）"""
        dirs_to_create = [
            cls.SAVE_ROOT,
            cls.IMAGE_SAVE_ROOT,
            cls.VIDEO_SAVE_ROOT,
            cls.CAMERA_SAVE_ROOT
        ]
        
        # 无异常兜底：直接创建目录，依赖目录权限和路径的正确性
        os.makedirs(cls.SAVE_ROOT, exist_ok=True)
        for dir_path in dirs_to_create:
            os.makedirs(dir_path, exist_ok=True)
        
        return None
    
    @classmethod
    def check_model_exists(cls):
        """检查模型文件是否存在（无兜底：仅保留明确验证，移除冗余容错）"""
        # 无兜底：直接验证路径有效性和文件类型，不做额外空值容错（依赖配置的正确性）
        return os.path.exists(cls.MODEL_WEIGHT_PATH) and os.path.isfile(cls.MODEL_WEIGHT_PATH)