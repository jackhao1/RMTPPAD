#!/usr/bin/env python3
import os
import time
import cv2
import torch
import threading
import shutil
from core.config import Config
from PIL import Image, ImageTk

class BasePredictor:
    """预测器基类：定义子目录规范，统一资源释放与预览更新（彻底移除runs依赖）"""
    def __init__(self, model, preview_panel, logger):
        self.model = model
        self.preview_panel = preview_panel
        self.logger = logger
        self.is_running = False
        self.cap = None
        self.result_path = ""
        self.lock = threading.Lock()
        self.save_root = None  # 保存根目录
        self.sub_dir_name = None  # 子类专属子目录名
        self.actual_save_dir = None  # 最终保存目录（根目录+子目录）

    def start(self, *args, **kwargs):
        raise NotImplementedError("子类必须实现start方法")

    def stop(self):
        """通用停止方法：释放视频资源，重置运行状态"""
        with self.lock:
            self.is_running = False
        
        if self.cap and isinstance(self.cap, cv2.VideoCapture):
            try:
                self.cap.release()
            except Exception as e:
                self.logger(f"⚠️ 释放视频资源失败：{str(e)}")
            self.cap = None
        
        self.logger("🛑 预测已停止，资源已释放")

    def _get_device(self):
        """获取推理设备：优先GPU，否则CPU"""
        return 0 if torch.cuda.is_available() else "cpu" 

    def _safe_update_preview_frame(self, frame, is_original):
        """安全更新预览帧：格式转换（BGR→RGB），主线程UI渲染"""
        try:
            # 颜色空间转换
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB) if len(frame.shape) == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 调整尺寸并更新UI
            target_label = self.preview_panel.left_label if is_original else self.preview_panel.right_label
            img = Image.fromarray(frame_rgb)
            img = self.preview_panel._resize_img_to_label(img, target_label)
            img_tk = ImageTk.PhotoImage(img)
            
            # 主线程UI，防止图片丢失（保留引用避免垃圾回收）
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
    
    @property
    def conf_threshold(self):
        """兼容配置：映射到Config置信度阈值"""
        return Config.CONF_THRESHOLD

    def _create_exclusive_sub_dir(self):
        """创建专属子目录：无有效根目录时，使用程序目录predict_results兜底（彻底移除runs依赖）"""
        if not self.save_root or not self.sub_dir_name:
            self.logger("⚠️ 保存目录无效，使用默认兜底目录")
            base_dir = os.path.dirname(os.path.abspath(__file__))
            # ========== 核心修改1：彻底移除 Config.SAVE_ROOT 依赖 ==========
            default_root = os.path.join(base_dir, "predict_results")
            self.actual_save_dir = os.path.join(default_root, self.sub_dir_name)
        else:
            self.actual_save_dir = os.path.join(self.save_root, self.sub_dir_name)
        
        # 创建目录（已存在则忽略）
        os.makedirs(self.actual_save_dir, exist_ok=True)
        self.logger(f"✅ 已创建{self.sub_dir_name}子目录：{self.actual_save_dir}")

# ===================== 图片预测器（仅创建images子目录） =====================
class ImagePredictor(BasePredictor):
    """图片预测器：专属images子目录，完成图片推理与结果保存"""
    def __init__(self, model, preview_panel, logger):
        super().__init__(model, preview_panel, logger)
        self.sub_dir_name = "images"

    def set_save_dir(self, new_save_root):
        """设置保存根目录：验证路径有效性"""
        if new_save_root and os.path.exists(new_save_root):
            self.save_root = new_save_root
            self.logger(f"ℹ️ 图片预测器已配置根目录：{self.save_root}")
        else:
            self.save_root = None
            self.logger(f"⚠️ 无效根目录，将使用默认目录")

    def start(self, image_path):
        """图片预测核心：创建子目录，执行推理，更新预览与保存结果"""
        if not self.model:
            self.logger("❌ 模型未加载，无法预测")
            return
        
        if not os.path.exists(image_path):
            self.logger(f"❌ 图片不存在：{image_path}")
            return
        
        # 创建专属子目录
        self._create_exclusive_sub_dir()
        
        # ========== 核心修改2：删除 Config.init_dirs() 调用 ==========
        try:
            orig_filename = os.path.basename(image_path)
            orig_frame = cv2.imread(image_path)
            if orig_frame is None:
                self.logger(f"❌ 无法读取图片：{image_path}")
                return
            
            # 更新原始图片预览
            self._safe_update_preview_frame(orig_frame, is_original=True)
            
            # 模型推理（强化路径参数，防止回退到runs）
            results = self.model.predict(
                source=image_path,
                save=True,
                project=os.path.dirname(self.actual_save_dir),
                name=os.path.basename(self.actual_save_dir),
                exist_ok=True,
                save_txt=False,
                save_conf=True,
                save_crop=False,
                device=self._get_device(),
                imgsz=Config.IMGSZ,
                conf=Config.CONF_THRESHOLD,
                mask_threshold=[0.4,0.9],
                verbose=False
            )
            
            # 读取并更新预测结果预览
            actual_result_path = os.path.join(self.actual_save_dir, orig_filename)
            model_saved_frame = cv2.imread(actual_result_path) if os.path.exists(actual_result_path) else orig_frame
            if model_saved_frame is None:
                model_saved_frame = orig_frame
                self.logger(f"⚠️ 读取预测图片失败，显示原图")
            
            # 保存结果路径并更新预览
            self.result_path = actual_result_path
            self._safe_update_preview_frame(model_saved_frame, is_original=False)
            self.logger(f"✅ 图片预测完成，结果保存至：{self.result_path}")
            
        except Exception as e:
            self.logger(f"❌ 图片预测失败：{str(e)}")



# ===================== 视频预测器（弹窗后启用独立播放，立即删除frame） =====================
class VideoPredictor(BasePredictor):
    """视频预测器：推理时逐帧预览+弹窗后启用独立循环播放+预测完成立即删除frame"""
    def __init__(self, model, preview_panel, logger, video_player):
        super().__init__(model, preview_panel, logger)
        self.sub_dir_name = "videos"
        self.video_player = video_player
        
        # 新增：右侧独立播放相关属性（弹窗后启用）
        self.right_video_cap = None  # 右侧完整视频捕获对象
        self.right_play_thread = None  # 右侧循环播放线程
        self.right_play_running = False  # 右侧播放开关（弹窗后设为True）
        
        self.infer_mp4_thread = None
        self.orig_video_path = ""
        self.pred_mp4_path = ""
        self.temp_frames_root = ""
        self.complete_callback = None
        self.frame_info_list = []
        self.realtime_video_writer = None  # 实时写入的视频写入器
        self._orig_video_loaded = False  # 标记原始视频是否已加载，防止覆盖

    def set_save_dir(self, new_save_root):
        if new_save_root and os.path.exists(new_save_root):
            self.save_root = new_save_root
            self.logger(f"ℹ️ 视频预测器已配置根目录：{self.save_root}")
        else:
            self.save_root = None
            self.logger(f"⚠️ 无效根目录，将使用默认目录")

    def set_complete_callback(self, callback):
        if callable(callback):
            self.complete_callback = callback
            self.logger(f"ℹ️ 已绑定视频预测完成回调")
        else:
            self.logger(f"⚠️ 回调函数不可调用，忽略绑定")

    def start(self, video_path):
        if not self.model:
            self.logger("❌ 模型未加载，无法预测")
            return
        
        if not os.path.exists(video_path):
            self.logger(f"❌ 视频不存在：{video_path}")
            return
        
        self._create_exclusive_sub_dir()
        # 初始化临时帧目录和实时MP4路径
        self.temp_frames_root = os.path.join(self.actual_save_dir, "frames")
        os.makedirs(self.temp_frames_root, exist_ok=True)
        self.logger(f"📂 已创建临时帧目录：{self.temp_frames_root}")

        self.orig_video_path = video_path
        orig_video_name = os.path.basename(video_path)
        orig_video_name_no_ext = os.path.splitext(orig_video_name)[0]
        # 实时MP4路径（推理时实时写入）
        self.pred_mp4_path = os.path.join(self.actual_save_dir, f"{orig_video_name_no_ext}_realtime.mp4")

        # 加载原始视频到左侧预览（仅加载一次，标记为已加载，防止后续覆盖）
        if not self._orig_video_loaded:
            if not self.video_player.load_video(video_path):
                return
            self.video_player.allow_loop = True
            self.video_player.start_play()
            self._orig_video_loaded = True
            self.logger("🎨 左侧原视频已开始循环播放")

        # 启动推理线程（推理时逐帧预览，不启动独立播放）
        with self.lock:
            self.is_running = True
        self.infer_mp4_thread = threading.Thread(target=self._infer_save_realtime, daemon=True)
        self.infer_mp4_thread.start()
        self.logger(f"🎬 开始视频推理，实时MP4将保存至：{self.pred_mp4_path}")

    def _init_realtime_writer(self, frame_size, fps):
        """初始化实时视频写入器"""
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.realtime_video_writer = cv2.VideoWriter(self.pred_mp4_path, fourcc, fps, frame_size, isColor=True)
        
        if not self.realtime_video_writer.isOpened():
            self.logger(f"⚠️ mp4v编码失败，尝试XVID格式")
            self.pred_mp4_path = os.path.splitext(self.pred_mp4_path)[0] + ".avi"
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self.realtime_video_writer = cv2.VideoWriter(self.pred_mp4_path, fourcc, fps, frame_size, isColor=True)
        
        if self.realtime_video_writer.isOpened():
            self.logger(f"📽️ 实时视频写入器初始化成功：{fps}fps，{frame_size}")
        else:
            self.logger(f"❌ 实时视频写入器初始化失败")

    def _infer_save_realtime(self):
        """核心：逐帧推理+实时保存帧+实时写入MP4+逐帧预览，推理完成立即清理frame+触发弹窗后播放"""
        cap = None
        # 新增：标记是否为正常完成推理（非中途打断）
        is_normal_complete = False
        try:
            cap = cv2.VideoCapture(self.orig_video_path)
            if not cap or not cap.isOpened():
                self.logger(f"❌ 无法打开原始视频：{self.orig_video_path}")
                return
            
            orig_fps = int(cap.get(cv2.CAP_PROP_FPS)) if cap.get(cv2.CAP_PROP_FPS) > 0 else 30
            orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_size = (orig_width, orig_height)
            # 初始化实时视频写入器
            self._init_realtime_writer(frame_size, orig_fps)

            frame_index = 0
            while self.is_running and cap.isOpened():
                # 新增：中途打断校验，收到停止指令立即退出推理循环
                if not self.is_running:
                    break
                
                ret, orig_frame = cap.read()
                if not ret:
                    self.logger(f"ℹ️ 视频帧读取完毕（共处理 {frame_index} 帧）")
                    # 仅正常读完所有帧，才标记为正常完成
                    is_normal_complete = True
                    break
                
                # 1. 创建当前帧唯一目录（避免覆盖）
                frame_unique_dir_name = f"frame_{frame_index:06d}_{int(time.time() * 1000)}"
                frame_unique_dir = os.path.join(self.temp_frames_root, frame_unique_dir_name)
                os.makedirs(frame_unique_dir, exist_ok=True)

                # 2. 模型推理（满足project/name非空、save=True）
                results = self.model.predict(
                    source=orig_frame,
                    save=True,
                    save_dir=frame_unique_dir,
                    project=self.temp_frames_root,
                    name=frame_unique_dir_name,
                    exist_ok=True,
                    save_txt=False,
                    save_conf=True,
                    save_crop=False,
                    device=self._get_device(),
                    imgsz=Config.IMGSZ,
                    conf=Config.CONF_THRESHOLD,
                    mask_threshold=[0.4, 0.9],
                    verbose=False,
                    stream=False
                )

                # 3. 读取YOLO保存的帧文件（不使用plot()，优化路径查找逻辑）
                yolo_saved_frame_path = None
                # 遍历可能的保存路径，兼容不同YOLO版本
                possible_paths = [
                    os.path.join(frame_unique_dir, frame_unique_dir_name, "image0.jpg"),
                    os.path.join(frame_unique_dir, "image0.jpg"),
                    os.path.join(frame_unique_dir, frame_unique_dir_name, "image0.png"),
                    os.path.join(frame_unique_dir, "image0.png")
                ]
                for path in possible_paths:
                    if os.path.exists(path):
                        yolo_saved_frame_path = path
                        break
                
                # 4. 推理过程中：逐帧更新右侧预览（不启动独立播放，仅单帧刷新）
                if yolo_saved_frame_path is not None:
                    pred_frame = cv2.imread(yolo_saved_frame_path)
                    if pred_frame is not None:
                        # 仅更新右侧预览（is_original=False），左侧保持原始视频不变
                        self._safe_update_preview_frame(pred_frame, is_original=False)
                        # 实时写入MP4（确保帧尺寸匹配）
                        if self.realtime_video_writer and self.realtime_video_writer.isOpened():
                            pred_frame_resized = cv2.resize(pred_frame, frame_size, interpolation=cv2.INTER_CUBIC)
                            self.realtime_video_writer.write(pred_frame_resized)
                        # 记录帧信息（仅用于日志）
                        self.frame_info_list.append({
                            "index": frame_index,
                            "path": yolo_saved_frame_path,
                            "dir": frame_unique_dir
                        })
                        if frame_index % 50 == 0:
                            self.logger(f"✅ 第 {frame_index} 帧：实时保存+预览+写入MP4完成")
                else:
                    self.logger(f"⚠️ 第 {frame_index} 帧保存失败，未找到有效帧文件，跳过")

                frame_index += 1
                # 控制推理速度匹配原视频帧率，避免过快
                time.sleep(1 / orig_fps)

        except Exception as e:
            self.logger(f"❌ 视频推理失败：{str(e)}")
            import traceback
            self.logger(f"📝 错误栈：{traceback.format_exc()}")
        finally:
            # 第一步：释放推理相关资源（避免文件占用）
            if cap:
                cap.release()
            if self.realtime_video_writer:
                self.realtime_video_writer.release()
                self.logger(f"✅ 实时MP4写入完成：{self.pred_mp4_path}")
            with self.lock:
                self.is_running = False
            self.orig_video_path = ""

            # 第二步：立即删除frame临时文件夹（无等待，直接清理）
            self._clean_temp_frames_immediately()

            # 第三步：仅当正常完成推理时，才触发弹窗和右侧播放（核心修改：阻断中途打断的无效操作）
            if is_normal_complete and self.complete_callback and callable(self.complete_callback):
                self.logger(f"ℹ️ 推理正常完成，触发弹窗并启用右侧独立播放")
                self.complete_callback()  # 执行弹窗逻辑
                self._enable_right_video_play_after_popup()  # 弹窗后启用独立播放
            else:
                if not is_normal_complete:
                    self.logger(f"ℹ️ 推理中途打断，不触发弹窗和右侧播放")

    def _enable_right_video_play_after_popup(self):
        """弹窗后启用右侧独立循环播放（核心：仅在弹窗后触发）"""
        # 校验视频文件是否存在，避免播放失败
        if not os.path.exists(self.pred_mp4_path):
            self.logger(f"❌ 预测视频文件不存在，无法启动右侧独立播放")
            return
        
        # 启动右侧独立播放线程
        self.right_play_running = True
        self.right_play_thread = threading.Thread(target=self._right_video_loop_play, daemon=True)
        self.right_play_thread.start()
        self.logger(f"✅ 弹窗后已启用右侧独立循环播放，播放文件：{self.pred_mp4_path}")

    def _right_video_loop_play(self):
        """右侧完整视频循环播放逻辑（优化：高频校验停止状态，支持中途打断立即响应）"""
        while self.right_play_running:
            # 校验1：外层循环开头，避免卡在视频打开失败的重试循环
            if not self.right_play_running:
                break
            
            # 初始化视频捕获对象
            self.right_video_cap = cv2.VideoCapture(self.pred_mp4_path)
            if not self.right_video_cap or not self.right_video_cap.isOpened():
                self.logger(f"⚠️ 右侧播放器无法打开视频文件，重试中...")
                # 校验2：重试前校验，避免无限重试不响应停止指令
                if not self.right_play_running:
                    break
                time.sleep(1)  # 缩短重试间隔，提升响应速度（2秒→1秒）
                continue
            
            # 循环播放当前视频（无缝循环）
            while self.right_play_running and self.right_video_cap.isOpened():
                # 校验3：内层循环首行，实时响应停止指令（核心修复）
                if not self.right_play_running:
                    break
                
                ret, frame = self.right_video_cap.read()
                if not ret:
                    # 校验4：重置帧位置前校验，避免无缝循环忽略停止指令
                    if not self.right_play_running:
                        break
                    # 播放到末尾，重置帧位置，重新循环
                    self.right_video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                
                # 更新右侧预览UI（持续循环播放）
                self._safe_update_preview_frame(frame, is_original=False)
                
                # 控制播放速度匹配视频原始帧率
                fps = self.right_video_cap.get(cv2.CAP_PROP_FPS) or 30
                time.sleep(1 / fps)
            
            # 释放捕获对象，防止资源泄露，重置对象引用
            if self.right_video_cap:
                self.right_video_cap.release()
                self.right_video_cap = None
            
            # 校验5：外层循环末尾，避免立即进入下一轮循环
            if not self.right_play_running:
                break
            time.sleep(0.5)  # 缩短间隔，提升终止响应速度（1秒→0.5秒）

    def _clean_temp_frames_immediately(self):
        """立即清理临时frames目录，无等待，处理文件占用异常"""
        if not os.path.exists(self.temp_frames_root):
            self.logger(f"⚠️ 无临时帧目录，无需清理")
            return
        
        # 立即删除目录（忽略部分临时文件占用，强制清理）
        try:
            shutil.rmtree(self.temp_frames_root, ignore_errors=True)
            # 校验删除结果
            if not os.path.exists(self.temp_frames_root):
                self.logger(f"🗑️ 已成功立即删除临时帧目录：{self.temp_frames_root}")
            else:
                self.logger(f"⚠️ 部分临时文件被占用，frame目录未完全删除")
        except PermissionError as e:
            self.logger(f"❌ 删除frame目录失败：文件被占用（{str(e)}）")
        except Exception as e:
            self.logger(f"❌ 删除frame目录失败：{str(e)}")

    def stop(self):
        """停止所有线程（包括右侧独立播放），优化中途打断逻辑，确保立即终止无残留"""
        # 第一步：优先终止右侧独立播放（核心：中途打断时先停播放器，再处理其他逻辑）
        self.logger(f"ℹ️ 正在终止右侧独立视频播放器...")
        # 1. 立即关闭播放开关，让双层循环检测到停止状态
        self.right_play_running = False
        
        # 2. 强制释放视频捕获资源，避免句柄泄露
        if self.right_video_cap:
            try:
                self.right_video_cap.release()
                self.right_video_cap = None
                self.logger(f"✅ 右侧视频捕获资源已强制释放")
            except Exception as e:
                self.logger(f"⚠️ 释放右侧视频捕获资源失败：{str(e)}")
        
        # 3. 等待播放线程正常退出，缩短超时提升响应速度（3秒→2秒）
        if self.right_play_thread and self.right_play_thread.is_alive():
            try:
                self.right_play_thread.join(timeout=2)
                self.logger(f"✅ 右侧独立播放线程已正常退出")
            except Exception as e:
                self.logger(f"⚠️ 等待右侧播放线程退出超时：{str(e)}")
        
        # 第二步：执行父类停止逻辑，终止推理循环
        super().stop()
        
        # 第三步：原有停止逻辑，释放其他资源
        if self.video_player:
            self.video_player.stop()
        if self.infer_mp4_thread and self.infer_mp4_thread.is_alive():
            self.infer_mp4_thread.join(timeout=2)
        if self.realtime_video_writer:
            self.realtime_video_writer.release()
        
        # 第四步：强制清理frame目录，重置播放线程状态
        self._clean_temp_frames_immediately()
        self.right_play_thread = None  # 重置线程对象，避免多次启停状态混乱
        self.logger("🛑 视频预测已完全停止，保留实时MP4文件")

# ===================== 摄像头预测器（仅创建camera子目录，修复resize dsize参数错误） =====================
class CameraPredictor(BasePredictor):
    """摄像头预测器：专属camera子目录，逐帧保存到frame+读取frame预览+写入MP4+结束清理frame"""
    def __init__(self, model, preview_panel, logger):
        super().__init__(model, preview_panel, logger)
        self.sub_dir_name = "camera"
        self.out = None
        self.predict_thread = None
        self.temp_frames_root = ""  # 临时frame目录根路径
        self.frame_index = 0  # 帧索引，用于命名唯一帧目录
        self.video_width = 640  # 视频写入宽度（固定/从摄像头获取）
        self.video_height = 480  # 视频写入高度（固定/从摄像头获取）

    def set_save_dir(self, new_save_root):
        """设置保存根目录：验证路径有效性与可写性"""
        try:
            if new_save_root and os.path.exists(new_save_root):
                abs_root = os.path.abspath(new_save_root)
                # 验证目录可写性
                test_file = os.path.join(abs_root, f".test_{int(time.time())}")
                with open(test_file, "w") as f:
                    f.write("test")
                os.remove(test_file)
                self.save_root = abs_root
                self.logger(f"ℹ️ 摄像头预测器已配置根目录：{self.save_root}")
            else:
                self.save_root = None
                self.logger(f"⚠️ 无效根目录，将使用默认目录")
        except Exception as e:
            self.save_root = None
            self.logger(f"⚠️ 目录不可写，使用默认目录：{str(e)}")

    def start(self, camera_id=0):
        """摄像头预测核心：创建frame目录+初始化写入器+逐帧采集/保存/推理/读取预览/写入MP4"""
        if not self.model:
            self.logger("❌ 模型未加载，无法预测")
            return
        
        # 1. 创建专属子目录和临时frame目录（与VideoPredictor格式一致）
        self._create_exclusive_sub_dir()
        self.temp_frames_root = os.path.join(self.actual_save_dir, "frames")
        os.makedirs(self.temp_frames_root, exist_ok=True)
        self.logger(f"📂 已创建摄像头临时frame目录：{self.temp_frames_root}")

        # 2. 启动采集逻辑
        with self.lock:
            self.is_running = True
        
        # 3. 初始化外接摄像头（DirectShow后端，减少帧丢失）
        try:
            camera_id_int = int(camera_id)
            self.cap = cv2.VideoCapture(camera_id_int, cv2.CAP_DSHOW)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 减少帧缓存，降低延迟
        except ValueError as e:
            self.logger(f"❌ 摄像头ID无效：{camera_id}（请输入数字）")
            with self.lock:
                self.is_running = False
            return
        
        # 4. 验证摄像头有效性
        ret, _ = self.cap.read()
        if not self.cap.isOpened() or not ret:
            self.logger(f"❌ 无法打开摄像头（ID={camera_id}），检查设备或驱动")
            if self.cap:
                self.cap.release()
            with self.lock:
                self.is_running = False
            return
        
        # 5. 配置摄像头分辨率并等待初始化，同时记录视频写入尺寸（关键：固定为整数）
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        time.sleep(0.2)
        
        # 6. 获取摄像头实际参数，初始化视频写入器（MP4优先），记录整数尺寸
        fps = int(self.cap.get(cv2.CAP_PROP_FPS)) or 30
        self.video_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 转为整数
        self.video_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 转为整数
        save_name = f"camera_pred_{int(time.time())}.mp4"
        self.result_path = os.path.join(self.actual_save_dir, save_name)
        
        # 初始化MP4视频写入器
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.out = cv2.VideoWriter(self.result_path, fourcc, fps, (self.video_width, self.video_height))
        if not self.out.isOpened():
            self.logger("⚠️ mp4v编码失败，尝试XVID格式（AVI）")
            save_name = f"camera_pred_{int(time.time())}.avi"
            self.result_path = os.path.join(self.actual_save_dir, save_name)
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            self.out = cv2.VideoWriter(self.result_path, fourcc, fps, (self.video_width, self.video_height))
        
        if self.out.isOpened():
            self.logger(f"📽️ 视频写入器初始化成功：{fps}fps，{self.video_width}x{self.video_height}，保存至：{self.result_path}")
        else:
            self.logger(f"❌ 视频写入器初始化失败，将仅保存帧图片")
        
        # 7. 启动推理线程（沿用VideoPredictor逻辑：保存frame→读取frame→预览→写入）
        self.logger(f"📹 摄像头预测已启动（ID={camera_id}），开始逐帧采集与推理")
        self.predict_thread = threading.Thread(target=self._predict_loop, daemon=True)
        self.predict_thread.start()

    def _predict_loop(self):
        """摄像头推理循环：逐帧采集→创建唯一帧目录→save=True保存→读取frame图片→预览→写入MP4"""
        while True:
            with self.lock:
                if not self.is_running:
                    break
            
            # 1. 读取摄像头帧（重试机制，避免帧丢失）
            ret, frame = self.cap.read()
            if not ret:
                self.logger(f"⚠️ 无法读取摄像头帧，重试中...")
                ret, frame = self.cap.read()
                if not ret:
                    time.sleep(0.05)
                    continue
            
            try:
                # 2. 创建当前帧的唯一目录（与VideoPredictor格式一致：frame_xxxxxx_xxxxxxxx）
                frame_unique_dir_name = f"frame_{self.frame_index:06d}_{int(time.time() * 1000)}"
                frame_unique_dir = os.path.join(self.temp_frames_root, frame_unique_dir_name)
                os.makedirs(frame_unique_dir, exist_ok=True)
                
                # 3. 更新左侧原始帧预览
                self._safe_update_preview_frame(frame, is_original=True)
                
                # 4. 模型推理（关键：save=True + 非空project + 非空name，保存推理结果到frame目录）
                results = self.model.predict(
                    source=frame,
                    save=True,  # 启用保存功能，将推理结果图片保存到指定目录
                    project=self.temp_frames_root,  # 非空project：指定保存根目录（frame目录）
                    name=frame_unique_dir_name,  # 非空name：指定当前帧的子目录名（与唯一帧目录一致）
                    exist_ok=True,  # 允许目录已存在，避免报错
                    save_txt=False,
                    save_conf=True,
                    save_crop=False,
                    device=self._get_device(),
                    imgsz=Config.IMGSZ,
                    conf=Config.CONF_THRESHOLD,
                    verbose=False
                )
                
                # 5. 读取YOLO保存到frame目录的推理图片（兼容VideoPredictor的路径查找逻辑）
                yolo_saved_frame_path = None
                possible_paths = [
                    os.path.join(frame_unique_dir, frame_unique_dir_name, "image0.jpg"),
                    os.path.join(frame_unique_dir, "image0.jpg"),
                    os.path.join(frame_unique_dir, frame_unique_dir_name, "image0.png"),
                    os.path.join(frame_unique_dir, "image0.png")
                ]
                for path in possible_paths:
                    if os.path.exists(path):
                        yolo_saved_frame_path = path
                        break
                
                # 6. 从frame目录读取图片，更新右侧实时预览（核心需求）
                pred_frame = None
                if yolo_saved_frame_path is not None:
                    pred_frame = cv2.imread(yolo_saved_frame_path)
                    if pred_frame is not None:
                        self._safe_update_preview_frame(pred_frame, is_original=False)
                    else:
                        self.logger(f"⚠️ 第 {self.frame_index} 帧：无法读取frame目录中的推理图片")
                        pred_frame = frame  # 降级显示原始帧
                else:
                    self.logger(f"⚠️ 第 {self.frame_index} 帧：frame目录中未找到推理图片")
                    pred_frame = frame  # 降级显示原始帧
                
                # 7. 实时写入推理结果到MP4视频文件（修复resize dsize参数错误，关键修改）
                if self.out and self.out.isOpened() and pred_frame is not None:
                    # 关键1：dsize使用预存的整数元组（宽度, 高度），符合OpenCV要求
                    # 关键2：确保dsize是(int, int)类型，避免float类型错误
                    target_size = (self.video_width, self.video_height)
                    # 关键3：仅当帧尺寸与目标尺寸不一致时才resize，提升效率
                    if pred_frame.shape[1] != target_size[0] or pred_frame.shape[0] != target_size[1]:
                        pred_frame_resized = cv2.resize(
                            pred_frame,
                            dsize=target_size,  # 合法的整数元组，解决核心错误
                            interpolation=cv2.INTER_CUBIC
                        )
                    else:
                        pred_frame_resized = pred_frame  # 尺寸一致，无需resize
                    self.out.write(pred_frame_resized)
                
                # 8. 日志记录（每50帧打印一次，避免日志刷屏）
                if self.frame_index % 50 == 0:
                    self.logger(f"✅ 第 {self.frame_index} 帧：保存至frame+读取预览+写入MP4完成")
                
                # 9. 更新帧索引，控制推理帧率（与摄像头帧率同步）
                self.frame_index += 1
                time.sleep(1/30)  # 对应30fps，可根据实际摄像头帧率调整
                    
            except Exception as e:
                self.logger(f"⚠️ 第 {self.frame_index} 帧处理失败：{str(e)}")
                self.frame_index += 1
                continue
        
        # 推理停止后：释放资源+清理frame目录
        self._release_resources_and_clean_frame()

    def _release_resources_and_clean_frame(self):
        """释放所有资源，删除临时frame目录（与VideoPredictor清理逻辑一致）"""
        # 1. 释放视频写入器
        if self.out:
            self.out.release()
            self.logger(f"✅ 摄像头视频写入完成，保存至：{self.result_path}（共处理 {self.frame_index} 帧）")
        
        # 2. 释放摄像头资源
        if self.cap:
            self.cap.release()
            cv2.destroyAllWindows()
            cv2.waitKey(1)
        
        # 3. 立即删除临时frame目录（无等待，强制清理，与VideoPredictor一致）
        if os.path.exists(self.temp_frames_root):
            try:
                shutil.rmtree(self.temp_frames_root, ignore_errors=True)
                if not os.path.exists(self.temp_frames_root):
                    self.logger(f"🗑️ 已成功删除摄像头临时frame目录：{self.temp_frames_root}")
                else:
                    self.logger(f"⚠️ 部分frame文件被占用，目录未完全删除")
            except PermissionError as e:
                self.logger(f"❌ 删除frame目录失败：文件被占用（{str(e)}）")
            except Exception as e:
                self.logger(f"❌ 删除frame目录失败：{str(e)}")
        
        # 4. 完成日志
        self.logger(f"✅ 摄像头预测子线程已正常退出，共处理 {self.frame_index} 帧")

    def stop(self):
        """重写停止方法：释放摄像头、视频写入器、清理frame目录，等待推理线程结束"""
        super().stop()
        
        # 1. 等待推理线程结束
        if self.predict_thread and self.predict_thread.is_alive():
            self.predict_thread.join(timeout=1)
        
        # 2. 释放视频写入器
        if self.out:
            try:
                self.out.release()
            except Exception as e:
                self.logger(f"⚠️ 释放视频写入器失败：{str(e)}")
            self.out = None
        
        # 3. 释放摄像头资源
        if self.cap:
            try:
                self.cap.release()
                cv2.destroyAllWindows()
            except Exception as e:
                self.logger(f"⚠️ 释放摄像头资源失败：{str(e)}")
        
        # 4. 再次确认清理frame目录（双重保障，与VideoPredictor一致）
        if os.path.exists(self.temp_frames_root):
            try:
                shutil.rmtree(self.temp_frames_root, ignore_errors=True)
            except Exception as e:
                self.logger(f"❌ 强制删除frame目录失败：{str(e)}")
        
        # 5. 最终日志
        self.logger(f"🛑 摄像头预测已完全停止，视频文件保存至：{self.result_path}")