import sys
import os
import threading
import queue
from argparse import ArgumentParser
from pathlib import Path

import cv2
import mmcv
from PyQt5.QtCore import QTimer, Qt, pyqtSignal, QObject
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QWidget, QFileDialog
from mmyolo.registry import MODELS, VISUALIZERS  # 确保导入 VISUALIZERS

from mmdet.apis import inference_detector, init_detector
from mmengine.config import Config
import torch
from deep_sort_realtime.deepsort_tracker import DeepSort
import numpy as np
import matplotlib.path as mpltPath  # 确保导入
import signal
import sys
import time
import csv
from math import sqrt
from datetime import datetime

# 创建信号类，用于线程和主线程通信
class Communicate(QObject):
    update_frame_signal = pyqtSignal(np.ndarray)  # 信号传递 np.ndarray 类型的数据

class VideoDemo(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()  # 初始化界面
        self.polygon_points = []  # 存储用户点击的点
        self.is_drawing_polygon = False  # 是否正在绘制多边形
        self.timer = QTimer()  # 创建定时器
        self.timer.timeout.connect(self.update_frame)  # 定时器中断函数 update_frame
        self.video_reader = None
        self.model = None
        self.visualizer = None
        self.frame_queue = queue.Queue(maxsize=30)  # 加大队列大小
        self.stop_flag = False
        self.frame_count = 0
        self.tracker_enter = DeepSort(max_age=30)  # 初始化 Deep SORT 跟踪器
        self.tracker_exit1 = DeepSort(max_age=50, n_init = 1, nn_budget = 30)  # 初始化 Deep SORT 跟踪器
        self.tracker_exit2 = DeepSort(max_age=30)  # 初始化 Deep SORT 跟踪器
        self.communicate = Communicate()  # 初始化信号机制
        self.communicate.update_frame_signal.connect(self.display_frame)  # 连接信号到 display_frame
        self.counted_tracks = set()  # 初始化一个用于记录已计数的轨迹ID的集合
        self.entry_region = []
        self.exit1_region = []
        self.exit2_region = []
        self.enter_egg = 0
        self.exit1_egg = 0
        self.exit2_egg = 0
        self.count_egg = 0
        # 初始化各个区域内当前记录的最大ID值，用于记录鸡蛋数量
        self.enter_max_ID = 0
        self.exit1_max_ID = 0
        self.exit2_max_ID = 0
        self.track_id_enter = 0
        self.track_id_exit1 = 0
        self.track_id_exit2 = 0

        # 记录鸡蛋位置和时间戳的字典
        self.egg_positions = {}
        
        self.egg_speeds = []     # 用于存储计算出的速度

        # 初始化边界框和得分
        self.bboxes1 = []
        self.scores1 = []
        self.bboxes2 = []
        self.scores2 = []
        self.bboxes3 = []
        self.scores3 = []
        self.bboxes = []
        self.scores = []
        os.environ["CUDA_VISIBLE_DEVICES"] = '0'

        self.results = []  # 用于存储结果
        self.save_timer = QTimer()  # 创建定时器
        self.save_timer.timeout.connect(self.save_results)  # 定时器中断函数 save_results

        self.detect_stoptime = 0  # 记录检测停止的时间
        self.stop_longtime = 0  # 记录暂停的时间
        self.detect_alltime = 0  # 记录检测总时间
        self.x = 0  # 一次运行标志位
        self.zone_1 = 0  # 区域1是否划定区域做检测的标志位
        self.zone_2 = 0  # 区域2是否划定区域做检测的标志位
        self.zone_3 = 0  # 区域3是否划定区域做检测的标志位
        

    def initUI(self):
        # 初始化各种元素

        self.setWindowTitle('坤蛋检测系统')
        self.setGeometry(100, 100, 800, 600)

        self.label = QLabel(self)
        self.label.setFixedSize(800, 450)
        self.label.mousePressEvent = self.get_mouse_click  # 捕获鼠标点击事件

        self.label_frame1 = QLabel(self)
        self.label_frame1.setFixedSize(200, 150)
        self.label_frame2 = QLabel(self)
        self.label_frame2.setFixedSize(200, 150)
        self.label_frame3 = QLabel(self)
        self.label_frame3.setFixedSize(200, 150)

        self.load_button = QPushButton('载入视频', self)
        self.load_button.clicked.connect(self.load_video)
        self.load_button.setStyleSheet('font-size: 20px')  # 设置按钮字体为20号字体

        self.start_button = QPushButton('开始检测', self)
        self.start_button.clicked.connect(self.start_detection)
        self.start_button.setStyleSheet('font-size: 20px')

        self.stop_button = QPushButton('停止检测', self)
        self.stop_button.clicked.connect(self.stop_detection)
        self.stop_button.setStyleSheet('font-size: 20px')

        self.entry_button = QPushButton('划定入口区域', self)
        self.entry_button.clicked.connect(lambda: self.start_rectangle_drawing('entry'))
        self.entry_button.setStyleSheet('font-size: 20px')

        self.exit1_button = QPushButton('划定出口区域1', self)
        self.exit1_button.clicked.connect(lambda: self.start_rectangle_drawing('exit1'))
        self.exit1_button.setStyleSheet('font-size: 20px')

        self.exit2_button = QPushButton('划定出口区域2', self)
        self.exit2_button.clicked.connect(lambda: self.start_rectangle_drawing('exit2'))
        self.exit2_button.setStyleSheet('font-size: 20px')

        self.end_polygon_button = QPushButton('完成划定', self)
        self.end_polygon_button.clicked.connect(self.end_rectangle_drawing)
        self.end_polygon_button.setStyleSheet('font-size: 20px')

        # 实时显示的数量
        self.egg_count_label = QLabel(self)
        self.egg_count_label.setText('当前传送带中鸡蛋数量: 0')
        self.egg_count_label.setStyleSheet('font-size: 20px')

        # 显示送入鸡蛋数量
        self.egg_enter_label = QLabel(self)
        self.egg_enter_label.setText('入口区域鸡蛋数量: 0')
        self.egg_enter_label.setStyleSheet('font-size: 20px')

        # 显示送出鸡蛋数量
        self.egg_exit1_label = QLabel(self)
        self.egg_exit1_label.setText('出口1区域鸡蛋数量: 0')
        self.egg_exit1_label.setStyleSheet('font-size: 20px')

        # 显示送出鸡蛋数量
        self.egg_exit2_label = QLabel(self)
        self.egg_exit2_label.setText('出口2区域鸡蛋数量: 0')
        self.egg_exit2_label.setStyleSheet('font-size: 20px')

        # 显示当前区域流量
        self.flow_rate_label = QLabel(self)
        self.flow_rate_label.setText('流量: 0 个鸡蛋/秒')
        self.flow_rate_label.setStyleSheet('font-size: 20px')

        # 显示当前已经检测的时间
        self.time_label = QLabel(self)
        self.time_label.setText('已检测时间: 0 秒')
        self.time_label.setStyleSheet('font-size: 20px')


        # 导出结果按钮
        self.export_button = QPushButton('导出结果', self)
        self.export_button.clicked.connect(self.export_results)
        self.export_button.setStyleSheet('font-size: 20px')

        # 清除数据按钮
        self.clear_button = QPushButton('清空数据', self)
        self.clear_button.clicked.connect(self.clear_data)
        self.clear_button.setStyleSheet('font-size: 20px')

        # 显示当前区域流速，有BUG，暂时不显示
        # self.speed_label = QLabel(self)
        # self.speed_label.setText('速度: 0 像素/秒')
        # self.speed_label.setStyleSheet('font-size: 20px')


        # 设置布局
        layout = QVBoxLayout()
        layout.addWidget(self.load_button)
        layout.addWidget(self.start_button)
        layout.addWidget(self.stop_button)
        layout.addWidget(self.entry_button)
        layout.addWidget(self.exit1_button)
        layout.addWidget(self.exit2_button)
        layout.addWidget(self.end_polygon_button)
        layout.addWidget(self.export_button)  
        layout.addWidget(self.clear_button)  # 添加清空数据按钮
        layout.addWidget(self.egg_count_label)
        layout.addWidget(self.egg_enter_label)
        layout.addWidget(self.egg_exit1_label)
        layout.addWidget(self.egg_exit2_label)
        layout.addWidget(self.flow_rate_label)
        
        layout.addWidget(self.time_label)

        # layout.addWidget(self.speed_label) # 计速功能有问题，暂时不显示，后续修复



        frame_three = QHBoxLayout()
        frame_three.addWidget(self.label_frame1)
        frame_three.addWidget(self.label_frame2)
        frame_three.addWidget(self.label_frame3)

        frame_all = QVBoxLayout()
        frame_all.addWidget(self.label)
        frame_all.addLayout(frame_three)

        system = QHBoxLayout()
        system.addLayout(frame_all)
        system.addLayout(layout)

        self.setLayout(system)


    def clear_data(self):
        """清空初始数据"""

        self.start_time = time.time()  # 重新记录开始时间
        self.egg_count = 0  # 重置通过入口的鸡蛋数量
        self.enter_egg = 0
        self.exit1_egg = 0
        self.exit2_egg = 0
        self.count_egg = 0
        self.x = 0  # 重置一次运行标志位
        self.track_id_enter = 0
        self.track_id_exit1 = 0
        self.track_id_exit2 = 0
        self.enter_max_ID = 0
        self.exit1_max_ID = 0
        self.exit2_max_ID = 0
        self.stop_longtime = 0
        self.detect_alltime = 0
        self.detect_stoptime = 0
        self.egg_speeds = []  # 清空速度列表
        self.counted_tracks = set()  # 清空已计数轨迹ID集合
        self.egg_positions = {}
        self.egg_speeds = []  # 清空速度列表
        self.results = []  # 清空结果列表
        self.save_timer.stop()  # 停止保存结果的定时器
        self.stop_detection()  # 停止检测
        self.tracker_enter = DeepSort(max_age=30)  # 重新初始化 Deep SORT 跟踪器
        self.tracker_exit1 = DeepSort(max_age=30)
        self.tracker_exit2 = DeepSort(max_age=30)
        self.frame_queue.queue.clear()  # 清空队列
        self.label.clear()  # 清空显示
        self.label_frame1.clear()
        self.label_frame2.clear()
        self.label_frame3.clear()
        self.entry_region = []
        self.exit1_region = []
        self.exit2_region = []
        self.bboxes1 = []
        self.scores1 = []
        self.bboxes2 = []
        self.scores2 = []
        self.bboxes3 = []
        self.scores3 = []
        self.bboxes = []
        self.scores = []
        self.egg_count_label.setText('当前传送带中鸡蛋数量: 0')  # 重置标签文本
        self.egg_enter_label.setText('入口区域鸡蛋数量: 0')
        self.egg_exit1_label.setText('出口1区域鸡蛋数量: 0')
        self.egg_exit2_label.setText('出口2区域鸡蛋数量: 0')
        self.flow_rate_label.setText('流量: 0 个鸡蛋/秒')
        self.time_label.setText('已检测时间: 0 秒')
        # self.speed_label.setText('速度: 0 像素/秒')
        print("数据已清空，可以重新导入视频开始检测")



    def export_results(self):
        """导出结果到用户指定路径"""
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(self, "导出结果", "", "CSV Files (*.csv);;All Files (*)", options=options)
        if file_path:
            # 获取最新保存的文件
            save_dir = os.path.join(os.path.dirname(__file__), 'save_result')
            latest_file = max([os.path.join(save_dir, f) for f in os.listdir(save_dir)], key=os.path.getctime)
            if os.path.exists(latest_file):
                os.rename(latest_file, file_path)
                print(f"结果已导出到 {file_path}")
            else:
                print("没有找到结果文件")


    def load_video(self):
        try:
            options = QFileDialog.Options()
            file_name, _ = QFileDialog.getOpenFileName(self, "Open Video File", "", "Videos (*.mp4 *.avi *.mov)", options=options)
            if file_name:
                print(f"加载视频文件: {file_name}")
                self.video_reader = mmcv.VideoReader(file_name)  # 使用 mmcv.VideoReader 读取视频文件
                
                # 读取第一帧并显示
                first_frame = self.video_reader.read()
                if first_frame is not None:
                    self.display_frame(first_frame)  # 显示第一帧
                
                self.init_model()  # 初始化模型
                self.start_video_thread()  # 启动视频读取线程
        except Exception as e:
            print(f"加载视频时发生错误: {e}")

    def calculate_smoothed_speed(self, egg_speeds, window_size=5):
        """
        计算平滑后的速度，使用滑动平均法来平滑速度变化。
        """
        if len(egg_speeds) < window_size:
            # 如果当前速度数据点少于窗口大小，返回平均速度
            return np.mean(egg_speeds)
        
        # 使用滑动窗口计算平均速度
        smoothed_speeds = []
        for i in range(len(egg_speeds) - window_size + 1):
            window = egg_speeds[i:i + window_size]
            smoothed_speeds.append(np.mean(window))
        
        return np.mean(smoothed_speeds)
    def get_mouse_click(self, event):
        """捕获鼠标点击坐标并进行坐标转换"""
        if not self.is_drawing_rectangle:
            return

        label_width = self.label.width()
        label_height = self.label.height()

        frame = self.frame_queue.queue[-1] if not self.frame_queue.empty() else None

        if frame is None:
            return

        frame_height, frame_width, _ = frame.shape

        scale_x = label_width / frame_width
        scale_y = label_height / frame_height

        x = int(event.pos().x() / scale_x)
        y = int(event.pos().y() / scale_y)

        if len(self.rectangle_points) == 0:
            # 第一次点击，记录矩形的起点
            self.rectangle_points.append((x, y))
            print(f"矩形起点 ({x}, {y}) 已记录")
        else:
            # 第二次点击，记录矩形的终点并完成矩形
            self.rectangle_points.append((x, y))
            print(f"矩形终点 ({x}, {y}) 已记录")
            self.draw_rectangle()  # 绘制矩形
            self.is_drawing_rectangle = False

    def draw_rectangle(self):
        """在视频帧上绘制矩形区域"""
        if len(self.rectangle_points) == 2:
            frame = self.frame_queue.get()
            pt1 = self.rectangle_points[0]
            pt2 = self.rectangle_points[1]
            cv2.rectangle(frame, pt1, pt2, (255, 0, 0), 2)  # 绘制矩形
            self.display_frame(frame)


    def start_rectangle_drawing(self, region_type):
        """开始绘制矩形区域"""
        self.rectangle_points = []
        self.is_drawing_rectangle = True
        self.current_region_type = region_type  # 保存当前区域类型

    def end_rectangle_drawing(self):
        """结束绘制矩形区域"""
        if len(self.rectangle_points) == 2:
            if self.current_region_type == 'entry':
                self.entry_region = self.rectangle_points
                self.zone_1 = 0  # 划定区域后，将区域标志位设为0,表示划定
            elif self.current_region_type == 'exit1':
                self.exit1_region = self.rectangle_points
                self.zone_2 = 0  # 划定区域后，将区域标志位设为0,表示划定
            elif self.current_region_type == 'exit2':
                self.exit2_region = self.rectangle_points
                self.zone_3 = 0  # 划定区域后，将区域标志位设为0,表示划定
            print(f"{self.current_region_type} 区域矩形已设定")

            # 确保 x1 < x2 和 y1 < y2
            for region in [self.entry_region, self.exit1_region, self.exit2_region]:
                if region and len(region) == 2:
                    pt1, pt2 = region
                    x1, y1 = min(pt1[0], pt2[0]), min(pt1[1], pt2[1])
                    x2, y2 = max(pt1[0], pt2[0]), max(pt1[1], pt2[1])
                    region[:] = [(x1, y1), (x2, y2)]  # 更新区域坐标
                    
        else:
            print("请确保绘制完整的矩形区域。")

    def init_model(self):
        try:
            config_path = 'F:\\money_project\\openmmlab\\mmyolo\\configs\\yyx_egg\\yolov5_s-v61_syncbn_fast_1xb32-100e_egg.py'
            checkpoint_path = 'F:/money_project/openmmlab/mmyolo/best_coco_bbox_mAP_epoch_98.pth'

            config = Config.fromfile(config_path)
            if 'init_cfg' in config.model.backbone:
                config.model.backbone.init_cfg = None

            if torch.cuda.is_available():
                device = 'cuda:0'
                print("使用GPU进行运算")
            else:
                device = 'cpu'
                print("CUDA 不可用，使用 CPU 进行运算")

            self.model = init_detector(config, checkpoint_path, device=device, cfg_options={})
            self.visualizer = VISUALIZERS.build(self.model.cfg.visualizer)
            self.visualizer.dataset_meta = self.model.dataset_meta
        except Exception as e:
            print(f"初始化模型时发生错误: {e}")

    def start_video_thread(self):
        self.stop_flag = False  # 初始化停止标志为 False
        self.video_thread = threading.Thread(target=self.read_frames)  # 创建一个线程来读取视频帧
        self.video_thread.start()  # 启动线程
        

    def read_frames(self):
        """读取视频帧"""
        try:
            while not self.stop_flag:
                if not self.frame_queue.full():
                    frame = self.video_reader.read()  # 读取视频帧
                    if frame is None:
                        self.stop_flag = True
                        break
                    self.frame_queue.put(frame)  # 将帧加入队列
                else:
                    threading.Event().wait(0.01)  # 防止过快读取，等待
        except Exception as e:
            print(f"读取视频帧时发生错误: {e}")


    def start_detection(self):
        """开始检测"""
        if self.entry_region == []:
            self.zone_1 = 1 # 如果没有划定区域，将区域标志位设为1,表示不划定

        if self.exit1_region == []:
            self.zone_2 = 1 # 如果没有划定区域，将区域标志位设为1,表示不划定

        if self.exit2_region == []:
            self.zone_3 = 1 # 如果没有划定区域，将区域标志位设为1,表示不划定
        # 打印3个标志位
        print(f"zone_1: {self.zone_1}, zone_2: {self.zone_2}, zone_3: {self.zone_3}")
        
        if not self.video_reader:
            print("视频文件未加载")
            return

        # 如果定时器未启动，或者之前已停止，重新启动定时器
        if not self.timer.isActive():
            self.timer.start(1000 // self.video_reader.fps)
            print("开始检测")

        # 如果线程已经停止，需要重新启动视频读取线程
        if self.stop_flag:
            self.stop_flag = False
            self.start_video_thread()
        
        
        if self.x == 0: # 此条件只执行一次
            self.x = 1
            self.detect_stoptime = time.time() # 记录首次暂停时间
            self.start_time = time.time()  # 记录首次开始检测时间

        self.save_timer.start(10000)  # 每10秒保存一次结果定时器开启
        
        self.stop_longtime += time.time() - self.detect_stoptime # 记录这一段暂停的时间并累加
        


    def stop_detection(self):
        """停止检测"""
        self.stop_flag = True  # 设置停止标志
        self.timer.stop()  # 停止定时器
        self.save_timer.stop()  # 停止保存结果的定时器
        
        self.detect_stoptime = time.time() # 正在暂停的时间
        if self.video_thread.is_alive():
            self.video_thread.join()  # 等待线程结束
        print("检测已停止")

    def save_results(self):
        """保存当前结果到CSV文件"""
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        result = {
            'time': current_time,
            'enter_egg': self.enter_egg,
            'exit1_egg': self.exit1_egg,
            'exit2_egg': self.exit2_egg,
            'flow_rate': self.flow_rate
        }
        self.results.append(result)

        # 确保保存目录存在
        save_dir = os.path.join(os.path.dirname(__file__), 'save_result')
        os.makedirs(save_dir, exist_ok=True)

        # 使用当前时间作为文件名的一部分
        file_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(save_dir, f'egg_result_{file_time}.csv')
        
        with open(save_path, 'w', newline='') as csvfile:
            fieldnames = ['time', 'enter_egg', 'exit1_egg', 'exit2_egg', 'flow_rate']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.results)
        print(f"结果已保存到 {save_path}")

    def update_frame(self):
        if self.model and not self.frame_queue.empty():
            threshold = 0.9
            frame = self.frame_queue.get()  # 从队列中获取一帧
            h, w = frame.shape[:2]  # 获取帧的高度和宽度

            try:
                # print(f"Frame size: {w}x{h}")

                # 初始化区域帧变量
                frame1, frame2, frame3 = None, None, None
                entry_polygon, exit1_polygon, exit2_polygon = None, None, None  # 初始化为 None
                # 提取入口区域
                if len(self.entry_region) == 2 and self.zone_1 == 0:
                    x1, y1 = self.entry_region[0]
                    x2, y2 = self.entry_region[1]

                    x1, x2 = sorted([x1, x2])
                    y1, y2 = sorted([y1, y2])

                    if x1 < 0 or x2 > w or y1 < 0 or y2 > h:
                        raise ValueError("Entry region out of bounds.")
                    entry_polygon = np.array([
                        [x1, y1],  # 左上角
                        [x2, y1],  # 右上角
                        [x2, y2],  # 右下角
                        [x1, y2]   # 左下角
                    ])
                    frame1 = frame[y1:y2, x1:x2]  # 提取区域
                    # print(f"Frame1 size: {frame1.shape}")

                # 提取出口1区域
                if len(self.exit1_region) == 2 and self.zone_2 == 0:
                    x1, y1 = self.exit1_region[0]
                    x2, y2 = self.exit1_region[1]

                    x1, x2 = sorted([x1, x2])
                    y1, y2 = sorted([y1, y2])

                    if x1 < 0 or x2 > w or y1 < 0 or y2 > h:
                        raise ValueError("Exit1 region out of bounds.")
                    exit1_polygon = np.array([
                        [x1, y1],  # 左上角
                        [x2, y1],  # 右上角
                        [x2, y2],  # 右下角
                        [x1, y2]   # 左下角
                    ])
                    frame2 = frame[y1:y2, x1:x2]
                    # print(f"Frame2 size: {frame2.shape}")

                # 提取出口2区域
                if len(self.exit2_region) == 2 and self.zone_3 == 0:
                    x1, y1 = self.exit2_region[0]
                    x2, y2 = self.exit2_region[1]

                    x1, x2 = sorted([x1, x2])
                    y1, y2 = sorted([y1, y2])

                    if x1 < 0 or x2 > w or y1 < 0 or y2 > h:
                        raise ValueError("Exit2 region out of bounds.")
                    exit2_polygon = np.array([
                        [x1, y1],  # 左上角
                        [x2, y1],  # 右上角
                        [x2, y2],  # 右下角
                        [x1, y2]   # 左下角
                    ])
                    frame3 = frame[y1:y2, x1:x2]
                    # print(f"Frame3 size: {frame3.shape}")

                # 对所有区域进行检测
                result = inference_detector(self.model, frame)

                # 获取全图检测结果
                if result is not None and result.pred_instances.bboxes.numel() > 0:
                    self.bboxes = result.pred_instances.bboxes.cpu().numpy()
                    self.scores = result.pred_instances.scores.cpu().numpy()
                else:
                    print("No detections in frame or result is invalid")

                # 全图检测结果
                detections = [
                    [
                        [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]],  # (left, top, width, height)
                        float(score),  # 置信度
                        0  # 类别（假设是单一类别）
                    ]
                    for bbox, score in zip(self.bboxes, self.scores) if score > threshold
                ]
                # print(f"全图检测结果: {detections}")
                # 画出全图检测结果中的框的左上角点
                for detection in detections:
                    bbox, score, _ = detection
                    cv2.circle(frame, (int(bbox[0]), int(bbox[1])), 5, (0, 0, 255), -1)

                filtered_enter = []
                filtered_exit1 = []
                filtered_exit2 = []

                for detection in detections:
                    bbox, score, _ = detection
                    if score > threshold:
                        center_x = bbox[0] + (bbox[2] / 2)
                        center_y = bbox[1] + (bbox[3] / 2)
                        # print(f"yyz: {bbox[2] - bbox[0], bbox[3] - bbox[1]}")
                        center_point = (center_x, center_y)
                        # print(f"中心点: {center_point}")
                        # print(f"入口区域: {entry_polygon}")
                        # print(f"出口1区域: {exit1_polygon}")
                        # print(f"出口2区域: {exit2_polygon}")
                        # 画出中心点和entry_polygon
                        cv2.circle(frame, (int(center_x), int(center_y)), 5, (0, 0, 255), -1)
                        # 画出bbox
                        # print(f"bbox: {bbox}")
                        # cv2.polylines(frame, [boxx], isClosed=True, color=(0, 255, 0), thickness=2)
                        # cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3]), (255, 0, 0), 2))
                        if self.zone_1 == 0: 
                            cv2.polylines(frame, [entry_polygon], isClosed=True, color=(0, 255, 0), thickness=2)
                        if self.zone_2 == 0:
                            cv2.polylines(frame, [exit1_polygon], isClosed=True, color=(0, 255, 0), thickness=2)
                        if self.zone_3 == 0:
                            cv2.polylines(frame, [exit2_polygon], isClosed=True, color=(0, 255, 0), thickness=2)
                        current_time = time.time()
                        # 检查中心点是否在入口区域内
                        if self.zone_1 == 0:
                            if cv2.pointPolygonTest(entry_polygon, center_point, False) >= 0:
                                filtered_enter.append([bbox, score])
                                track_id = self.track_id_enter

                                # 更新鸡蛋位置和时间戳
                                if track_id not in self.egg_positions:
                                    self.egg_positions[track_id] = []
                                self.egg_positions[track_id].append((center_x, center_y, current_time))

                                # 计算速度（确保至少有两个位置记录）
                                if len(self.egg_positions[track_id]) > 1:
                                    prev_position = self.egg_positions[track_id][-2]
                                    prev_x, prev_y, prev_time = prev_position
                                    distance = np.sqrt((center_x - prev_x) ** 2 + (center_y - prev_y) ** 2)
                                    time_diff = current_time - prev_time

                                    if time_diff > 0:
                                        speed = distance / time_diff
                                        self.egg_speeds.append(speed)

                        if self.zone_2 == 0:
                            if cv2.pointPolygonTest(exit1_polygon, center_point, False) >= 0:
                                filtered_exit1.append([bbox, score])
                        if self.zone_3 == 0:
                            if cv2.pointPolygonTest(exit2_polygon, center_point, False) >= 0:
                                filtered_exit2.append([bbox, score])
                    # print(f"filtered_enter: {filtered_enter}")
                    # print(f"filtered_exit1: {filtered_exit1}")
                    # print(f"filtered_exit2: {filtered_exit2}")



                # 计算平滑后的速度
                # if self.egg_speeds:
                #     if len(self.egg_speeds) > 10:
                #         recent_speeds = self.egg_speeds[-10:]  # 取最新的10个鸡蛋速度
                #         smoothed_speed = self.calculate_smoothed_speed(recent_speeds)  # 平滑处理
                #     else:
                #         smoothed_speed = self.calculate_smoothed_speed(self.egg_speeds)  # 鸡蛋不足10个时，平滑所有鸡蛋速度
                    
                #     print(f"当前鸡蛋的平滑平均速度: {smoothed_speed:.2f} 像素/秒")
                #     # self.speed_label.setText(f'速度: {smoothed_speed:.2f} 像素/秒') # 暂时不修复计速功能，后续修复
                # else:
                #     print(f"没有足够的数据来计算速度")

                if filtered_enter and self.zone_1 == 0:
                    tracks_enter = self.tracker_enter.update_tracks(filtered_enter, frame=frame) if filtered_enter else []
                    for track in tracks_enter:
                        if not track.is_confirmed() or track.time_since_update > 1:
                            continue
                        self.track_id_enter = track.track_id
                        if self.enter_max_ID < int(self.track_id_enter):# 更新入口区域最大ID
                            self.enter_max_ID = int(self.track_id_enter)
                        bbox_enter = track.to_tlbr()
                        cv2.rectangle(frame, (int(bbox_enter[0]), int(bbox_enter[1])), (int(bbox_enter[2]), int(bbox_enter[3])), (255, 0, 0), 2)
                        cv2.putText(frame, f'ID: {self.track_id_enter}', (int(bbox_enter[0]), int(bbox_enter[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)                
                
                if filtered_exit1 and self.zone_2 == 0:
                    tracks_exit1 = self.tracker_exit1.update_tracks(filtered_exit1, frame=frame) if filtered_exit1 else []
                    for track in tracks_exit1:
                        if not track.is_confirmed() or track.time_since_update > 1:
                            continue
                        self.track_id_exit1 = track.track_id
                        if self.exit1_max_ID < int(self.track_id_exit1):# 更新出口1区域最大ID
                            self.exit1_max_ID = int(self.track_id_exit1)
                        bbox_exit1 = track.to_tlbr()
                        # cv2.rectangle(frame, (int(bbox_exit1[0]), int(bbox_exit1[1])), (int(bbox_exit1[2]), int(bbox_exit1[3]), (255, 0, 0), 2))
                        cv2.putText(frame, f'ID: {self.track_id_exit1}', (int(bbox_exit1[0]), int(bbox_exit1[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)

                if filtered_exit2 and self.zone_3 == 0:
                    tracks_exit2 = self.tracker_exit2.update_tracks(filtered_exit2, frame=frame) if filtered_exit2 else []
                    for track in tracks_exit2:
                        if not track.is_confirmed() or track.time_since_update > 1:
                            continue
                        self.track_id_exit2 = track.track_id
                        if self.exit2_max_ID < int(self.track_id_exit2):# 更新出口2区域最大ID
                            self.exit2_max_ID = int(self.track_id_exit2)
                        bbox_exit2 = track.to_tlbr()
                        # cv2.rectangle(frame, (int(bbox_exit2[0]), int(bbox_exit2[1])), (int(bbox_exit2[2]), int(bbox_exit2[3]), (255, 0, 0), 2))
                        cv2.putText(frame, f'ID: {self.track_id_exit2}', (int(bbox_exit2[0]), int(bbox_exit2[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)
                


                # 计算流量
                elapsed_time = time.time() - self.start_time
                if elapsed_time > 0:
                    self.detect_alltime = elapsed_time - self.stop_longtime
                    self.flow_rate = self.enter_egg / self.detect_alltime # 根据入口区域的鸡蛋数量计算流量
                    
                    print(f"流量: {self.flow_rate:.2f} 个鸡蛋/秒")
                    self.time_label.setText(f'已检测时间: {self.detect_alltime:.2f} 秒')
                    self.flow_rate_label.setText(f'流量: {self.flow_rate:.2f} 个鸡蛋/秒') 

                # 记录当前鸡蛋数量
                
                self.enter_egg = self.enter_max_ID
                self.exit1_egg = self.exit1_max_ID
                self.exit2_egg = self.exit2_max_ID
                self.count_egg = self.enter_max_ID - self.exit1_max_ID - self.exit2_max_ID
                # 更新显示
                print(f"当前传送带中鸡蛋数量: {self.count_egg}")
                print(f"入口区域鸡蛋数量: {self.enter_egg}")
                print(f"出口1区域鸡蛋数量: {self.exit1_egg}")
                print(f"出口2区域鸡蛋数量: {self.exit2_egg}")
                self.egg_count_label.setText(f'当前传送带中鸡蛋数量: {self.count_egg}')
                self.egg_enter_label.setText(f'入口区域鸡蛋数量: {self.enter_egg}')
                self.egg_exit1_label.setText(f'出口1区域鸡蛋数量: {self.exit1_egg}')
                self.egg_exit2_label.setText(f'出口2区域鸡蛋数量: {self.exit2_egg}')
                self.display_frame(frame)  # 显示帧
                if self.zone_1 == 0:
                    self.display_frame1(frame1)  # 显示入口区域帧
                if self.zone_2 == 0:
                    self.display_frame2(frame2)  # 显示出口1区域帧
                if self.zone_3 == 0:
                    self.display_frame3(frame3)  # 显示出口2区域帧
                

            except ValueError as e:
                print(f"检测更新时发生ValueError错误: {e}")
            except Exception as e:
                print(f"检测更新时发生未知错误: {e}")


    def display_frame(self, frame):
        try:
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            p = convert_to_Qt_format.scaled(self.label.width(), self.label.height(), Qt.KeepAspectRatio)
            self.label.setPixmap(QPixmap.fromImage(p))
        except Exception as e:
            print(f"显示帧时发生错误: {e}")

    def display_frame1(self, frame):
        try:
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            p = convert_to_Qt_format.scaled(self.label_frame1.width(), self.label_frame1.height(), Qt.KeepAspectRatio)
            self.label_frame1.setPixmap(QPixmap.fromImage(p))
        except Exception as e:
            print(f"显示帧时发生错误: {e}")
    
    def display_frame2(self, frame):
        try:
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            p = convert_to_Qt_format.scaled(self.label_frame2.width(), self.label_frame2.height(), Qt.KeepAspectRatio)
            self.label_frame2.setPixmap(QPixmap.fromImage(p))
        except Exception as e:
            print(f"显示帧时发生错误: {e}")
    
    def display_frame3(self, frame):
        try:
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            p = convert_to_Qt_format.scaled(self.label_frame3.width(), self.label_frame3.height(), Qt.KeepAspectRatio)
            self.label_frame3.setPixmap(QPixmap.fromImage(p))
        except Exception as e:
            print(f"显示帧时发生错误: {e}")

    def calculate_speed(self, positions):
        if len(positions) < 2:
            return 0  # 如果没有足够的点来计算速度

        (x1, y1), t1 = positions[-2]  # 倒数第二帧的位置和时间
        (x2, y2), t2 = positions[-1]  # 最后一帧的位置和时间

        distance = sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        time_diff = t2 - t1

        if time_diff == 0:
            return 0  # 防止时间差为零时除零错误

        speed = distance / time_diff
        return speed

if __name__ == '__main__':
    app = QApplication(sys.argv)
    demo = VideoDemo()
    demo.show()
    sys.exit(app.exec_())
