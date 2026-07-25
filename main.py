import os

import cv2
import pyautogui

from data_process_dnn import GazeDataset
from gaze_feature_extractor_mac import GazeFeatureExtractor
from global_info import GlobalInfo
from gui import run_gui
from train_and_predict_dnn import GazeController
from user_config import load_gui_config

# 确保数据目录存在（打包后也可靠）
os.makedirs(GlobalInfo.path_dir, exist_ok=True)

# init
GlobalInfo.train_and_predict_instance = GazeController()
# 启动时先读用户上次保存的摄像头索引（多摄像头设备），避免先开 0 再重开的浪费
try:
    GlobalInfo.camera_index = int(load_gui_config().get("camera_index", 0))
except Exception:
    GlobalInfo.camera_index = 0
GlobalInfo.video_steam = cv2.VideoCapture(GlobalInfo.camera_index)
GlobalInfo.train_data = GazeDataset()
GlobalInfo.screen_width, GlobalInfo.screen_height = pyautogui.size()
GlobalInfo.gaze_feature_extractor = GazeFeatureExtractor()

# start gui
run_gui()
