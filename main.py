import os

import cv2
import pyautogui

from data_process_dnn import GazeDataset
from gaze_feature_extractor_mac import GazeFeatureExtractor
from global_info import GlobalInfo
from gui import run_gui
from train_and_predict_dnn import GazeController

# 确保数据目录存在（打包后也可靠）
os.makedirs(GlobalInfo.path_dir, exist_ok=True)

# init
GlobalInfo.train_and_predict_instance = GazeController()
GlobalInfo.video_steam = cv2.VideoCapture(0)
GlobalInfo.train_data = GazeDataset()
GlobalInfo.screen_width, GlobalInfo.screen_height = pyautogui.size()
GlobalInfo.gaze_feature_extractor = GazeFeatureExtractor()

# start gui
run_gui()
