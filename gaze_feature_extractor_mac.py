import cv2
import mediapipe as mp
import numpy as np
import math
import time
from utils import resource_path

# 单调递增时间戳，避免 detect_async 报 ValueError
_last_timestamp_ms = 0


class GazeFeatureExtractor:
    def __init__(self):
        # 初始化MediaPipe面部检测和面部网格
        # 配置基础选项
        self.BaseOptions = mp.tasks.BaseOptions
        self.FaceLandmarker = mp.tasks.vision.FaceLandmarker
        self.FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
        self.VisionRunningMode = mp.tasks.vision.RunningMode
        self.target_idx = 0
        self.face_data = None
        self.options = self.FaceLandmarkerOptions(
            base_options=self.BaseOptions(model_asset_path=resource_path('face_landmarker.task')),
            running_mode=self.VisionRunningMode.LIVE_STREAM,
            result_callback=self.callback_result,
            output_facial_transformation_matrixes=True,  # 必须开启以获取角度
            num_faces=1
        )
        self.landmarker = self.FaceLandmarker.create_from_options(self.options)

        # 3D模型点（标准化坐标）
        self.model_points = np.array([
            [0.0, 0.0, 0.0],  # 鼻尖
            [0.0, -330.0, -65.0],  # 下巴
            [-225.0, 170.0, -135.0],  # 左眼角
            [225.0, 170.0, -135.0],  # 右眼角
            [-150.0, -150.0, -125.0],  # 左嘴角
            [150.0, -150.0, -125.0]  # 右嘴角
        ], dtype=np.float64)

        # 眼部关键点索引
        self.LEFT_EYE_INDICES = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        self.RIGHT_EYE_INDICES = [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466]
        self.LEFT_PUPIL_INDEX = 468
        self.RIGHT_PUPIL_INDEX = 473
        self.LEFT_IRIS_INDICES = [468, 469, 470, 471, 472]
        self.RIGHT_IRIS_INDICES = [473, 474, 475, 476, 477]

    def callback_result(self, result, output_image, timestamp_ms):

        if not result.face_landmarks:
            return

        selected_lms = result.face_landmarks[self.target_idx]

        # 提取变换矩阵（用于姿态角）
        matrix = None
        if result.facial_transformation_matrixes:
            matrix = result.facial_transformation_matrixes[self.target_idx]

        self.face_data = {
            "landmarks": selected_lms,
            "matrix": matrix,
        }

    def extract_features_from_image(self, image):
        """从图像中提取全部特征"""
        global _last_timestamp_ms
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        # 保证时间戳严格单调递增，防止 detect_async 报 ValueError
        ts = int(time.time() * 1000)
        if ts <= _last_timestamp_ms:
            ts = _last_timestamp_ms + 1
        _last_timestamp_ms = ts
        self.landmarker.detect_async(mp_image, ts)

        if not self.face_data:
            return None
        h, w, _ = image.shape
        lms = self.face_data["landmarks"]
        mat = self.face_data["matrix"]

        # 计算特征向量
        features = []

        # A. 人脸位置 (像素单位)
        px_coords = np.array([(lm.x * w, lm.y * h) for lm in lms])
        xmin, ymin = np.min(px_coords, axis=0).astype(int)
        xmax, ymax = np.max(px_coords, axis=0).astype(int)
        face_area = (ymax - ymin) * (xmax - xmin)

        features.extend([xmin, ymin, xmax, ymax, face_area])  # idx 0..4

        # B. 姿态角提取 (基于矩阵分解，单位度)
        if mat is not None:
            # 俯仰角 Pitch (绕X轴)
            pitch = np.arcsin(-mat[1][2]) * (180 / np.pi)
            # 偏航角 Yaw (绕Y轴)
            yaw = np.arctan2(mat[0][2], mat[2][2]) * (180 / np.pi)
            # 翻滚角 Roll (绕Z轴)
            roll = np.arctan2(mat[1][0], mat[1][1]) * (180 / np.pi)

        features.extend([pitch, yaw, roll])  # idx 5..7

        # C. 虹膜与眼睛相对位置
        # 468为左虹膜中心，33为左眼左外角，133为左眼右内角
        iris_l = lms[468]
        eye_l_outer = lms[33]
        eye_l_inner = lms[133]
        eye_l_top = lms[159]    # 左眼上眼睑
        eye_l_bot = lms[145]    # 左眼下眼睑
        eye_l_width = abs(eye_l_inner.x - eye_l_outer.x)
        eye_l_height = abs(eye_l_top.y - eye_l_bot.y)

        eye_l_mid_x = (eye_l_inner.x + eye_l_outer.x) / 2
        eye_l_mid_y = (eye_l_inner.y + eye_l_outer.y) / 2
        eye_l_rel_x = iris_l.x - eye_l_mid_x
        eye_l_rel_y = iris_l.y - eye_l_mid_y
        eye_l_openness = eye_l_height / (eye_l_width + 1e-6)  # 眼睛开合度

        # 方案一：虹膜在眼眶内的位置比例（不归一化，保留原始比值）
        eye_l_ratio_x = (iris_l.x - eye_l_outer.x) / (eye_l_inner.x - eye_l_outer.x + 1e-6)
        eye_l_ratio_y = (iris_l.y - eye_l_top.y) / (eye_l_bot.y - eye_l_top.y + 1e-6)

        # 方案三：虹膜可见面积比（虹膜边缘点：469=left, 470=top, 471=right, 472=bottom）
        iris_l_left   = lms[469]
        iris_l_top_pt = lms[470]
        iris_l_right  = lms[471]
        iris_l_bot_pt = lms[472]
        iris_l_vis_w = abs(iris_l_left.x - iris_l_right.x)
        iris_l_vis_h = abs(iris_l_top_pt.y - iris_l_bot_pt.y)
        iris_l_aspect = iris_l_vis_h / (iris_l_vis_w + 1e-6)

        features.extend([eye_l_rel_x, eye_l_rel_y, iris_l.x, iris_l.y,  # idx 8..11
                         eye_l_openness,  # idx 12
                         eye_l_ratio_x, eye_l_ratio_y,  # idx 13..14
                         iris_l_aspect])  # idx 15

        iris_r = lms[473]
        eye_r_outer = lms[263]
        eye_r_inner = lms[362]
        eye_r_top = lms[386]    # 右眼上眼睑
        eye_r_bot = lms[374]    # 右眼下眼睑
        eye_r_width = abs(eye_r_inner.x - eye_r_outer.x)
        eye_r_height = abs(eye_r_top.y - eye_r_bot.y)

        eye_r_mid_x = (eye_r_inner.x + eye_r_outer.x) / 2
        eye_r_mid_y = (eye_r_inner.y + eye_r_outer.y) / 2
        eye_r_rel_x = iris_r.x - eye_r_mid_x
        eye_r_rel_y = iris_r.y - eye_r_mid_y
        eye_r_openness = eye_r_height / (eye_r_width + 1e-6)

        # 方案一：右眼
        eye_r_ratio_x = (iris_r.x - eye_r_outer.x) / (eye_r_inner.x - eye_r_outer.x + 1e-6)
        eye_r_ratio_y = (iris_r.y - eye_r_top.y) / (eye_r_bot.y - eye_r_top.y + 1e-6)

        # 方案三：右眼（474=left, 475=top, 476=right, 477=bottom）
        iris_r_left   = lms[474]
        iris_r_top_pt = lms[475]
        iris_r_right  = lms[476]
        iris_r_bot_pt = lms[477]
        iris_r_vis_w = abs(iris_r_left.x - iris_r_right.x)
        iris_r_vis_h = abs(iris_r_top_pt.y - iris_r_bot_pt.y)
        iris_r_aspect = iris_r_vis_h / (iris_r_vis_w + 1e-6)

        features.extend([eye_r_rel_x, eye_r_rel_y, iris_r.x, iris_r.y,  # idx 16..19
                         eye_r_openness,  # idx 20
                         eye_r_ratio_x, eye_r_ratio_y,  # idx 21..22
                         iris_r_aspect])  # idx 23

        # 双眼虹膜水平偏移差（辐辏信号）
        features.append(eye_l_rel_x - eye_r_rel_x)  # idx 24

        # ============ 新增：3D 视线方向向量（利用 MediaPipe 提供的 z 坐标） ============
        # 左眼 3D 视线方向：虹膜中心 - 眼眶 3D 中心
        eye_l_center_3d = np.array([
            (eye_l_inner.x + eye_l_outer.x) / 2,
            (eye_l_inner.y + eye_l_outer.y) / 2,
            (eye_l_inner.z + eye_l_outer.z) / 2,
        ])
        iris_l_3d = np.array([iris_l.x, iris_l.y, iris_l.z])
        gaze_vec_l = iris_l_3d - eye_l_center_3d
        gaze_vec_l_norm = gaze_vec_l / (np.linalg.norm(gaze_vec_l) + 1e-6)
        features.extend(gaze_vec_l_norm.tolist())  # 26 27 28

        # 右眼 3D 视线方向
        eye_r_center_3d = np.array([
            (eye_r_inner.x + eye_r_outer.x) / 2,
            (eye_r_inner.y + eye_r_outer.y) / 2,
            (eye_r_inner.z + eye_r_outer.z) / 2,
        ])
        iris_r_3d = np.array([iris_r.x, iris_r.y, iris_r.z])
        gaze_vec_r = iris_r_3d - eye_r_center_3d
        gaze_vec_r_norm = gaze_vec_r / (np.linalg.norm(gaze_vec_r) + 1e-6)
        features.extend(gaze_vec_r_norm.tolist())  # 29 30 31

        # ============ 新增：瞳孔间距 IPD（3D） ============
        ipd_3d = float(np.linalg.norm(iris_l_3d - iris_r_3d))
        features.append(ipd_3d)  # 32

        # other features
        for i in (self.LEFT_EYE_INDICES + self.RIGHT_EYE_INDICES + self.LEFT_IRIS_INDICES + self.RIGHT_IRIS_INDICES +
                  [168, 4, 8, 9]):
            features.append(lms[i].x)
            features.append(lms[i].y)

        return np.array(features, dtype=np.float32)


def main():
    estimator = GazeFeatureExtractor()

    # 打开摄像头
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("无法获取摄像头画面")
            break

        # 处理图像
        processed_image, head_pose, eye_features, face_position = estimator.process_frame(image)

        # 显示头部姿态信息
        if head_pose:
            cv2.putText(processed_image, f"Yaw: {head_pose['yaw']:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(processed_image, f"Pitch: {head_pose['pitch']:.2f}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(processed_image, f"Roll: {head_pose['roll']:.2f}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # 显示眼部特征信息
        if eye_features:
            # 绘制左眼关键点
            for point in eye_features['left_eye_points']:
                cv2.circle(processed_image, tuple(point), 2, (255, 0, 0), -1)
            cv2.circle(processed_image, eye_features['left_pupil_center'], 3, (0, 0, 255), -1)

            # 绘制右眼关键点
            for point in eye_features['right_eye_points']:
                cv2.circle(processed_image, tuple(point), 2, (255, 0, 0), -1)
            cv2.circle(processed_image, eye_features['right_pupil_center'], 3, (0, 0, 255), -1)

            # 显示人脸位置和面积信息
            if face_position:
                center = face_position['center']
                area = face_position['area_pixels']
                rel_center = face_position['relative_center']
                rel_size = face_position['relative_size']

                # 绘制人脸中心点
                cv2.circle(processed_image, center, 5, (255, 255, 0), -1)

                # 显示详细信息
                cv2.putText(processed_image, f"Center: ({center[0]}, {center[1]})", (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                cv2.putText(processed_image, f"Relative Pos: ({rel_center[0]:.2f}, {rel_center[1]:.2f})", (10, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                cv2.putText(processed_image, f"Size: {face_position['width']}x{face_position['height']}", (10, 180),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                cv2.putText(processed_image, f"Relative Size: ({rel_size[0]:.2f}, {rel_size[1]:.2f})", (10, 210),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # 显示图像
        cv2.imshow('Head Pose Estimation - Largest Face Only', processed_image)

        if cv2.waitKey(5) & 0xFF == 27:  # ESC键退出
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
