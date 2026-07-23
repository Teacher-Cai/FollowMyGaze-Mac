import cv2
import mediapipe as mp
import numpy as np
import math


class GazeFeatureExtractor:
    def __init__(self):
        # 初始化MediaPipe面部检测和面部网格
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,  # 设置为1，只检测最大人脸
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

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

    def calculate_head_pose(self, image, face_landmarks):
        """计算头部姿态角"""
        height, width = image.shape[:2]

        # 获取2D图像点
        image_points = []
        for idx in [1, 152, 263, 33, 308, 78]:  # 对应特征点索引
            landmark = face_landmarks.landmark[idx]
            x, y = int(landmark.x * width), int(landmark.y * height)
            image_points.append([x, y])

        image_points = np.array(image_points, dtype=np.float64)

        # 相机参数估算
        focal_length = width
        center = (width / 2, height / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)

        dist_coeffs = np.zeros((4, 1))

        # Solve PnP
        success, rotation_vector, translation_vector = cv2.solvePnP(
            self.model_points,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if success:
            # 将旋转向量转换为欧拉角
            rmat, _ = cv2.Rodrigues(rotation_vector)
            pitch, yaw, roll = self.rotation_matrix_to_euler_angles(rmat)

            return {
                'yaw': yaw,
                'pitch': pitch,
                'roll': roll,
                'rotation_vector': rotation_vector,
                'translation_vector': translation_vector
            }
        return None

    def rotation_matrix_to_euler_angles(self, R):
        """将旋转矩阵转换为欧拉角"""
        sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

        singular = sy < 1e-6

        if not singular:
            x = math.atan2(R[2, 1], R[2, 2])
            y = math.atan2(-R[2, 0], sy)
            z = math.atan2(R[1, 0], R[0, 0])
        else:
            x = math.atan2(-R[1, 2], R[1, 1])
            y = math.atan2(-R[2, 0], sy)
            z = 0

        # 转换为度数
        x = math.degrees(x)
        y = math.degrees(y)
        z = math.degrees(z)

        return x, y, z

    def euclidean_distance(self, point1, point2):
        """计算两点间欧氏距离"""
        return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

    def get_eye_features(self, face_landmarks, image_shape):
        """获取眼部特征点"""
        height, width = image_shape[:2]

        # 左眼特征
        left_eye_points = []
        for idx in self.LEFT_EYE_INDICES:
            landmark = face_landmarks.landmark[idx]
            x, y = int(landmark.x * width), int(landmark.y * height)
            left_eye_points.append([x, y])

        # 右眼特征
        right_eye_points = []
        for idx in self.RIGHT_EYE_INDICES:
            landmark = face_landmarks.landmark[idx]
            x, y = int(landmark.x * width), int(landmark.y * height)
            right_eye_points.append([x, y])

        # 瞳孔中心
        left_pupil = face_landmarks.landmark[self.LEFT_PUPIL_INDEX]
        left_pupil_x, left_pupil_y = int(left_pupil.x * width), int(left_pupil.y * height)

        right_pupil = face_landmarks.landmark[self.RIGHT_PUPIL_INDEX]
        right_pupil_x, right_pupil_y = int(right_pupil.x * width), int(right_pupil.y * height)

        # 虹膜关键点
        left_iris_points = []
        for idx in self.LEFT_IRIS_INDICES:
            landmark = face_landmarks.landmark[idx]
            x, y = int(landmark.x * width), int(landmark.y * height)
            left_iris_points.append([x, y])

        right_iris_points = []
        for idx in self.RIGHT_IRIS_INDICES:
            landmark = face_landmarks.landmark[idx]
            x, y = int(landmark.x * width), int(landmark.y * height)
            right_iris_points.append([x, y])

        # 眼球中心（近似为眼睑中点）
        left_eye_center = (
            (left_eye_points[0][0] + left_eye_points[3][0]) // 2,
            (left_eye_points[0][1] + left_eye_points[3][1]) // 2
        )
        right_eye_center = (
            (right_eye_points[0][0] + right_eye_points[3][0]) // 2,
            (right_eye_points[0][1] + right_eye_points[3][1]) // 2
        )

        # 眼睛宽度
        left_eye_width = self.euclidean_distance(left_eye_points[0], left_eye_points[3])
        right_eye_width = self.euclidean_distance(right_eye_points[0], right_eye_points[3])


        return {
            'left_eye_points': left_eye_points,
            'right_eye_points': right_eye_points,
            'left_pupil_center': (left_pupil_x, left_pupil_y),
            'right_pupil_center': (right_pupil_x, right_pupil_y),
            'left_iris_points': left_iris_points,
            'right_iris_points': right_iris_points,
            'left_eye_center': left_eye_center,
            'right_eye_center': right_eye_center
        }

    def calculate_face_position_and_area(self, face_landmarks, image_shape):
        """计算人脸位置和面积"""
        height, width = image_shape[:2]

        # 获取所有人脸关键点的坐标
        x_coords = [lm.x * width for lm in face_landmarks.landmark]
        y_coords = [lm.y * height for lm in face_landmarks.landmark]

        # 计算边界框
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)

        # 计算面积（像素）
        area = (x_max - x_min) * (y_max - y_min)

        # 计算人脸中心位置
        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2

        # 计算相对于图像的比例位置
        relative_center_x = center_x / width
        relative_center_y = center_y / height

        # 计算人脸宽度和高度
        face_width = x_max - x_min
        face_height = y_max - y_min

        # 计算相对大小（相对于图像尺寸）
        relative_width = face_width / width
        relative_height = face_height / height

        return {
            'bbox': (int(x_min), int(y_min), int(x_max), int(y_max)),
            'area_pixels': int(area),
            'center': (int(center_x), int(center_y)),
            'relative_center': (relative_center_x, relative_center_y),
            'width': int(face_width),
            'height': int(face_height),
            'relative_size': (relative_width, relative_height)
        }

    def find_largest_face(self, multi_face_landmarks, image_shape):
        """找到最大的人脸"""
        if not multi_face_landmarks:
            return None

        height, width = image_shape[:2]
        largest_face = None
        max_area = 0

        for face_landmarks in multi_face_landmarks:
            # 计算人脸边界框
            x_coords = [lm.x for lm in face_landmarks.landmark]
            y_coords = [lm.y for lm in face_landmarks.landmark]

            x_min, x_max = min(x_coords) * width, max(x_coords) * width
            y_min, y_max = min(y_coords) * height, max(y_coords) * height

            area = (x_max - x_min) * (y_max - y_min)

            if area > max_area:
                max_area = area
                largest_face = face_landmarks

        return largest_face

    def process_frame(self, image):
        """处理单帧图像，仅处理最大人脸"""
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_image)

        head_pose_data = None
        eye_features = None
        face_position_data = None

        if results.multi_face_landmarks:
            # 找到最大的人脸
            largest_face = self.find_largest_face(results.multi_face_landmarks, image.shape)

            if largest_face:
                # 计算头部姿态
                head_pose_data = self.calculate_head_pose(image, largest_face)

                # 获取眼部特征
                eye_features = self.get_eye_features(largest_face, image.shape)

                # 计算人脸位置和面积
                face_position_data = self.calculate_face_position_and_area(largest_face, image.shape)

                # 绘制面部网格
                self.mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=largest_face,
                    connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing_styles
                    .get_default_face_mesh_tesselation_style())

                self.mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=largest_face,
                    connections=self.mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing_styles
                    .get_default_face_mesh_contours_style())

        return image, head_pose_data, eye_features, face_position_data

    def extract_features_from_image(self, image):
        """从图像中提取全部特征"""
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_image)

        if not results.multi_face_landmarks:
            return None

        # 找到最大的人脸
        largest_face = self.find_largest_face(results.multi_face_landmarks, image.shape)

        if largest_face:
            face_landmarks = largest_face
            height, width = image.shape[:2]

            # 提取眼部关键点
            left_eye_points = []
            for idx in self.LEFT_EYE_INDICES:
                landmark = face_landmarks.landmark[idx]
                x, y = landmark.x * width, landmark.y * height
                left_eye_points.append([x, y])

            right_eye_points = []
            for idx in self.RIGHT_EYE_INDICES:
                landmark = face_landmarks.landmark[idx]
                x, y = landmark.x * width, landmark.y * height
                right_eye_points.append([x, y])

            # 提取虹膜关键点
            left_iris_points = []
            for idx in self.LEFT_IRIS_INDICES:
                landmark = face_landmarks.landmark[idx]
                x, y = landmark.x * width, landmark.y * height
                left_iris_points.append([x, y])

            right_iris_points = []
            for idx in self.RIGHT_IRIS_INDICES:
                landmark = face_landmarks.landmark[idx]
                x, y = landmark.x * width, landmark.y * height
                right_iris_points.append([x, y])

            # 计算特征向量
            features = []

            # 眼部几何特征
            left_eye_array = np.array(left_eye_points)
            right_eye_array = np.array(right_eye_points)

            # 眼宽和眼高
            left_eye_width = np.max(left_eye_array[:, 0]) - np.min(left_eye_array[:, 0])
            left_eye_height = np.max(left_eye_array[:, 1]) - np.min(left_eye_array[:, 1])
            right_eye_width = np.max(right_eye_array[:, 0]) - np.min(right_eye_array[:, 0])
            right_eye_height = np.max(right_eye_array[:, 1]) - np.min(right_eye_array[:, 1])

            features.extend([left_eye_width, left_eye_height, right_eye_width, right_eye_height])

            # 虹膜位置特征
            left_iris_center = np.mean(left_iris_points, axis=0)
            right_iris_center = np.mean(right_iris_points, axis=0)

            # 瞳孔与眼睑相对位置
            left_eye_center = np.mean(left_eye_array, axis=0)
            right_eye_center = np.mean(right_eye_array, axis=0)

            left_pupil_offset = left_iris_center - left_eye_center
            right_pupil_offset = right_iris_center - right_eye_center

            features.extend([
                left_pupil_offset[0], left_pupil_offset[1],
                right_pupil_offset[0], right_pupil_offset[1]
            ])

            # 眼部关键点相对位置
            for point in left_eye_points:
                normalized_point = [(point[0] - left_eye_center[0]) / left_eye_width,
                                    (point[1] - left_eye_center[1]) / left_eye_height]
                features.extend(normalized_point)

            for point in right_eye_points:
                normalized_point = [(point[0] - right_eye_center[0]) / right_eye_width,
                                    (point[1] - right_eye_center[1]) / right_eye_height]
                features.extend(normalized_point)

            # 头部信息
            head_pose = self.calculate_head_pose(image, largest_face)
            features.extend([head_pose['yaw'], head_pose['pitch'], head_pose['roll']])

            face_position_data = self.calculate_face_position_and_area(largest_face, image.shape)
            features.extend(face_position_data['bbox'])
            features.extend(face_position_data['relative_size'])

            return np.array(features)


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
