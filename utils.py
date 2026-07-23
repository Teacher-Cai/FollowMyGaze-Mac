import threading
import tkinter as tk

import pyautogui
from PIL import Image
from pynput import keyboard
from pynput import mouse
import os
import sys
import queue
from appdirs import user_config_dir

from data_process_dnn import GazeDataset
from global_info import GlobalInfo
import time
from threading import Thread, Event


def resize_image(image, max_width, max_height):
    width, height = image.size
    aspect_ratio = width / height

    if width > max_width or height > max_height:
        if width / max_width > height / max_height:
            new_width = max_width
            new_height = int(max_width / aspect_ratio)
        else:
            new_height = max_height
            new_width = int(max_height * aspect_ratio)
        image = image.resize((new_width, new_height), Image.LANCZOS)

    return image


def show_red_dot(root=None):
    def toggle_dot():
        if hasattr(toggle_dot, "dot_id"):
            canvas.delete(toggle_dot.dot_id)  # 删除现有红点
            del toggle_dot.dot_id  # 清除属性
        else:
            # root_dot.attributes("-topmost", True)  # 置于所有窗口最上方

            x = GlobalInfo.red_dot_x  # 20 是红点直径
            y = GlobalInfo.red_dot_y  # 距离顶部 10 像素
            toggle_dot.dot_id = canvas.create_oval(x, y, x + 20, y + 20, fill="red")
            canvas.lift(toggle_dot.dot_id)  # 确保红点在最上层

        if GlobalInfo.show_red_dot_win:
            root_dot.after(100, toggle_dot)
        else:
            root_dot.destroy()

    # 创建透明、无边框的顶层窗口
    root_dot = tk.Toplevel(root)
    # root_dot.overrideredirect(True)  # 去除窗口边框和标题栏
    root_dot.attributes("-topmost", True)  # 置于所有窗口最上方
    root_dot.attributes("-alpha", 0.7)  # 设置红点的透明度， 1不透明
    # root_dot.attributes("-transparentcolor", "white")  # 设置白色为透明色
    root_dot.attributes('-fullscreen', True)  # 标准全屏
    # screen_width, screen_height = pyautogui.size()
    # root_dot.geometry(f"{screen_width}x{screen_height}")

    # 创建画布并绘制红点
    canvas = tk.Canvas(root_dot, width=GlobalInfo.screen_width, height=GlobalInfo.screen_height, bg='white', highlightthickness=0)
    canvas.pack()
    toggle_dot()


def predict_and_draw_pot(frame):
    if GlobalInfo.train_and_predict_instance is None:
        return
    if not GlobalInfo.train_and_predict_instance.is_predict and frame is not None:
        result = GlobalInfo.train_and_predict_instance.predict_gaze(frame)
        if result is None:
            return
        x, y = result

        # —— 红点位置更新（窗口的生死由 GUI 的 RedDotOverlay 控制） ——
        GlobalInfo.red_dot_x = x
        GlobalInfo.red_dot_y = y

        # —— 模式：silent_train / move_cursor ——
        mode_selected = GlobalInfo.mode_select.get() if GlobalInfo.mode_select is not None else 'silent_train'
        if mode_selected == 'move_cursor':
            GlobalInfo.enable_move_cursor = True
            GlobalInfo.move_cursor_x = x
            GlobalInfo.move_cursor_y = y
        else:
            GlobalInfo.enable_move_cursor = False


def start_listening_click():
    def on_click(x, y, button, pressed):
        if pressed and button == mouse.Button.left and GlobalInfo.current_frame is not None:
            coords = [float(x), float(y)]
            GlobalInfo.train_data.save_sample(GlobalInfo.current_frame, coords)
            print(coords, "saving sample...")

            if len(GlobalInfo.train_data) >= 16:
                train_thread = threading.Thread(target=GlobalInfo.train_and_predict_instance.train_model)
                train_thread.daemon = False
                train_thread.start()

    listener = mouse.Listener(on_click=on_click)
    listener.start()


def start_listening_click_dnn():
    """启动全局鼠标点击监听器，用于采集训练样本。

    设计要点：
    - on_click 回调【瞬间返回】：只把样本入队，重活（save_sample / pandas 拼接）
      全部交给 worker 线程处理。这从根源上避免回调过慢触发 CGEventTap
      的【超时禁用】(kCGEventTapDisabledByTimeout)。
    - 单一长存监听器：正常授予「输入监控」权限后，CGEventTap 会长期稳定运行，
      无需定期重建。注意 CGEventTap 挂在【全局事件流】上，系统每个鼠标事件都要
      流经它——若同时存在多个 tap（例如反复重建而旧的未彻底释放），会拖慢整个
      系统的输入响应（表现为"用久了全电脑变卡，一关程序就恢复"）。故这里只创建一个。
    """
    sample_q = queue.Queue()

    def on_click(x, y, button, pressed):
        # 极简回调：只入队，绝不做重操作，防止 CGEventTap 超时被禁用
        if pressed and button == mouse.Button.left and GlobalInfo.current_features is not None:
            sample_q.put((GlobalInfo.current_features, [float(x), float(y)]))

    def sample_worker():
        while True:
            features, coords = sample_q.get()
            try:
                GlobalInfo.train_data.save_sample(features, coords)
            except Exception:
                pass

    Thread(target=sample_worker, daemon=True, name="SampleWorker").start()
    # 单一监听器，长期存活；daemon 确保主程序退出时随之销毁 tap
    listener = mouse.Listener(on_click=on_click)
    listener.daemon = True
    listener.start()


TARGET_KEY = keyboard.Key.alt

# 用于控制动作循环的事件
stop_action_event = Event()
action_thread = None


def action_loop():
    """在后台线程中循环执行动作，直到收到停止信号"""
    while not stop_action_event.is_set():
        pyautogui.moveTo(GlobalInfo.move_cursor_x, GlobalInfo.move_cursor_y)
        time.sleep(0.05)


def move_cursor_when_press_key():
    def on_press(key):
        global action_thread
        if key == keyboard.Key.alt and GlobalInfo.enable_move_cursor\
            and (action_thread is None or not action_thread.is_alive()):
            stop_action_event.clear()  # 重置停止事件
            # 启动后台线程执行动作循环
            action_thread = Thread(target=action_loop)
            action_thread.start()

    def on_release(key):
        # 检查释放的键是否为目标键
        if key == TARGET_KEY:
            stop_action_event.set()  # 设置停止事件，终止动作循环

    def start_keyboard_listener():
        with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
            listener.join()

    listener_thread = threading.Thread(target=start_keyboard_listener)
    listener_thread.daemon = True  # 设置为守护线程（主线程退出时自动结束）
    listener_thread.start()


def resource_path(relative_path):
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)


if __name__ == "__main__":
    move_cursor_when_press_key()
