import time
from pynput import keyboard
from threading import Thread, Event

# 定义要监听的键和目标动作
TARGET_KEY = keyboard.Key.space  # 以空格键为例
action_interval = 0.1  # 执行动作的间隔（秒）

# 用于控制动作循环的事件
stop_action_event = Event()
action_thread = None

def your_action():
    """这里定义长按时要重复执行的动作"""
    print("执行动作...")

def action_loop():
    """在后台线程中循环执行动作，直到收到停止信号"""
    while not stop_action_event.is_set():
        your_action()
        time.sleep(action_interval)

def on_press(key):
    global action_thread
    # 检查按下的键是否为目标键，且动作循环尚未启动
    if key == TARGET_KEY and (action_thread is None or not action_thread.is_alive()):
        print(f"检测到 {key} 键按下，开始执行动作")
        stop_action_event.clear()  # 重置停止事件
        # 启动后台线程执行动作循环
        action_thread = Thread(target=action_loop)
        action_thread.start()

def on_release(key):
    # 检查释放的键是否为目标键
    if key == TARGET_KEY:
        print(f"{key} 键释放，停止动作")
        stop_action_event.set()  # 设置停止事件，终止动作循环
    # 按ESC键退出整个监听器
    if key == keyboard.Key.esc:
        return False

# 启动键盘监听器
with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
    listener.join()
