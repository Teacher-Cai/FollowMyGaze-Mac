"""GUI 用户个性化配置持久化（轻量版）。

只保存 / 恢复 GUI 上用户可见可调的配置项：
  - 自动训练开关（checkbox）
  - 视线跳转触发距离 / 冷却时间
  - 视线跟随暂停时长 / 顺滑度
  - 视线滑翔加速倍数 / 减速起始距离 / 全速加速距离 / 减速陡峭度

注意：模式选择（radio button）不做持久化，每次启动都由用户重新选择。

文件位置：<path_dir>/user_config.json
"""

import json
import logging
import os

from global_info import GlobalInfo

logger = logging.getLogger(__name__)


def _config_path():
    base = getattr(GlobalInfo, "path_dir", "FollowMyGaze") or "FollowMyGaze"
    return os.path.join(base, "user_config.json")


def load_gui_config():
    """读取保存的 GUI 配置，返回 dict；文件不存在或损坏时返回 {}。"""
    path = _config_path()
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return {}
        return data
    except Exception:
        logger.exception("user_config: load failed")
        return {}


def save_gui_config(**kwargs):
    """将任意 GUI 配置项原子写入本地 JSON 文件。

    只序列化 JSON 原生类型（bool / int / float / str）。
    使用 kwargs 而不是固定签名，方便后续增删参数不用改函数签名。
    """
    path = _config_path()
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    except Exception:
        logger.exception("user_config: mkdir failed")
        return

    data = {}
    for k, v in kwargs.items():
        if isinstance(v, bool):
            data[k] = bool(v)
        elif isinstance(v, int):
            data[k] = int(v)
        elif isinstance(v, float):
            data[k] = float(v)
        elif isinstance(v, str):
            data[k] = str(v)
        # 其他类型忽略，避免 JSON 序列化错误

    tmp = path + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)
        os.replace(tmp, path)
        logger.info("user_config: saved -> %s (%d keys)", path, len(data))
    except Exception:
        logger.exception("user_config: save failed")
        try:
            os.remove(tmp)
        except Exception:
            pass
