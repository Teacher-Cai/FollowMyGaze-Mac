"""
删除 FollowMyGaze/samples/samples.pkl 的最后 100 个样本。

用法：
    python test.py            # 删除最后 100 条
    python test.py 50         # 删除最后 50 条

行为：
- 读取 pkl（pandas DataFrame）
- 丢弃末尾 N 行
- 覆盖回原文件
- 覆盖前先做一次备份 samples.pkl.bak
"""

import os
import shutil
import sys

import pandas as pd


DEFAULT_TAIL_DROP = 100
SAMPLES_PATH = os.path.join("/Users/caijiawei/FollowMyGaze", "samples", "samples.pkl")


def main():
    n = DEFAULT_TAIL_DROP
    if len(sys.argv) > 1:
        try:
            n = int(sys.argv[1])
        except ValueError:
            print(f"参数无效: {sys.argv[1]}，使用默认值 {DEFAULT_TAIL_DROP}")
            n = DEFAULT_TAIL_DROP

    if n <= 0:
        print(f"删除数量必须为正整数，当前为 {n}，退出。")
        return

    if not os.path.exists(SAMPLES_PATH):
        print(f"样本文件不存在: {SAMPLES_PATH}")
        return

    df = pd.read_pickle(SAMPLES_PATH)
    total = len(df)
    print(f"当前样本总数: {total}")

    if total == 0:
        print("样本为空，无需删除。")
        return

    if n >= total:
        print(f"要删除的数量 {n} >= 样本总数 {total}，将清空文件（保留 0 条）。")
        df_new = df.iloc[0:0]
    else:
        df_new = df.iloc[:-n].reset_index(drop=True)

    # 备份
    backup_path = SAMPLES_PATH + ".bak"
    try:
        shutil.copy2(SAMPLES_PATH, backup_path)
        print(f"已备份原文件到: {backup_path}")
    except Exception as e:
        print(f"备份失败: {e}，仍继续覆盖")

    df_new.to_pickle(SAMPLES_PATH)
    print(f"已删除最后 {min(n, total)} 条，剩余样本数: {len(df_new)}")


if __name__ == "__main__":
    main()
