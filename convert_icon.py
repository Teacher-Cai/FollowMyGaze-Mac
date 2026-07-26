"""将 asset/aaa.jpg 转为各平台所需的图标格式。
用法：python convert_icon.py
产出（均在 asset/ 目录下）：
  - asset/icon.png   Tkinter 窗口图标
  - asset/icon.icns  macOS 应用图标 (需要 sips/iconutil)
  - asset/icon.ico   Windows 可执行文件图标
"""
import subprocess
import sys
from pathlib import Path

from PIL import Image


def convert_jpg_to_png(src: str, dst: str, size: tuple = (256, 256)) -> None:
    img = Image.open(src).convert("RGBA")
    img = img.resize(size, Image.LANCZOS)
    img.save(dst, "PNG")
    print(f"[OK] {dst}  ({size[0]}x{size[1]})")


def png_to_icns_macos(src_png: str, dst_icns: str) -> None:
    """macOS: 用 sips + iconutil 生成 .icns"""
    if sys.platform != "darwin":
        print("[跳过] .icns 只能在 macOS 上生成")
        return

    tmp_dir = Path("asset/icon.iconset")
    tmp_dir.mkdir(exist_ok=True)

    sizes = {
        "icon_16x16.png": 16,
        "icon_16x16@2x.png": 32,
        "icon_32x32.png": 32,
        "icon_32x32@2x.png": 64,
        "icon_128x128.png": 128,
        "icon_128x128@2x.png": 256,
        "icon_256x256.png": 256,
        "icon_256x256@2x.png": 512,
        "icon_512x512.png": 512,
        "icon_512x512@2x.png": 1024,
    }

    img = Image.open(src_png)
    for name, size in sizes.items():
        resized = img.resize((size, size), Image.LANCZOS)
        resized.save(tmp_dir / name, "PNG")

    subprocess.run(["iconutil", "-c", "icns", str(tmp_dir), "-o", dst_icns], check=True)

    # 清理临时目录
    import shutil

    shutil.rmtree(tmp_dir)
    print(f"[OK] {dst_icns}")


def png_to_ico(src_png: str, dst_ico: str) -> None:
    """Windows: 用 PIL 生成 .ico"""
    img = Image.open(src_png)
    # 保存为多尺寸 ico
    sizes = [(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]
    img.save(dst_ico, format="ICO", sizes=sizes)
    print(f"[OK] {dst_ico}")


if __name__ == "__main__":
    src = "asset/aaa.jpg"
    if not Path(src).exists():
        print(f"[错误] 找不到 {src}")
        sys.exit(1)

    # 1. PNG (Tkinter runtime)
    convert_jpg_to_png(src, "asset/icon.png", size=(256, 256))

    # 2. ICNS (macOS app bundle)
    png_to_icns_macos("asset/icon.png", "asset/icon.icns")

    # 3. ICO (Windows exe)
    png_to_ico("asset/icon.png", "asset/icon.ico")

    print("\n完成！各平台图标文件已生成。")