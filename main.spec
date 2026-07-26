# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_data_files

datas = [('face_landmarker.task', '.'),
         ('asset/icon.png', 'asset')]
datas += collect_data_files('mediapipe')
datas += collect_data_files('mediapipe.tasks')


a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='FollowMyGaze',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
app = BUNDLE(
    exe,
    name='FollowMyGaze.app',
    icon='asset/icon.icns',
    bundle_identifier='com.followmygaze.app',
    info_plist={
        'NSCameraUsageDescription':
            'FollowMyGaze 需要使用摄像头来追踪您的视线，用于预测注视位置并控制鼠标。',
        'NSMicrophoneUsageDescription':
            'FollowMyGaze 不会主动录音，此权限仅为兼容底层库需求。',
        'NSHighResolutionCapable': True,
        # 让应用以普通窗口形式出现在 Dock 与前台
        'LSUIElement': False,
        'CFBundleName': 'FollowMyGaze',
        'CFBundleDisplayName': 'FollowMyGaze',
    },
)
