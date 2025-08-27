#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
截图模块，根据Excel表格中的时间信息对视频进行截图
"""

import subprocess
import argparse
from pathlib import Path
import pandas as pd
import sys
import json
from typing import List, Union


def get_script_dir() -> Path:
    """
    获取脚本所在的目录路径
    返回：Path对象，表示脚本所在目录
    """
    return Path(sys.argv[0]).parent.absolute()


def check_ffmpeg() -> bool:
    """
    验证系统是否安装FFmpeg
    返回: True/False
    """
    try:
        # 尝试执行ffmpeg -version命令验证安装，修复编码问题
        subprocess.run(['ffmpeg', '-version'],
                       stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE,
                       check=True,
                       encoding='utf-8',
                       errors='ignore')
        return True
    except FileNotFoundError:
        print("错误: 未找到FFmpeg, 请确保: ")
        print("1. 已安装FFmpeg")
        print("2. 已添加到系统环境变量PATH")
        return False
    except Exception as e:
        print(f"FFmpeg验证时发生未知错误: {str(e)}")
        return False


def format_time_for_ffmpeg(seconds: float) -> str:
    """
    将秒数转换为FFmpeg可识别的时间格式(HH:MM:SS.mmm)
    """
    if seconds <= 0:
        return "00:00:00.000"

    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millisecs = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millisecs:03d}"


def parse_time_from_excel(time_str: str) -> float:
    """
    将Excel中的时间解析为秒数。
    支持两种格式：
    1) 新格式: HH:MM:SS.nnnnnnnnn（纳秒精度）
    2) 旧格式: H.MM.SS
    返回对应的秒数（浮点数）。
    """
    try:
        s = time_str.strip()
        # 新格式: HH:MM:SS.nnnnnnnnn 或 HH:MM:SS
        if ':' in s:
            hms, *frac = s.split('.')
            h, m, sec = hms.split(':')
            hours = int(h)
            minutes = int(m)
            seconds = int(sec)
            frac_str = frac[0] if frac else '0'
            frac_str = frac_str.ljust(9, '0')[:9]
            nanos = int(frac_str)
            return (hours * 3600 + minutes * 60 + seconds +
                    nanos / 1_000_000_000)
        # 旧格式: H.MM.SS
        parts = s.split('.')
        if len(parts) == 3:
            hours = int(parts[0])
            minutes = int(parts[1])
            seconds = int(parts[2])
            return hours * 3600 + minutes * 60 + seconds
        print(f"无法解析时间格式: {time_str}")
        return 0.0
    except Exception as e:
        print(f"解析时间 {time_str} 时出错: {str(e)}")
        return 0.0


def take_screenshot(video_path: Path,
                    screenshot_time: float,
                    output_path: Path,
                    index: int = 0) -> bool:
    """
    在指定时间点对视频进行截图
    参数:
        video_path: 视频文件路径
        screenshot_time: 截图时间点（秒）
        output_path: 输出截图文件路径
        index: 截图索引，用于命名
    返回:
        截图是否成功
    """
    try:
        # 确保输出目录存在
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 构造ffmpeg命令
        cmd = [
            'ffmpeg',
            '-y',  # 覆盖输出文件
            '-ss',
            format_time_for_ffmpeg(screenshot_time),  # 指定截图时间
            '-i',
            str(video_path),  # 输入视频文件
            '-vframes',
            '1',  # 只截取一帧
            '-q:v',
            '2',  # 图片质量 (2表示高质量)
            str(output_path)
        ]

        print(f"执行截图命令: {' '.join(cmd)}")

        # 执行截图命令
        result = subprocess.run(cmd,
                                capture_output=True,
                                text=True,
                                encoding='utf-8',
                                errors='ignore')

        if result.returncode == 0:
            print(f"截图成功: {output_path}")
            return True
        else:
            print(f"截图失败: {result.stderr}")
            return False

    except Exception as e:
        print(f"截图时发生错误: {str(e)}")
        return False


def _validate_excel_data(df: pd.DataFrame) -> bool:
    """验证Excel数据是否有效"""
    required_columns = ['时间']
    for col in required_columns:
        if col not in df.columns:
            print(f"Excel文件缺少必要列: {col}")
            return False
    return True


def _parse_row_time(row: pd.Series) -> float:
    """解析行中的时间数据"""
    time_str = str(row['时间'])
    try:
        base_time = parse_time_from_excel(time_str)
        if base_time <= 0 and time_str:
            print(f"时间格式不正确: {time_str}")
            return 0.0
        return base_time
    except Exception as e:
        print(f"解析时间失败 {time_str}: {e}")
        return 0.0


def _take_screenshot_with_offset(video_path: Path, base_time: float,
                                 time_str: str, offset: float, offset_idx: int,
                                 index: int, output_dir: Path) -> None:
    """为指定偏移量拍摄截图"""
    screenshot_time = max(0, base_time + offset)
    safe_time_str = time_str.replace(':', '-')
    output_filename = (f"{video_path.stem}_{index+1:02d}_{offset_idx+1:02d}_"
                       f"{safe_time_str}.jpg")
    output_path = output_dir / output_filename

    cmd = [
        'ffmpeg', '-y', '-ss',
        format_time_for_ffmpeg(screenshot_time), '-i',
        str(video_path), '-vframes', '1', '-q:v', '2',
        str(output_path)
    ]

    print(f"正在对视频 {video_path.name} 在时间 {screenshot_time:.3f} 进行截图")
    print(f"执行截图命令: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd,
                                capture_output=True,
                                text=True,
                                encoding='utf-8',
                                errors='ignore')
        if result.returncode == 0:
            print(f"截图成功: {output_path}")
        else:
            print(f"截图失败: {result.stderr}")
    except Exception as e:
        print(f"截图过程中出现错误: {e}")


def process_excel_and_screenshots(excel_path: Union[str, Path],
                                  video_path: Union[str, Path],
                                  output_dir: Union[str, Path],
                                  time_offsets: List[int]) -> None:
    """
    根据Excel表格中的时间信息对视频进行截图
    参数:
        excel_path: Excel文件路径
        video_path: 视频文件路径
        output_dir: 截图输出目录
        time_offsets: 时间偏移列表（相对于事件时间的秒数）
    """
    excel_path = Path(excel_path)
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        df = pd.read_excel(excel_path)
    except Exception as e:
        print(f"读取Excel文件失败: {e}")
        return

    if not _validate_excel_data(df):
        return

    for index, row in df.iterrows():
        base_time = _parse_row_time(row)
        if base_time <= 0:
            continue

        time_str = str(row['时间'])
        for offset_idx, offset in enumerate(time_offsets):
            _take_screenshot_with_offset(video_path, base_time, time_str,
                                         offset, offset_idx, index, output_dir)


def main():
    """
    主函数，用于直接运行截图模块
    """
    # 检查FFmpeg是否可用
    if not check_ffmpeg():
        sys.exit(1)

    # 获取脚本所在目录
    script_dir = get_script_dir()

    # 加载配置文件
    config_path = script_dir / 'data_process_config.json'
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
    except Exception as e:
        print(f"无法加载配置文件: {e}")
        sys.exit(1)

    # 获取截图模块配置
    screenshot_settings = config.get("modules", {}).get("screenshot", {})
    default_input = config.get("input_path", str(script_dir / "video"))
    default_output = config.get("output_path", str(script_dir / "Output"))

    # 命令行参数解析
    parser = argparse.ArgumentParser(description='根据Excel表格中的时间信息对视频进行截图')
    parser.add_argument('excel', help='Excel文件路径')
    parser.add_argument('video',
                        nargs='?',
                        default=default_input,
                        help='视频文件路径')
    parser.add_argument('output',
                        nargs='?',
                        default=default_output,
                        help='截图输出目录')
    parser.add_argument('--offsets',
                        type=str,
                        default=','.join(
                            map(str,
                                screenshot_settings.get("offsets", [-1, 0]))),
                        help='时间偏移（相对于事件时间的秒数，用逗号分隔）')
    parser.add_argument('--disable', action='store_true', help='禁用截图模块')

    args = parser.parse_args()

    # 如果禁用了模块，则直接返回
    if args.disable:
        print("截图模块已禁用")
        sys.exit(0)

    # 解析时间偏移
    try:
        time_offsets = [int(x) for x in args.offsets.split(',')]
    except Exception as e:
        print(f"时间偏移格式不正确: {e}")
        sys.exit(1)

    # 执行截图操作
    process_excel_and_screenshots(args.excel, args.video, args.output,
                                  time_offsets)


if __name__ == '__main__':
    main()
