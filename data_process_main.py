#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
主程序文件，用于协调调用各个子程序模块
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List
import pandas as pd

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent))

# 导入音频处理单元
try:
    from audio_processing_unit import process_audio, process_folder
    AUDIO_PROCESSING_MODULE_AVAILABLE = True
except ImportError as e:
    print(f"音频处理模块导入失败: {e}")
    AUDIO_PROCESSING_MODULE_AVAILABLE = False

# 导入截图模块
try:
    from screenshot_module import process_excel_and_screenshots, check_ffmpeg
    SCREENSHOT_MODULE_AVAILABLE = True
except ImportError as e:
    print(f"截图模块导入失败: {e}")
    SCREENSHOT_MODULE_AVAILABLE = False

# 导入警告音检测模块
try:
    from warningSoundDetection_unit import main as warning_sound_detection_main
    WARNING_SOUND_DETECTION_MODULE_AVAILABLE = True
except ImportError as e:
    print(f"警告音检测模块导入失败: {e}")
    WARNING_SOUND_DETECTION_MODULE_AVAILABLE = False


def load_config(config_path: Path) -> dict:
    """从JSON文件加载配置"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"加载配置文件失败: {e}")
        sys.exit(1)


def get_config_path() -> Path:
    """获取配置文件路径"""
    script_dir = Path(__file__).parent.absolute()
    return script_dir / 'data_process_config.json'


def get_config_value(config: dict, *keys: str) -> any:
    """获取配置值"""
    current = config
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return None
    return current


def format_time(seconds: float) -> str:
    """
    将秒数转换为H.MM.SS格式的时间字符串
    """

    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours}.{minutes:02d}.{secs:02d}"


# 新增：从纳秒格式化为 HH:MM:SS.nnnnnnnnn 的函数


def format_time_from_ns(ns: int) -> str:
    """将纳秒整数格式化为 HH:MM:SS.nnnnnnnnn"""
    if ns < 0:
        ns = 0
    hours, rem_ns = divmod(ns, 3_600_000_000_000)
    minutes, rem_ns = divmod(rem_ns, 60_000_000_000)
    seconds, nanos = divmod(rem_ns, 1_000_000_000)
    return f"{hours:01d}:{minutes:02d}:{seconds:02d}.{nanos:09d}"


def parse_filename_settings(filename: str) -> dict:
    """
    解析文件名中的设置信息
    参数:
        filename: 文件名字符串
    返回:
        包含设置信息的字典
    """
    settings = {
        'S': '',
        'D': '',
        '超速': '',
        '区间': '',
        '移动': '',
        '桥': '',
        'F': '',
        'C': ''
    }

    # 使用正则表达式直接提取各设置项的值
    import re

    # 首先找到"fakeNMEA_"到最后一个"_"之间的部分
    fake_nmea_match = re.search(r'fakeNMEA_(.+?)_[^_]*$', filename)
    if not fake_nmea_match:
        return settings

    # 提取fakeNMEA_后面的设置部分
    settings_part = fake_nmea_match.group(1)

    # 在设置部分中查找各个设置项
    # 提取S设置项的值
    s_match = re.search(r'S([^-]+)(?=-|$)', settings_part)
    if s_match:
        settings['S'] = s_match.group(1)

    # 提取D设置项的值
    d_match = re.search(r'D([^-]+)(?=-|$)', settings_part)
    if d_match:
        settings['D'] = d_match.group(1)

    # 提取超速设置项的值
    chaosu_match = re.search(r'超速([^-]+)(?=-|$)', settings_part)
    if chaosu_match:
        settings['超速'] = chaosu_match.group(1)

    # 提取区间设置项的值
    qujian_match = re.search(r'区间([^-]+)(?=-|$)', settings_part)
    if qujian_match:
        settings['区间'] = qujian_match.group(1)

    # 提取移动设置项的值
    yidong_match = re.search(r'移动([^-]+)(?=-|$)', settings_part)
    if yidong_match:
        settings['移动'] = yidong_match.group(1)

    # 提取桥设置项的值
    qiao_match = re.search(r'桥([^-]+)(?=-|$)', settings_part)
    if qiao_match:
        settings['桥'] = qiao_match.group(1)

    # 提取F设置项的值
    f_match = re.search(r'F([^-]+)(?=-|$)', settings_part)
    if f_match:
        settings['F'] = f_match.group(1)

    # 提取C设置项的值（数值，可正可负）
    c_match = re.search(r'C(-?\d+)', settings_part)
    if c_match:
        settings['C'] = c_match.group(1)

    return settings


def create_empty_table(output_path: Path, audio_name: str, filename: str = ""):
    """
    在程序一开始就创建空的表格数据文件
    参数:
        output_path: 输出路径
        audio_name: 音频文件名
        filename: 原始文件名（用于解析设置信息）
    """
    print("构建初始表格数据")
    # 创建空的DataFrame，但包含所有必要的列
    df = pd.DataFrame({
        '时间': [],
        '限速': [],
        '距离': [],
        '速度': [],
        'Avg': [],
        'Sec': [],
        '播报': [],
        '超速': [],
        'event类别': [],
        '警示声音': [],
        'icon': [],
        '异常': []
    })

    # 生成表格文件名
    table_filename = f"{audio_name}.xlsx"
    table_path = output_path / table_filename
    print(f"初始表格文件路径: {table_path}")

    # 创建并准备Sheet2数据
    settings_df = pd.DataFrame({
        '': ['S', 'D', '超速', '区间', '移动', '桥', 'F', 'C'],
        'Settings': ['', '', '', '', '', '', '', ''],
        'Actual': ['', '', '', '', '', '', '', ''],
        'Result': ['', '', '', '', '', '', '', '']
    })

    # 如果提供了文件名，则解析设置信息并填充到Settings列
    if filename:
        try:
            settings = parse_filename_settings(filename)
            # 更新Settings列的值
            settings_df.loc[0, 'Settings'] = settings['S']  # S设置
            settings_df.loc[1, 'Settings'] = settings['D']  # D设置
            settings_df.loc[2, 'Settings'] = settings['超速']  # 超速设置
            settings_df.loc[3, 'Settings'] = settings['区间']  # 区间设置
            settings_df.loc[4, 'Settings'] = settings['移动']  # 移动设置
            settings_df.loc[5, 'Settings'] = settings['桥']  # 桥设置
            settings_df.loc[6, 'Settings'] = settings['F']  # F设置
            settings_df.loc[7, 'Settings'] = settings['C']  # C设置
        except Exception as e:
            print(f"解析文件名设置时出错: {e}")

    # 导出空表格
    try:
        with pd.ExcelWriter(table_path, engine='openpyxl') as writer:
            # 写入主表
            df.to_excel(writer, index=False, sheet_name='Sheet1')

            # 写入Sheet2
            settings_df.to_excel(writer, index=False, sheet_name='Sheet2')

        print(f"初始表格已生成: {table_path}")
    except PermissionError:
        print(f"权限错误：无法写入文件 {table_path}，文件可能正在被其他程序使用")
    except Exception as e:
        print(f"创建初始表格时出错: {e}")


def update_table_data(event_times: List[int],
                      subtitles: List[str],
                      output_path: Path,
                      audio_name: str,
                      filename: str = ""):
    """
    更新表格数据并保存为Excel文件
    参数:
        event_times: 事件时间点列表（纳秒）
        subtitles: 字幕内容列表
        output_path: 输出路径
        audio_name: 音频文件名
        filename: 原始文件名（用于解析设置信息）
    """
    print(f"更新表格数据，事件时间点数量: {len(event_times)}, 字幕数量: {len(subtitles)}")

    if not event_times and not subtitles:
        print("没有数据可用于更新表格")
        return

    print("更新表格数据")
    # 确保两个列表长度一致
    max_len = max(len(event_times), len(subtitles))
    print(f"最大长度: {max_len}")

    # 如果列表长度不一致，用空值填充较短的列表
    if len(event_times) < max_len:
        event_times.extend([0] * (max_len - len(event_times)))
    if len(subtitles) < max_len:
        subtitles.extend([''] * (max_len - len(subtitles)))

    df = pd.DataFrame({
        '时间': [format_time_from_ns(t) for t in event_times],
        '限速': [''] * max_len,
        '距离': [''] * max_len,
        '速度': [''] * max_len,
        'Avg': [''] * max_len,
        'Sec': [''] * max_len,
        '播报': subtitles,
        '超速': [''] * max_len,
        'event类别': [''] * max_len,
        '警示声音': [''] * max_len,
        'icon': [''] * max_len,
        '异常': [''] * max_len
    })
    print(f"表格数据更新完成，共 {max_len} 行")

    # 生成表格文件名
    table_filename = f"{audio_name}.xlsx"
    table_path = output_path / table_filename
    print(f"表格文件路径: {table_path}")

    # 创建并准备Sheet2数据
    settings_df = pd.DataFrame({
        '': ['S', 'D', '超速', '区间', '移动', '桥', 'F', 'C'],
        'Settings': ['', '', '', '', '', '', '', ''],
        'Actual': ['', '', '', '', '', '', '', ''],
        'Result': ['', '', '', '', '', '', '', '']
    })

    # 如果提供了文件名，则解析设置信息并填充到Settings列
    if filename:
        try:
            settings = parse_filename_settings(filename)
            # 更新Settings列的值
            settings_df.loc[0, 'Settings'] = settings['S']  # S设置
            settings_df.loc[1, 'Settings'] = settings['D']  # D设置
            settings_df.loc[2, 'Settings'] = settings['超速']  # 超速设置
            settings_df.loc[3, 'Settings'] = settings['区间']  # 区间设置
            settings_df.loc[4, 'Settings'] = settings['移动']  # 移动设置
            settings_df.loc[5, 'Settings'] = settings['桥']  # 桥设置
            settings_df.loc[6, 'Settings'] = settings['F']  # F设置
            settings_df.loc[7, 'Settings'] = settings['C']  # C设置
        except Exception as e:
            print(f"解析文件名设置时出错: {e}")

    # 导出表格
    try:
        with pd.ExcelWriter(table_path, engine='openpyxl') as writer:
            # 写入主表
            df.to_excel(writer, index=False, sheet_name='Sheet1')

            # 写入Sheet2
            settings_df.to_excel(writer, index=False, sheet_name='Sheet2')

        print(f"表格已更新: {table_path}")
    except PermissionError:
        print(f"权限错误：无法写入文件 {table_path}，文件可能正在被其他程序使用")
    except Exception as e:
        print(f"更新表格时出错: {e}")


def main():
    """
    主程序入口
    """

    # 加载配置文件
    config_path = get_config_path()
    config = load_config(config_path)

    # 从配置文件中获取输入输出路径
    input_path_str = get_config_value(config, "input_path")
    output_path_str = get_config_value(config, "output_path")

    if not input_path_str or not output_path_str:
        print("配置文件中缺少输入或输出路径配置")
        sys.exit(1)

    # 转换为Path对象
    default_input = Path(input_path_str)
    default_output = Path(output_path_str)

    # 获取模块设置
    modules_settings = get_config_value(config, "modules") if config else {}
    audio_processing_settings = get_config_value(
        modules_settings, "audio_processing") if modules_settings else {}
    screenshot_settings = get_config_value(
        modules_settings, "screenshot") if modules_settings else {}
    warning_sound_detection_settings = get_config_value(
        modules_settings, "warning_sound_detection") if modules_settings else {}

    if not audio_processing_settings or not screenshot_settings:
        print("配置文件中缺少模块设置")
        sys.exit(1)

    audio_processing_enabled = get_config_value(audio_processing_settings,
                                                "enabled")
    screenshot_enabled = get_config_value(screenshot_settings, "enabled")
    warning_sound_detection_enabled = get_config_value(warning_sound_detection_settings,
                                                       "enabled") if warning_sound_detection_settings else True

    # 命令行参数解析
    parser = argparse.ArgumentParser(description='SpeedGuard音频处理主程序')
    parser.add_argument(
        'input',
        nargs='?',  # 使参数可选
        default=str(default_input),
        help='输入音频文件或文件夹路径')
    parser.add_argument(
        'output',
        nargs='?',  # 使参数可选
        default=str(default_output),
        help='输出目录路径（用于表格等输出文件）')
    parser.add_argument('--screenshot',
                        action='store_true',
                        default=screenshot_enabled,
                        help='是否在处理后进行截图')
    parser.add_argument('--no-audio-processing',
                        action='store_true',
                        default=not audio_processing_enabled,
                        help='禁用音频处理功能')
    parser.add_argument('--no-warning-sound-detection',
                        action='store_true',
                        default=not warning_sound_detection_enabled,
                        help='禁用警告音检测功能')

    args = parser.parse_args()

    # 转换为Path对象
    input_path = Path(args.input)
    output_path = Path(args.output)

    print(f"输入路径: {input_path}")
    print(f"输出路径: {output_path}")

    # 输入路径验证
    if not input_path.exists():
        print(f"输入路径不存在: {input_path}")
        sys.exit(1)

    # 创建输出目录（如果不存在）
    if not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True)

    # 判断输入是文件还是目录
    if input_path.is_file():
        # 处理单个音频文件
        print(f"处理单个音频文件: {input_path}")

        # 只有在启用音频处理时才创建初始表格
        if not args.no_audio_processing and AUDIO_PROCESSING_MODULE_AVAILABLE:
            # 在处理任何内容之前，先创建初始表格
            print("创建初始表格...")
            create_empty_table(output_path, input_path.stem, input_path.name)

            # 使用output_path作为表格输出路径
            # 从配置中获取音频处理参数
            silence_thresh = get_config_value(audio_processing_settings,
                                              "silence_thresh")
            silence_duration = get_config_value(audio_processing_settings,
                                                "silence_duration")

            if silence_thresh is None or silence_duration is None:
                print("配置文件中缺少音频处理参数")
                sys.exit(1)

            # 处理音频并获取事件时间和字幕
            print("开始音频处理...")
            event_times, subtitles = process_audio(input_path, output_path,
                                                   silence_thresh,
                                                   silence_duration)
            print(f"音频处理完成，获得事件时间点: {event_times}")
            print(f"音频处理完成，获得字幕: {subtitles}")

            # 更新表格数据
            print("开始更新表格数据...")
            update_table_data(event_times, subtitles, output_path,
                              input_path.stem, input_path.name)
        elif (not args.no_audio_processing
              and not AUDIO_PROCESSING_MODULE_AVAILABLE):
            print("音频处理模块不可用，请检查模块是否正确安装")
            sys.exit(1)
        elif args.no_audio_processing:
            print("音频处理功能已禁用")

        # 如果需要截图，则执行截图操作
        if args.screenshot:
            # 检查截图模块是否可用
            if not SCREENSHOT_MODULE_AVAILABLE:
                print("截图模块不可用，请检查模块是否正确安装")
                sys.exit(1)

            # 检查FFmpeg是否可用
            if not check_ffmpeg():
                print("截图功能需要FFmpeg支持，请确保已安装FFmpeg")
                sys.exit(1)

            # 生成Excel文件名
            excel_filename = f"{input_path.stem}.xlsx"
            excel_path = output_path / excel_filename

            # 检查Excel文件是否存在
            if excel_path.exists():
                # 设置截图输出目录
                output_subdir = get_config_value(screenshot_settings,
                                                 "output_subdir")
                if not output_subdir:
                    print("配置文件中缺少截图输出子目录配置")
                    sys.exit(1)

                screenshots_dir = output_path / output_subdir

                # 获取截图偏移设置
                offsets = get_config_value(screenshot_settings, "offsets")
                if not offsets:
                    print("配置文件中缺少截图时间偏移配置")
                    sys.exit(1)

                # 执行截图操作
                print(f"开始根据Excel文件 {excel_path} 进行截图")
                process_excel_and_screenshots(excel_path, input_path,
                                              screenshots_dir, offsets)
            else:
                print(f"未找到Excel文件: {excel_path}")
    else:
        # 处理文件夹内所有音频文件
        print(f"处理文件夹内所有音频文件: {input_path}")

        # 从配置文件中获取支持的音频格式
        supported_formats = config.get(
            "supported_formats",
            ['.mp3', '.wav', '.flac', '.aac', '.m4a', '.wma', '.mp4'])
        audio_files = [
            f for f in input_path.glob('*.*')
            if f.suffix.lower() in supported_formats
        ]
        print(f"找到 {len(audio_files)} 个音频文件")

        # 只有在启用音频处理时才创建初始表格
        if not args.no_audio_processing and AUDIO_PROCESSING_MODULE_AVAILABLE:
            for audio_file in audio_files:
                print(f"为音频文件 {audio_file.name} 创建初始表格")
                create_empty_table(output_path, audio_file.stem,
                                   audio_file.name)

            # 使用output_path作为表格输出路径
            # 从配置中获取音频处理参数
            silence_thresh = get_config_value(audio_processing_settings,
                                              "silence_thresh")
            silence_duration = get_config_value(audio_processing_settings,
                                                "silence_duration")

            if silence_thresh is None or silence_duration is None:
                print("配置文件中缺少音频处理参数")
                sys.exit(1)

            # 处理文件夹中的音频并生成事件时间和字幕
            print("开始处理文件夹中的音频...")
            event_times_dict, subtitles_dict = process_folder(
                input_path, output_path, silence_thresh, silence_duration,
                config)
            print(f"文件夹处理完成，获得事件时间点: {event_times_dict}")
            print(f"文件夹处理完成，获得字幕: {subtitles_dict}")

            # 为每个音频更新表格数据
            for audio_file, event_times in event_times_dict.items():
                subtitles = subtitles_dict.get(audio_file, [])
                print(f"更新 {audio_file.name} 的表格数据")
                update_table_data(event_times, subtitles, output_path,
                                  audio_file.stem, audio_file.name)
        elif (not args.no_audio_processing
              and not AUDIO_PROCESSING_MODULE_AVAILABLE):
            print("音频处理模块不可用，请检查模块是否正确安装")
            sys.exit(1)
        elif args.no_audio_processing:
            print("音频处理功能已禁用")

        # 如果需要截图，则对每个音频执行截图操作
        if args.screenshot:
            # 检查截图模块是否可用
            if not SCREENSHOT_MODULE_AVAILABLE:
                print("截图模块不可用，请检查模块是否正确安装")
                sys.exit(1)

            # 检查FFmpeg是否可用
            if not check_ffmpeg():
                print("截图功能需要FFmpeg支持，请确保已安装FFmpeg")
                sys.exit(1)

            # 获取截图偏移设置
            offsets = get_config_value(screenshot_settings, "offsets")
            if not offsets:
                print("配置文件中缺少截图时间偏移配置")
                sys.exit(1)

            # 获取截图输出子目录
            output_subdir = get_config_value(screenshot_settings,
                                             "output_subdir")
            if not output_subdir:
                print("配置文件中缺少截图输出子目录配置")
                sys.exit(1)

            # 遍历处理过的音频文件
            for audio_file in audio_files:
                # 生成对应的Excel文件名
                excel_filename = f"{audio_file.stem}.xlsx"
                excel_path = output_path / excel_filename

                # 检查Excel文件是否存在
                if excel_path.exists():
                    # 设置截图输出目录
                    screenshots_dir = output_path / output_subdir

                    # 执行截图操作
                    print(
                        f"开始根据Excel文件 {excel_path} 对音频 {audio_file.name} 进行截图")
                    process_excel_and_screenshots(excel_path, audio_file,
                                                  screenshots_dir, offsets)
                else:
                    print(f"未找到Excel文件: {excel_path}")

    # 执行警告音检测
    if not args.no_warning_sound_detection and WARNING_SOUND_DETECTION_MODULE_AVAILABLE:
        print("开始执行警告音检测...")
        try:
            warning_sound_detection_main()
            print("警告音检测完成")
        except Exception as e:
            print(f"警告音检测过程中出现错误: {e}")
    elif not args.no_warning_sound_detection and not WARNING_SOUND_DETECTION_MODULE_AVAILABLE:
        print("警告音检测模块不可用，请检查模块是否正确安装")
    elif args.no_warning_sound_detection:
        print("警告音检测功能已禁用")

    print("所有处理已完成")


if __name__ == '__main__':
    main()