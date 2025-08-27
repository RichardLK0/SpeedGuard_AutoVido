#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
音频处理模块，用于从音频文件中检测有声片段并进行语音识别
"""

import subprocess
import json
import argparse
from pathlib import Path
# 移除未使用的pandas导入
from typing import List, Tuple, Dict, Any
import re
try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    print("警告: 未安装whisper库，语音识别功能不可用")
    WHISPER_AVAILABLE = False
import sys

# ======================
# 系统级功能函数
# ======================


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


def get_audio_duration(audio_path: Path) -> float:
    """
    获取音频总时长(秒)
    参数: audio_path (Path对象)
    返回: 时长(浮点数)或0(失败时)
    """
    cmd = [
        'ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of',
        'default=noprint_wrappers=1:nokey=1',
        str(audio_path)
    ]

    try:
        # 执行命令并读取输出，修复编码问题
        result = subprocess.check_output(cmd,
                                         stderr=subprocess.DEVNULL,
                                         encoding='utf-8',
                                         errors='ignore')
        duration = float(result.strip())
        return duration
    except Exception as e:
        print(f"获取时长失败 {audio_path}: {str(e)}")
        return 0


def format_time(seconds: float) -> str:
    """
    生成兼容文件名的时间格式(H.MM.SS)
    """

    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds_remaining = seconds % 60

    # 分离整数和小数部分
    seconds_int = int(seconds_remaining)

    return f"{hours:01d}.{minutes:02d}.{seconds_int:02d}"


# ======================
# 音频分析函数
# ======================


def detect_audio_activity(
        audio_path: Path,
        silence_thresh: float = -30,
        silence_duration: float = 1.0,
        total_duration: float = 0) -> List[Tuple[float, float]]:
    """
    检测音频中的有声区间
    参数：
        audio_path: 音频文件路径
        silence_thresh: 静音阈值(dB)值越小越严格
        silence_duration: 静音持续时间(秒)
        total_duration: 音频总时长
    返回：有声区间列表 [(开始时间, 结束时间)]
    """
    cmd = [
        'ffmpeg',
        '-i',
        str(audio_path),  # 输入文件
        '-af',
        (f'silencedetect=noise={silence_thresh}dB:'
         f'd={silence_duration}'),  # 静音检测滤镜
        '-f',
        'null',  # 输出格式设为null(不生成文件)
        '-'
    ]

    try:
        # 执行命令并捕获输出，修复编码问题
        result = subprocess.run(cmd,
                                capture_output=True,
                                text=True,
                                encoding='utf-8',
                                errors='ignore',
                                check=True)
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg执行失败: {e.stderr}")
        return []

    if not result.stderr:
        print("警告: FFmpeg未返回有效数据")
        return []

    output = result.stderr
    active_intervals = []
    prev_end = 0.0  # 上一个有声区间的结束时间

    # 解析FFmpeg输出
    for line in output.split('\n'):
        if 'silence_start' in line:
            try:
                start = float(
                    re.search(r'silence_start: (\d+\.\d+)', line).group(1))
                # 如果静音开始时间在上一个有声区间之后，记录之前的区间
                if start > prev_end:
                    active_intervals.append((prev_end, start))
                prev_end = start  # 更新静音开始时间
            except AttributeError:
                continue
        elif 'silence_end' in line:
            try:
                end = float(
                    re.search(r'silence_end: (\d+\.\d+)', line).group(1))
                prev_end = end  # 更新最后一个静音结束时间
            except AttributeError:
                continue

    # 处理最后一个有声区间
    if total_duration > 0 and prev_end < total_duration:
        active_intervals.append((prev_end, total_duration))

    return active_intervals


# ======================
# 音频区间预处理函数
# ======================


def preprocess_intervals(
        intervals: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """
    预处理有声区间，合并间隔小于1.5秒的区间
    返回所有有声区间（包括短且孤立的区间）
    """
    if not intervals:
        return []

    # 按开始时间排序
    intervals = sorted(intervals, key=lambda x: x[0])

    # 第一步：合并间隔小于1.5秒的区间
    merged = []
    current_start, current_end = intervals[0]

    for i in range(1, len(intervals)):
        start, end = intervals[i]

        # 如果当前区间结束与下一区间开始间隔小于1.5秒，则合并
        if start - current_end <= 1.5:
            current_end = max(current_end, end)
        else:
            merged.append((current_start, current_end))
            current_start, current_end = start, end

    merged.append((current_start, current_end))

    return merged


# ======================
# 音频处理函数
# ======================


def simple_audio_transcribe(audio_path: Path) -> List[Dict[str, Any]]:
    """
    简单的音频转录函数，当whisper不可用时使用
    参数：
        audio_path: 音频文件路径
    返回：空的识别结果列表
    """
    print("警告: 由于whisper不可用，跳过语音识别步骤")
    return []


def transcribe_audio_segments(audio_path: Path, intervals: List[Tuple[float,
                                                                      float]],
                              total_duration: float) -> List[Dict[str, Any]]:
    """
    对指定的时间区间分别进行语音识别
    参数：
        audio_path: 音频文件路径
        intervals: 需要识别的时间区间列表
        total_duration: 音频总时长
    返回：识别结果列表，每个元素包含开始时间、结束时间和文本
    """
    # 检查whisper是否可用
    if not WHISPER_AVAILABLE:
        return simple_audio_transcribe(audio_path)

    try:
        # 加载Whisper模型
        model = whisper.load_model("medium")

        all_segments = []

        # 对每个区间单独进行语音识别
        for i, (start, end) in enumerate(intervals):
            print(
                f"正在识别第 {i+1}/{len(intervals)} 个音频区间: {start:.2f} - {end:.2f}")

            # 创建临时音频片段文件
            segment_temp = audio_path.with_name(f'temp_segment_{i}.wav')

            # 使用ffmpeg截取音频片段
            extract_segment_cmd = [
                'ffmpeg',
                '-y',
                '-ss',
                str(max(0, start - 1)),  # 提前1秒开始，确保不丢失开头
                '-t',
                str(min(end - start + 2, total_duration)),  # 延长2秒，确保不丢失结尾
                '-i',
                str(audio_path),
                '-vn',
                '-acodec',
                'pcm_s16le',
                '-ar',
                '44100',
                '-ac',
                '1',
                str(segment_temp)
            ]

            result = subprocess.run(extract_segment_cmd,
                                    capture_output=True,
                                    text=True,
                                    encoding='utf-8',
                                    errors='ignore')
            if result.returncode != 0:
                print(f"音频片段提取失败: {result.stderr}")
                continue

            # 转录音频片段
            segment_result = model.transcribe(str(segment_temp), fp16=False)

            # 调整时间戳以匹配原始音频位置
            for segment in segment_result["segments"]:
                segment["start"] += max(0, start - 1)
                segment["end"] += max(0, start - 1)
                all_segments.append(segment)

            # 删除临时音频片段文件
            if segment_temp.exists():
                segment_temp.unlink()

        # 按开始时间排序
        all_segments.sort(key=lambda x: x["start"])

        # 合并相邻的片段
        merged_segments = []
        current_segment = None

        for segment in all_segments:
            if current_segment is None:
                current_segment = segment
            else:
                # 如果当前片段与上一个片段的间隔小于1秒，则合并
                if segment["start"] - current_segment["end"] < 1.0:
                    current_segment["text"] += " " + segment["text"]
                    current_segment["end"] = segment["end"]
                else:
                    # 过滤孤立短片段：
                    # 只保留长度≥1.5秒的片段
                    duration = (current_segment["end"] -
                                current_segment["start"])
                    if duration >= 1.5:
                        merged_segments.append(current_segment)
                    current_segment = segment

        # 处理最后一个片段
        if current_segment is not None:
            duration = current_segment["end"] - current_segment["start"]
            if duration >= 1.5:
                merged_segments.append(current_segment)

        return merged_segments
    except Exception as e:
        print(f"语音识别失败: {str(e)}")
        return []


def process_audio(audio_path: Path, output_dir: Path, silence_thresh: float,
                  silence_duration: float) -> Tuple[List[int], List[str]]:
    """
    处理单个音频文件，提取事件时间点和字幕内容
    参数：
        audio_path: 音频文件路径
        output_dir: 输出目录（用于临时文件）
        silence_thresh: 静音阈值
        silence_duration: 静音持续时间
    返回：
        Tuple[List[int], List[str]]: 事件时间点（纳秒）列表和字幕内容列表
    """
    # 导入opencc库用于繁体转简体
    try:
        from opencc import OpenCC
        cc = OpenCC('t2s')  # 繁体转简体
        use_opencc = True
    except ImportError:
        print("警告: 未安装opencc库，无法进行繁简转换")
        use_opencc = False

    print(f"开始处理音频: {audio_path}")
    audio_temp = None
    try:
        # 获取原始文件名（不含扩展名）作为表格文件名
        base_name = audio_path.stem

        print(f"解析文件名: {base_name}")

        # 创建主输出目录（用于临时文件和表格输出）
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"输出目录: {output_dir}")

        # 获取音频总时长
        total_duration = get_audio_duration(audio_path)
        print(f"音频总时长: {total_duration} 秒")

        if total_duration == 0:
            print("错误：无法获取音频时长，可能音频文件损坏或格式不支持")
            return [], []

        # 创建临时文件目录 (项目根目录下的temporary_files文件夹)
        temp_dir = get_script_dir() / 'temporary_files'
        temp_dir.mkdir(parents=True, exist_ok=True)
        print(f"临时文件目录: {temp_dir}")

        # 步骤1：使用传入的音频文件路径
        audio_temp = audio_path
        print(f"音频文件路径: {audio_temp}")

        # 步骤2：检测有声区间
        print("开始检测音频活动区间")
        intervals = detect_audio_activity(audio_temp, silence_thresh,
                                          silence_duration, total_duration)
        print(f"检测到 {len(intervals)} 个有声区间")

        if not intervals:
            print(f"无有效音频: {audio_path}")
            # 即使没有检测到有声区间，也返回空列表
            event_times = []
            subtitles = []
        else:
            # 步骤3：预处理有声区间（合并间隔小于1秒的区间）
            print("开始预处理有声区间")
            intervals = preprocess_intervals(intervals)
            print(f"预处理后剩余 {len(intervals)} 个有声区间")
            if not intervals:
                print(f"预处理后无有效音频区间: {audio_path}")
                event_times = []
                subtitles = []
            else:
                # 步骤4：对每个音频区间单独进行语音识别
                print(f"开始对每个音频区间进行语音识别: {audio_path}")

                # 检查whisper是否可用
                if not WHISPER_AVAILABLE:
                    print("警告: whisper库不可用，跳过语音识别步骤")
                    all_segments = []
                else:
                    all_segments = transcribe_audio_segments(
                        audio_temp, intervals, total_duration)
                print(f"总共识别到 {len(all_segments)} 个语音片段")

                # 收集事件时间点和字幕内容
                event_times = []
                subtitles = []

                for i, (start, end) in enumerate(intervals):
                    print(f"处理第 {i+1} 个区间: {start} - {end}")
                    # 扩展时间窗口（前后3秒）
                    expanded_start = max(0.0, start - 3)
                    expanded_end = min(total_duration, end + 3)
                    print(f"扩展后的时间窗口: {expanded_start} - {expanded_end}")

                    # 查找在当前时间区间内的字幕片段
                    segment_text = ""
                    segment_count = 0
                    for segment in all_segments:
                        # 如果字幕片段在当前区间内
                        if ((segment["start"] >= start
                             and segment["start"] <= end) or
                            (segment["end"] >= start and segment["end"] <= end)
                                or (segment["start"] <= start
                                    and segment["end"] >= end)):
                            segment_text += segment["text"] + " "
                            segment_count += 1

                    print(f"在该区间内找到 {segment_count} 个相关片段")
                    # 以纳秒精度记录事件时间
                    event_time_ns = int(round(start * 1_000_000_000))
                    event_times.append(event_time_ns)

                    # 清理文本，移除多余空格
                    cleaned_text = " ".join(segment_text.strip().split())

                    # 繁体转简体
                    if use_opencc and cleaned_text:
                        cleaned_text = cc.convert(cleaned_text)

                    # 限制文本长度，避免过长
                    if len(cleaned_text) > 100:
                        # 如果文本过长，只取前100个字符并添加省略号
                        subtitles.append(cleaned_text[:100] + "...")
                    else:
                        subtitles.append(cleaned_text)

                    print(
                        f"事件时间(纳秒): {event_times[-1]}, 字幕内容: {subtitles[-1]}")

        print("处理完成")
        # 返回事件时间点和字幕内容
        return event_times, subtitles

    except Exception as e:
        print(f"处理失败 {audio_path}: {str(e)}")
        import traceback
        traceback.print_exc()
        return [], []


# ======================
# 文件夹处理函数
# ======================


def process_folder(
    input_folder: Path,
    output_base: Path,
    silence_thresh: float,
    silence_duration: float,
    config: dict = None
) -> Tuple[Dict[Path, List[int]], Dict[Path, List[str]]]:
    """
    处理文件夹内所有音频文件
    参数：
        input_folder: 输入文件夹路径
        output_base: 输出根目录
        silence_thresh: 静音阈值
        silence_duration: 静音持续时间
        config: 配置字典，包含支持的文件格式等信息
    返回：
        tuple: (事件时间点字典(纳秒), 字幕内容字典)
    """
    input_folder = Path(input_folder)
    output_base = Path(output_base)

    # 存储所有音频的事件时间和字幕
    all_event_times: Dict[Path, List[int]] = {}
    all_subtitles: Dict[Path, List[str]] = {}

    # 从配置文件中读取支持的文件格式
    if config is None:
        supported_formats = [
            '.mp3', '.wav', '.flac', '.aac', '.m4a', '.wma', '.mp4'
        ]
    else:
        supported_formats = config.get(
            "supported_formats",
            ['.mp3', '.wav', '.flac', '.aac', '.m4a', '.wma', '.mp4'])

    for audio_file in input_folder.glob('*.*'):  # 遍历所有文件
        if audio_file.suffix.lower() in supported_formats:
            print(f"正在处理: {audio_file.name}")
            # 处理音频并获取事件时间和字幕
            event_times, subtitles = process_audio(audio_file, output_base,
                                                   silence_thresh,
                                                   silence_duration)
            # 存储结果
            all_event_times[audio_file] = event_times
            all_subtitles[audio_file] = subtitles

    return all_event_times, all_subtitles


# ======================
# 主程序入口（仅在直接运行此脚本时执行）
# ======================

if __name__ == '__main__':
    # 初始化FFmpeg检查
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
        exit(1)

    # 从配置文件中获取路径设置
    input_path_str = config.get("input_path")
    output_path_str = config.get("output_path")

    if not input_path_str or not output_path_str:
        print("配置文件中缺少输入或输出路径配置")
        exit(1)

    default_input = Path(input_path_str)
    default_output = Path(output_path_str)

    # 命令行参数解析
    parser = argparse.ArgumentParser(description='从音频文件中检测有声片段并进行语音识别')
    parser.add_argument(
        'input',
        nargs='?',  # 使参数可选
        default=str(default_input),
        help='输入音频文件夹路径')
    parser.add_argument(
        'output',
        nargs='?',  # 使参数可选
        default=str(default_output),
        help='输出目录路径（用于表格等输出文件）')
    parser.add_argument('--disable', action='store_true', help='禁用音频处理模块')

    args = parser.parse_args()

    # 如果禁用了模块，则直接返回
    if args.disable:
        print("音频处理模块已禁用")
        exit(0)

    # 转换为Path对象
    input_path = Path(args.input)
    output_path = Path(args.output)

    print(f"输入路径: {input_path}")
    print(f"输出路径: {output_path}")

    # 输入路径验证
    if not input_path.exists():
        print(f"输入路径不存在: {input_path}")
        exit(1)

    # 创建输出目录（如果不存在）
    if not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True)

    # 从配置中获取音频处理参数
    audio_processing_settings = config.get("modules",
                                           {}).get("audio_processing", {})
    silence_thresh = audio_processing_settings.get("silence_thresh",
                                                   -30)  # 默认值-30dB
    silence_duration = audio_processing_settings.get("silence_duration",
                                                     1.0)  # 默认值1.0秒

    print(f"使用静音阈值: {silence_thresh}dB")
    print(f"使用静音持续时间: {silence_duration}秒")

    # 判断输入是文件还是目录
    if input_path.is_file():
        # 处理单个音频文件
        print(f"处理单个音频文件: {input_path}")
        process_audio(input_path, output_path, silence_thresh,
                      silence_duration)
    else:
        # 处理文件夹内所有音频文件
        print(f"处理文件夹内所有音频文件: {input_path}")
        process_folder(input_path, output_path, silence_thresh,
                       silence_duration, config)
