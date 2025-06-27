from typing import Any

import numpy as np

# 预定义支持的采样率集合
SAMPLE_RATES: set[int] = {
    8000,  # 电话通信标准(ITU-T G.711,窄带语音)
    11025,  # 低比特率音频(如早期MP3/AM广播,1/4 CD采样率)
    12000,  # 语音压缩标准(如G.722.1宽带语音编码)
    16000,  # 语音识别/通信(如VAD/ASR系统,宽带语音)
    22050,  # 音频压缩(MPEG-1 Layer 3低码率模式,1/2 CD采样率)
    24000,  # 数字广播(如DAB+扩展模式)
    32000,  # 数字广播(DAB标准,FM数字广播)
    37800,  # Super Audio CD(SACD,DSD转PCM的中间采样率)
    44100,  # 音频CD标准(Red Book,消费级音质基准)
    48000,  # 数字音频/视频(DVD/专业录音,影视制作标准)
    88200,  # 高解析度音频(Hi-Res Audio,2倍CD采样率)
    96000,  # 专业录音(母带制作/多轨混音,4倍CD采样率)
    176400,  # DSD音频(Direct Stream Digital,4倍CD采样率的DSD格式)
    192000,  # 超高清音频(专业后期制作,11.2声道环绕声标准)
}

# 预定义支持的比特深度集合
BIT_DEPTHS: set[int] = {8, 16, 24, 32, 64}

# 映射比特深度字符串到实际数值的字典
BIT_DEPTH_MAP: dict[str, int] = {
    "u8": 8,  # 8 位无符号整数
    "uint8": 8,  # u8规范名称
    "s16": 16,  # 16 位有符号整数
    "int16": 16,  # s16规范名称
    "s24": 24,  # 24 位有符号整数
    "int24": 24,  # s24规范名称
    "s32": 32,  # 32 位有符号整数
    "int32": 32,  # s32规范名称
    "flt": 32,  # 32 位浮点数
    "float32": 32,  # flt规范名称
    "dbl": 64,  # 64 位浮点数
    "float64": 64,  # dbl规范名称
}


def get_audio_info_from_gradio(
    sampling_rate: int, wav_data: np.ndarray
) -> dict[str, Any]:
    """
    提取音频信息并返回包含采样率/比特深度/通道数/帧数和时长的字典.

    参数:
        sampling_rate (int): 采样率,必须在支持范围内.
        wav_data (np.ndarray): 音频数据,要求不包含 NaN 或 Inf.

    返回:
        dict[str, Any]: 包含以下键值:
            - "sampling_rate": 校验后的采样率
            - "bit_depth": 比特深度标识(如 "s16", "s24" 等)
            - "channels": 通道数
            - "frames": 帧数(样本总数)
            - "duration": 时长,单位为秒

    异常:
        TypeError: 当 wav_data 不是 np.ndarray 时.
        ValueError: 当采样率不支持,或数据超出范围,或音频类型不支持时.
    """
    if not isinstance(wav_data, np.ndarray):
        raise TypeError("音频数据必须是 np.ndarray")
    return {
        "sample_rate": check_sample_rate(sampling_rate),
        "sample_width": wav_data.dtype.itemsize,
        "bit_depth": get_bit_depth_from_gradio(wav_data),
        "channels": get_channels_from_gradio(wav_data),
        "frames": wav_data.shape[0],
        "duration": wav_data.shape[0] / sampling_rate + 0.0,
    }


def check_sample_rate(sampling_rate: int) -> int:
    """
    校验采样率是否在支持范围内.

    参数:
        sampling_rate (int): 要检验的采样率.

    返回:
        int: 如果采样率合法则返回该采样率.

    异常:
        ValueError: 当采样率不在支持集合中时.
    """
    if sampling_rate not in SAMPLE_RATES:
        raise ValueError(f"采样率 {sampling_rate} 不在支持范围内.")
    return sampling_rate


def get_channels_from_gradio(data: np.ndarray) -> int:
    """
    根据输入的音频数据获取通道数.

    参数:
        data (np.ndarray): 音频数据,维度为 1 时表示单通道,维度为 2 时第二维为通道.

    返回:
        int: 通道数.如果数据维度不为 1 或 2, 则抛出 ValueError.
    """
    if data.ndim == 1:
        return 1
    elif data.ndim == 2:
        return data.shape[1]
    else:
        raise ValueError("音频数据维度不支持")


def get_bit_depth_from_gradio(data: np.ndarray) -> str:
    """
    根据 numpy 数组的 dtype 判断音频数据的比特深度,
    并返回对应的标识符字符串(与 BIT_DEPTH_NAMES 中的键对应).

    参数:
        data (np.ndarray): 音频数据

    返回:
        str: 表示比特深度的字符串,如 "u8", "s16", "s24", "s32", "flt" 或 "dbl".

    异常:
        ValueError: 当输入数据类型不支持或浮点数据超出范围 [-1.0, 1.0] 时.
    """
    if data.dtype == np.int8:
        return "u8"
    elif data.dtype == np.int16:
        return "s16"
    elif data.dtype == np.int32:
        # 对于 32 位数据,
        # 如果数据实际只有低 24 位有效,即 24 位数据 ("s24"),
        # 否则认为是真正的 32 位数据 ("s32").
        # 实现说明:
        # 1. 利用 data.view(np.uint32) 将数据转换为无符号 32 位视图,避免因符号产生的问题.
        # 2. 使用位与运算 ( & 0xFF000000 ) 提取每个样本的高 8 位.
        # 3. 通过 .max() 检查所有样本的高 8 位是否都为 0.
        if (data.view(np.uint32) & 0xFF000000).max() == 0:
            return "s24"
        return "s32"
    elif data.dtype == np.float32 or data.dtype == np.float64:
        # 检查浮点数据是否在 [-1.0, 1.0] 范围内
        if not np.all((data >= -1.0) & (data <= 1.0)):
            raise ValueError("float 音频数据超出 [-1.0, 1.0] 范围")
        return "flt" if data.dtype == np.float32 else "dbl"
    else:
        raise ValueError("音频数据类型不支持")
