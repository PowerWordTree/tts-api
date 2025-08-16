from typing import Any, Literal

import numpy as np

from pwt.utils import audio_utils

# fmt: off
# 预定义支持的信号维度集合
AUDIO_DATA_AXES: set[Literal["frames", "channels"]] = {
    "frames",    # 采样帧数(frame = 所有声道一个采样点的集合)
    "channels",  # 声道数
}

# 预定义支持的采样数据布局集合
AUDIO_DATA_LAYOUTS: dict[
    Literal["interleaved", "planar", "mono"], tuple[Literal["frames", "channels"], ...]
] = {
    "interleaved": ("frames",   "channels"),  # 交错布局, 常见于主流音频文件和 API
    "planar":      ("channels", "frames"),    # 平面布局, 多见于编解码器或内存优化场景
    "mono":        ("frames",),               # 单声道布局, 常用于语音处理
}
# fmt: on


def infer_data_layout(data: np.ndarray) -> Literal["interleaved", "planar", "mono"]:
    """
    推断音频数据的布局类型(启发式判断).

    根据输入数组的维度和形状, 尝试推断其可能的音频布局类型.
    注意: 该函数仅用于启发式猜测, 无法在帧数较少或形状模糊时保证准确性.

    参数:
        data (np.ndarray): 输入的音频数据数组, 可以是一维或二维.

    返回:
        Literal["interleaved", "planar", "mono"]:
            - "mono": 如果是单通道(一维数组).
            - "interleaved": 如果形状为 (样本数, 通道数), 且样本数较多.
            - "planar": 如果形状为 (通道数, 样本数), 且样本数较多.

    异常:
        ValueError: 如果输入数组维度超过二维, 或帧数太少无法可靠判断.
    """
    if data.ndim == 1:
        return "mono"
    elif data.ndim == 2:
        if max(data.shape) > 32:
            return "planar" if data.shape[0] < data.shape[1] else "interleaved"
        raise ValueError("帧数太少, 无法可靠判断布局")
    raise ValueError("音频数据维度不支持")


def convert_data_layout(
    data: np.ndarray, layout: Literal["interleaved", "planar"]
) -> np.ndarray:
    """
    转换音频数据的布局类型.
    根据指定的布局类型, 将输入音频数据从一种布局转换为另一种.
    支持的布局类型包括:
        - "interleaved": 交错布局, 常见于主流音频格式如 WAV/MP3.
        - "planar": 平面布局, 常用于编解码器内部或大模型中间表示.

    参数:
        data (np.ndarray): 输入的音频数据数组, 可以是一维或二维.
        layout (Literal["interleaved", "planar"]): 目标布局类型.

    返回:
        np.ndarray: 转换后的音频数据数组, 形状与输入保持一致.

    异常:
        ValueError: 如果输入数组维度超过二维, 或帧数太少无法可靠判断.
    """
    data_layout = infer_data_layout(data)
    if data_layout == layout:
        return data
    if data_layout == "mono":
        return data[:, None] if layout == "interleaved" else data[None, :]
    return data.T


def pad_to_min_frames(
    data: np.ndarray,
    layout: Literal["interleaved", "planar", "mono"],
    *,
    min_frames: int = 2048,
) -> np.ndarray:
    """
    保证音频数据满足最小帧数要求(2048帧), 不足则进行零填充.

    根据音频数据的布局类型, 判断对应轴上的帧数是否满足最小阈值.
    若不足, 则在末尾进行零填充以扩展到所需帧长. 该操作可用于
    滤波器预热/窗口函数应用/推理模型兼容等场景.

    参数:
        data (np.ndarray): 输入的音频数据数组, 可为一维或二维.
        layout (Literal["interleaved", "planar", "mono"]):
            音频数据的布局类型, 用于确定帧轴.
            - "mono": 单通道, 一维数组, 轴为 0.
            - "interleaved": 形状为 (样本数, 通道数), 轴为 0.
            - "planar": 形状为 (通道数, 样本数), 轴为 1.

    返回:
        np.ndarray: 原始或已填充的音频数据, 维度保持不变.
    """
    axis = 1 if layout == "planar" else 0
    current_frames = data.shape[axis]
    if current_frames >= min_frames:
        return data
    pad_width = [(0, 0)] * data.ndim
    pad_width[axis] = (0, min_frames - current_frames)
    return np.pad(data, pad_width, mode="constant", constant_values=0)


def probe_audio_metadata(sample_rate: int, audio_data: np.ndarray) -> dict[str, Any]:
    """
    从给定的音频 ndarray 中提取协议层可用的元数据, 集中生成采样/
    位深/声道/帧数/字节序等关键信息, 便于后续跨协议(如 WAV/RF64/MP3 等)
    的头部构造与参数判断.

    参数:
        sample_rate (int): 目标采样率 (Hz), 会经过范围及有效性检查.
        audio_data (np.ndarray): 音频数据数组, 要求为数值类型, 且不包含 NaN 或 Inf.

    返回:
        dict[str, Any]: 元数据字典, 包含以下键:
            - "data_layout": 数据布局描述(如 'interleaved' / 'planar' / 'mono')
            - "byte_order": 样本数据在内存中的字节序标记
                            ('<'=little-endian, '>'=big-endian, '='=本机, '|'=无关)
            - "sample_rate": 校验后的采样率 (Hz)
            - "sample_type": 样本类型标识(如 "s16", "s24", "flt")
            - "bits_per_sample": 每个样本的位数(协议层定义)
            - "bytes_per_sample": 每个样本的字节数(bits_per_sample // 8)
            - "num_channels": 通道数
            - "num_frames": 总帧数(每帧包含所有通道各 1 个样本)
            - "duration_seconds": 音频总时长(秒)

    异常:
        TypeError: 当 audio_data 不是 np.ndarray 类型时抛出.
        ValueError: 当采样率不受支持/数据类型不支持或数据值超出范围时抛出.
    """
    if not isinstance(audio_data, np.ndarray):
        raise TypeError("音频数据必须是 np.ndarray")

    metadata = {}

    data_layout = infer_data_layout(audio_data)
    metadata["data_layout"] = data_layout

    byte_order = audio_data.dtype.byteorder
    metadata["byte_order"] = byte_order

    sample_rate = audio_utils.check_sample_rate(sample_rate)
    metadata["sample_rate"] = sample_rate

    sample_type = get_sample_type(audio_data)
    metadata["sample_type"] = sample_type

    bits_per_sample = audio_utils.SAMPLE_TYPE_BITS[sample_type]
    metadata["bits_per_sample"] = bits_per_sample

    bytes_per_sample = bits_per_sample // 8
    metadata["bytes_per_sample"] = bytes_per_sample

    num_channels = get_channels(audio_data)
    metadata["num_channels"] = num_channels

    num_frames = get_frames(audio_data, data_layout)
    metadata["num_frames"] = num_frames

    duration_seconds = num_frames / sample_rate
    metadata["duration_seconds"] = duration_seconds

    return metadata


def get_channels(data: np.ndarray) -> int:
    """
    根据输入的音频数据获取通道数.

    参数:
        data (np.ndarray): 输入的音频数据数组, 可以是一维或二维

    返回:
        int: 通道数.如果数据维度不为 1 或 2, 则抛出 ValueError.
    """
    if data.ndim == 1:
        return 1
    elif data.ndim == 2:
        if max(data.shape) > 32:
            return min(data.shape)
        raise ValueError("帧数太少, 无法可靠判断布局")
    raise ValueError("音频数据维度不支持")


def get_frames(
    data: np.ndarray, layout: Literal["interleaved", "planar", "mono"]
) -> int:
    """
    根据输入的音频数据和布局类型获取帧数.

    参数:
        data (np.ndarray): 输入的音频数据数组, 可以是一维或二维
        layout (Literal["interleaved", "planar", "mono"]):
            音频数据的布局类型, 用于确定帧轴.
            - "interleaved": 形状为 (样本数, 通道数), 轴为 0.
            - "planar": 形状为 (通道数, 样本数), 轴为 1.
            - "mono": 单通道, 一维数组, 轴为 0.

    返回:
        int: 帧数.
    """
    if layout == "planar":
        return data.shape[1]
    return data.shape[0]


def get_sample_type(data: np.ndarray) -> str:
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
    if data.dtype == np.uint8:
        return "u8"
    elif data.dtype == np.int8:
        return "s8"
    elif data.dtype == np.int16:
        return "s16"
    elif data.dtype == np.int32:
        # 对于 32 位数据,
        # 如果数据实际只有低 24 位有效,即 24 位数据 ("s24"),
        # 否则认为是真正的 32 位数据 ("s32").
        # 实现说明:
        # 1. 利用 data.view(np.uint32) 将数据转换为无符号 32 位视图.
        # 2. 使用位与运算 >> 24 提取每个样本的高 8 位.
        # 3. 检查所有样本的高 8 位为 0 | 0xFF 时为 s24.
        high8 = data.view(np.uint32) >> 24
        if np.isin(high8, (0x00, 0xFF)).all():
            return "s24"
        return "s32"
    elif data.dtype == np.float32 or data.dtype == np.float64:
        # 检查浮点数据是否在 [-1.0, 1.0] 范围内
        # if not np.all((data >= -1.0) & (data <= 1.0)):
        if data.min() < -1.0 or data.max() > 1.0:
            raise ValueError("float 音频数据超出 [-1.0, 1.0] 范围")
        return "flt" if data.dtype == np.float32 else "dbl"
    else:
        raise ValueError("音频数据类型不支持")


def convert_pcm16(wav_data: np.ndarray, sample_type: str) -> np.ndarray:
    """
    将不同位深的音频数据转换为 int16 格式.

    支持从 uint8/int24/int32/float32/float64 等常见格式转换为 int16.
    对于高位深数据(如 int24/int32), 会使用 TPDF 抖动以减少量化误差.

    参数:
        wav_data (np.ndarray): 输入的音频数据数组, 类型和范围取决于 bit_depth.
        sample_type (str): 输入数据的位深类型, 可选值包括:
            - "u8", "uint8": 无符号 8 位整数, 范围 [0, 255]
            - "s8", "int8": 有符号 8 位整数, 范围 [-128, 127]
            - "s24", "int24": 有符号 24 位整数(以 int32 表示)
            - "s32", "int32": 有符号 32 位整数
            - "flt", "float32": 浮点数, 范围 [-1.0, 1.0]
            - "dbl", "float64": 双精度浮点数, 范围 [-1.0, 1.0]

    返回:
        np.ndarray: 转换后的 int16 音频数据, 范围 [-32768, 32767].
    """
    sample_type = sample_type.lower()

    if sample_type in ("u8", "uint8"):
        # uint8 [0,255] -> int16 [-32768,32767]
        return (wav_data.astype(np.int16) - 128) << 8
    if sample_type in ("s8", "int8"):
        # int8 [-128,127] -> int16 [-32768,32767]
        return wav_data.astype(np.int16) << 8
    if sample_type in ("s24", "int24"):
        # 右移8位转24bit->16bit
        return np.clip(wav_data >> 8, -32768, 32767).astype(np.int16)
    if sample_type in ("s32", "int32"):
        # 右移16位转int32 -> int16
        return np.clip(wav_data > 16, -32768, 32767).astype(np.int16)
    if sample_type in ("flt", "float32", "dbl", "float64"):
        # float [-1,1] -> int16
        return (wav_data * 32767).astype(np.int16)
    return wav_data


def trim_channels(
    wav_data: np.ndarray,
    layout: Literal["interleaved", "planar", "mono"],
    *,
    keep: int = 2,
) -> np.ndarray:
    """
    截取指定数量的声道并返回视图(零拷贝操作).

    用途:
        当多声道音频中只需保留前 `keep` 个声道时, 可在不复制数据的情况下裁剪声道.

    核心流程:
        1. 构造"全维度全取"的切片表(全部用 slice(None) 占位)
        2. 根据布局确定声道所在的维度索引
        3. 将声道维的切片替换为 slice(0, keep)
        4. 转为 tuple 传入索引, 确保触发标准切片(避免 NumPy 花式索引复制数据)

    参数:
        wav_data : np.ndarray
            输入的音频数据数组, 支持不同布局和维度.
        layout : {"interleaved", "planar", "mono"}
            数据布局:
              - "interleaved": 声道交错, 例如 (frames, channels)
              - "planar"     : 声道分平面, 例如 (channels, frames)
              - "mono"       : 单声道
        keep : int, default=2
            要保留的声道数(从第 0 声道开始计)

    返回:
        np.ndarray
            截取后的音频数据视图(零拷贝)
    """
    if layout == "mono" or wav_data.ndim < 2:
        return wav_data

    # 根据layout确定声道轴
    axis = 1 if layout == "interleaved" else 0

    # 生成切片表
    # slice(None) == ":"  → 全取
    # [slice(None)] * x.ndim 生成一个列表, 长度 = 维度数, 每个元素都是全取
    slc = [slice(None)] * wav_data.ndim

    # 修改声道轴的切片
    # 声道轴位置是axis, 把它的切片替换为 slice(0, keep)
    # 也就是 "只取前 keep 个声道"
    slc[axis] = slice(0, keep)

    # 执行切片
    # 这里必须 tuple(slc), 因为:
    # - 列表作为索引会触发 NumPy 的花式索引(可能复制数据, 语义不同)
    # - 元组才是"多维切片"的正统写法, 才会按照每个维度对应一个切片来处理
    return wav_data[tuple(slc)]

