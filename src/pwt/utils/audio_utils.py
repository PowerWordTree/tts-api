# fmt: off
# 预定义支持的采样率集合
SUPPORTED_SAMPLE_RATES: set[int] = {
    8000,    # 电话通信标准(ITU-T G.711,窄带语音)
    11025,   # 低比特率音频(如早期MP3/AM广播,1/4 CD采样率)
    12000,   # 语音压缩标准(如G.722.1宽带语音编码)
    16000,   # 语音识别/通信(如VAD/ASR系统,宽带语音)
    22050,   # 音频压缩(MPEG-1 Layer 3低码率模式,1/2 CD采样率)
    24000,   # 数字广播(如DAB+扩展模式)
    32000,   # 数字广播(DAB标准,FM数字广播)
    37800,   # Super Audio CD(SACD,DSD转PCM的中间采样率)
    44100,   # 音频CD标准(Red Book,消费级音质基准)
    48000,   # 数字音频/视频(DVD/专业录音,影视制作标准)
    88200,   # 高解析度音频(Hi-Res Audio,2倍CD采样率)
    96000,   # 专业录音(母带制作/多轨混音,4倍CD采样率)
    176400,  # DSD音频(Direct Stream Digital,4倍CD采样率的DSD格式)
    192000,  # 超高清音频(专业后期制作,11.2声道环绕声标准)
}

# 预定义支持的样本深度比特数集合
SUPPORTED_SAMPLE_BITS: set[int] = {8, 16, 24, 32, 64}

# 样本格式字符串 -> 对应 BitsPerSample 数值映射表
SAMPLE_TYPE_BITS: dict[str, int] = {
    "u8":      8,   # 8位无符号整数
    "uint8":   8,   # u8规范名称
    "s8":      8,   # 8位有符号整数
    "int8":    8,   # s8规范名称
    "s16":     16,  # 16位有符号整数
    "int16":   16,  # s16规范名称
    "s24":     24,  # 24位有符号整数
    "int24":   24,  # s24规范名称
    "s32":     32,  # 32位有符号整数
    "int32":   32,  # s32规范名称
    "flt":     32,  # 32位浮点数
    "float32": 32,  # flt规范名称
    "dbl":     64,  # 64位浮点数
    "float64": 64,  # dbl规范名称
}
# fmt: on


def check_sample_rate(sample_rate: int) -> int:
    """
    校验采样率是否在支持范围内.

    参数:
        sample_rate (int): 要检验的采样率.

    返回:
        int: 如果采样率合法则返回该采样率.

    异常:
        ValueError: 当采样率不在支持集合中时.
    """
    if sample_rate not in SUPPORTED_SAMPLE_RATES:
        raise ValueError(f"采样率 {sample_rate} 不在支持范围内.")
    return sample_rate
