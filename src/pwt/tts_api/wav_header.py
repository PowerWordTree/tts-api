from __future__ import annotations

import struct
from typing import Any, Literal, NamedTuple


class SampleInfo(NamedTuple):
    format_id: str
    bits_per_sample: int
    bytes_per_sample: int
    wave_tag: int
    wave_guid: bytes


class Field(NamedTuple):
    name: str
    fmt: str
    value: Any = None


class Chunk:
    def __init__(
        self,
        *fields: Field,
        byteorder: Literal["@", "=", "<", ">", "!"],
    ) -> None:
        self.byteorder = byteorder
        self.fields = fields
        self.struct = struct.Struct(byteorder + "".join(field.fmt for field in fields))

    @property
    def format(self) -> str:
        return self.struct.format

    @property
    def size(self) -> int:
        return self.struct.size

    def pack(self, params: dict[str, Any]) -> bytes:
        values = []
        for field in self.fields:
            value = params.get(field.name)
            if value is None:  # 外部没传或显式传 None → 用模板默认值
                value = field.value
            values.append(value)
        return self.struct.pack(*values)

    def unpack(self, buffer: bytes | bytearray | memoryview) -> dict[str, Any]:
        datas = self.struct.unpack(buffer)
        result = {field.name: data for data, field in zip(datas, self.fields)}
        return result


"""
|--------|--------------------|----------------|----------|----------------------------------------|
| 格式码 | C类型              | Python类型     | 字节数   | 说明                                   |
|--------|--------------------|----------------|----------|----------------------------------------|
| `x`    | pad byte           | 无             | 1        | 填充字节, 无值                         |
| `c`    | char               | `bytes(1)`     | 1        | 单个字节                               |
| `b`    | signed char        | `int`          | 1        | 有符号8位整数                          |
| `B`    | unsigned char      | `int`          | 1        | 无符号8位整数                          |
| `?`    | _Bool              | `bool`         | 1        | 布尔值                                 |
| `h`    | short              | `int`          | 2        | 有符号16位整数                         |
| `H`    | unsigned short     | `int`          | 2        | 无符号16位整数                         |
| `i`    | int                | `int`          | 4        | 有符号32位整数                         |
| `I`    | unsigned int       | `int`          | 4        | 无符号32位整数                         |
| `l`    | long               | `int`          | 4        | 有符号32位整数(C long)                 |
| `L`    | unsigned long      | `int`          | 4        | 无符号32位整数                         |
| `q`    | long long          | `int`          | 8        | 有符号64位整数                         |
| `Q`    | unsigned long long | `int`          | 8        | 无符号64位整数                         |
| `n`    | ssize_t            | `int`          | 机器相关 | 本机字长                               |
| `N`    | size_t             | `int`          | 机器相关 | 本机字长                               |
| `e`    | (IEEE 754) float16 | `float`        | 2        | 半精度浮点                             |
| `f`    | float              | `float`        | 4        | 单精度浮点                             |
| `d`    | double             | `float`        | 8        | 双精度浮点                             |
| `s`    | char[]             | `bytes`        | 按长度   | 定长字节串(格式前可指定长度, 如 `10s`) |
| `p`    | char[]             | `bytes`        | 按长度   | Pascal风格字节串(首字节为长度)         |
| `P`    | void*              | `int`          | 机器相关 | 指针大小(依平台)                       |
|--------|--------------------|----------------|----------|----------------------------------------|

'FL':  1 << 0,  'FR':  1 << 1,  'FC':  1 << 2,  'LFE': 1 << 3,
'BL':  1 << 4,  'BR':  1 << 5,  'FLC': 1 << 6,  'FRC': 1 << 7,
'BC':  1 << 8,  'SL':  1 << 9,  'SR':  1 << 10,
'TFC': 1 << 11, 'TFL': 1 << 12, 'TFc': 1 << 13, 'TFR': 1 << 14,
'TBL': 1 << 15, 'TBc': 1 << 16, 'TBR': 1 << 17,
"""

# fmt: off
BYTEORDER_SET = {"@", "=", "<", ">", "!"}

# PCM: {00000001-0000-0010-8000-00AA00389B71}
WAVE_SUBTYPE_PCM = bytes((
    0x01, 0x00, 0x00, 0x00,
    0x00, 0x00,
    0x10, 0x00,
    0x80, 0x00,
    0x00, 0xAA, 0x00, 0x38, 0x9B, 0x71
))

# IEEE Float: {00000003-0000-0010-8000-00AA00389B71}
WAVE_SUBTYPE_FLOAT = bytes((
    0x03, 0x00, 0x00, 0x00,
    0x00, 0x00,
    0x10, 0x00,
    0x80, 0x00,
    0x00, 0xAA, 0x00, 0x38, 0x9B, 0x71
))

SAMPLE_TYPES: dict[str, SampleInfo] = {
    # PCM
    "u8" : SampleInfo("uint8",   8, 1, 1, WAVE_SUBTYPE_PCM),
    "s16": SampleInfo("int16",   16, 2, 1, WAVE_SUBTYPE_PCM),
    "s24": SampleInfo("int24",   24, 3, 1, WAVE_SUBTYPE_PCM),
    "s32": SampleInfo("int32",   32, 4, 1, WAVE_SUBTYPE_PCM),
    # IEEE Float
    "flt": SampleInfo("float32", 32, 4, 3, WAVE_SUBTYPE_FLOAT),
    "dbl": SampleInfo("float64", 64, 8, 3, WAVE_SUBTYPE_FLOAT),
}

# 全部Chunk需要2字节对齐, xxx.size是单数, 需在末尾补 0x00 , 不用修改size, 除了riff.size
# 所有 Chunk 的数据区需 2 字节对齐. 当 data.size 或其他 chunk.size 为奇数时,
# 在末尾补 0x00, 但 size 字段值本身不变(除 riff.size).
# riff.size 表示整个文件长度减去 8, 包括所有 chunk 和 padding.

# -----------------------
# RIFF Chunk (大于4GB文件时: id="RF64", size=0xFFFFFFFF)
# -----------------------
WAVE_RIFF_CHUNK = Chunk(                   # 12 bytes
    Field("riff.id",      "4s",  b"RIFF"),  # 标识符, "RIFF"文件小于4G, "RF64"文件大于4G
    Field("riff.size",    "I"),             # 总文件字节数 - 8; RF64 时为 0xFFFFFFFF
    Field("riff.format",  "4s",  b"WAVE"),  # 格式类型标识(WAVE)
    byteorder="<",
)

# -----------------------
# ds64 Chunk(RF64 专用, 定义真实大小)
# -----------------------
WAVE_DS64_CHUNK = Chunk(                          # 32 bytes
    Field("ds64.id",            "4s",  b"ds64"),  # ds64 子块标识
    Field("ds64.size",          "I",   24),       # ds64 子块长度
    Field("ds64.riff_size",     "Q"),             # 实际 RIFF 尺寸(uint64)
    Field("ds64.data_size",     "Q"),             # 实际 data 区尺寸(uint64)
    Field("ds64.fact_samples",  "Q"),             # fact 样本总数(uint64)
    # 可能跟随多个表项(可选)
    byteorder="<",
)

# -----------------------
# JUNK Chunk(可选, 用于对齐或预留空间)
# -----------------------
WAVE_JUNK_CHUNK = Chunk(                  # 8 bytes
    Field("junk.id",    "4s",  b"JUNK"),  # 填充块标识
    Field("junk.size",  "I",   0x0000),   # 填充数据长度
    # 填充字节, 填充数据长度偶数, 内容可为 0x00 或 0x20
    byteorder="<",
)

# -----------------------
# fmt Chunk(PCM/压缩格式通用)
# -----------------------
WAVE_FMT_SUBCHUNK = Chunk(                          # 24 bytes
    Field("fmt.id",               "4s",  b"fmt "),  # 格式子块标识
    Field("fmt.size",             "I",   16),       # 子块长度, PCM 固定为 16; 扩展格式一般为 40
    Field("fmt.audio_format",     "H"),             # 音频编码, 1=PCM, 3=IEEE Float, 65534=WAVE_EXTENSIBLE
    Field("fmt.num_channels",     "H"),             # 声道数
    Field("fmt.sample_rate",      "I"),             # 采样率
    Field("fmt.byte_rate",        "I"),             # 每秒数据字节数
    Field("fmt.block_align",      "H"),             # 每个采样帧字节数
    Field("fmt.bits_per_sample",  "H"),             # 每样本位数
    byteorder="<",
)

# -----------------------
# fmt extensible Chunk(专用, fmt.audio_format非PCM使用)
# -----------------------
WAVE_FMT_EXTENSION = Chunk(                          # 24 bytes
    Field("fmt.cb_size",                "H",  22),   # 扩展信息字节数(一般22, 不含前 16 字节)
    Field("fmt.valid_bits_per_sample",  "H"),        # 实际有效位数(通常等于 bits_per_sample)
    Field("fmt.channel_mask",           "I"),        # 扬声器通道掩码(各声道位标志组合)
    Field("fmt.subformat",              "16s"),      # 子格式 GUID(PCM/Float 等)
    # 其他厂商自定义扩展字段可在此追加
    byteorder="<",
)

# -----------------------
# fact Chunk(可选, 非PCM或有额外信息时, 常用于压缩格式)
# -----------------------
WAVE_FACT_CHUNK = Chunk(                           # 12 bytes
    Field("fact.id",             "4s",  b"fact"),  # fact 子块标识
    Field("fact.size",           "I",   4),        # 子块字节数
    Field("fact.sample_length",  "I"),             # 总采样帧数(total sample frames)
    # 厂商可扩展其他字段
    byteorder="<",
)

# -----------------------
# data Chunk(实际音频数据)
# -----------------------
WAVE_DATA_SUBCHUNK = Chunk(               # 8 bytes
    Field("data.id",    "4s",  b"data"),  # 音频数据标识
    Field("data.size",  "I"),             # 音频数据字节数; RF64 时为 0xFFFFFFFF
    # 紧随其后为原始音频数据, 若字节数为奇数需在数据末尾补 0x00
    byteorder="<",
)
# fmt: on


def wav_header(
    sample_type: Literal["u8", "s16", "s24", "s32", "flt", "dbl"],
    sample_rate: int,
    num_frames: int,
    num_channels: int,
    channel_mask: int = 0,
) -> bytes:
    sample_info = SAMPLE_TYPES[sample_type]
    num_samples = num_frames * num_channels
    data_size = num_samples * sample_info.bytes_per_sample
    data_size_aligned = (data_size + 1) & ~1  # 确保偶数
    byte_rate = sample_rate * num_channels * sample_info.bytes_per_sample
    block_align = num_channels * sample_info.bytes_per_sample
    has_ds64 = data_size > 0xFF000000
    has_fmt_ext = num_channels > 2 or channel_mask != 0
    chunks = [
        WAVE_RIFF_CHUNK,
        *([WAVE_DS64_CHUNK] if has_ds64 else []),
        WAVE_FMT_SUBCHUNK,
        *([WAVE_FMT_EXTENSION] if has_fmt_ext else []),
        WAVE_DATA_SUBCHUNK,
    ]
    riff_size = sum(chunk.size for chunk in chunks) - 8 + data_size_aligned
    context = {}
    # RIFF_HEADER
    if WAVE_RIFF_CHUNK in chunks:
        context.update(
            {
                "riff.id": b"RF64" if has_ds64 else b"RIFF",
                "riff.size": 0xFFFFFFFF if has_ds64 else riff_size,
            }
        )
    # DS64_CHUNK
    if WAVE_DS64_CHUNK in chunks:
        context.update(
            {
                "ds64.riff_size": riff_size,
                "ds64.data_size": data_size,
                "ds64.fact_samples": num_samples,
            }
        )
    # FMT_SUBCHUNK
    if WAVE_FMT_SUBCHUNK in chunks:
        context.update(
            {
                "fmt.size": 40 if has_fmt_ext else 16,
                "fmt.audio_format": 0xFFFE if has_fmt_ext else sample_info.wave_tag,
                "fmt.num_channels": num_channels,
                "fmt.sample_rate": sample_rate,
                "fmt.byte_rate": byte_rate,
                "fmt.block_align": block_align,
                "fmt.bits_per_sample": sample_info.bits_per_sample,
            }
        )
    # FMT_EXTENSION
    if WAVE_FMT_EXTENSION in chunks:
        context.update(
            {
                "fmt.valid_bits_per_sample": sample_info.bits_per_sample,
                "fmt.channel_mask": channel_mask,
                "fmt.subformat": sample_info.wave_guid,
            }
        )
    # DATA_SUBCHUNK
    if WAVE_DATA_SUBCHUNK in chunks:
        context.update({"data.size": 0xFFFFFFFF if has_ds64 else data_size})
    # 打包生成并返回
    return b"".join(chunk.pack(context) for chunk in chunks)
