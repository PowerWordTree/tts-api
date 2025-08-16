from __future__ import annotations

import asyncio
from typing import (
    AsyncIterable,
    Callable,
    Final,
    Literal,
)

import janus
import lameenc
import numpy as np

from pwt.tts_api.wav_header import wav_header
from pwt.utils import ndarray_audio_utils

DEFAULT_LIMIT = 512
_EOF: Final = object()

AsyncContentStream = AsyncIterable[str | bytes | memoryview]


async def queue_iter(queue: janus.Queue) -> AsyncContentStream:
    try:
        while True:
            data = await queue.async_q.get()
            if data is _EOF:
                break
            if isinstance(data, Exception):
                raise data
            yield data
    except (janus.ShutDown, janus.QueueShutDown):
        pass
    except BaseException:
        raise asyncio.CancelledError()
    finally:
        await queue.aclose()


def make_done_callback(queue: janus.Queue) -> Callable[[asyncio.Task[None]], None]:
    def done_callback(task: asyncio.Task[None]) -> None:
        try:
            task.result()
        except (janus.ShutDown, janus.QueueShutDown):
            pass
        except BaseException as exc:
            queue.sync_q.put(exc)
        finally:
            queue.shutdown(False)

    return done_callback


# fmt: off
# 预定义MP3帧大小(单位: 采样点数)
MP3_SAMPLES_PER_FRAME = 1152
# MP3标准比特率列表(单位: kbps)
MP3_BITRATE_LIST = [8,16,24,32,40,48,56,64,80,96,112,128,160,192,224,256,320]
# fmt: on


def infer_mp3_bitrate(sample_rate: int, channels: Literal[1, 2]) -> int:
    """
    推荐公式:
        bitrate = min(320, round((sample_rate / 1000) * 4 * channels))
    说明:
        sample_rate / 1000: 把采样率归一化到 kHz.
        * 4: 每 kHz 单声道推荐 4kbps(实际语音可用3~4, 音乐常用4~6), 这里选偏高一点.
        * channels: 单声道 x1, 双声道 x2.
        round: 取整, 方便后续与标准比特率列表比对.
        min(320, ...): 最高不超过320kbps(MP3标准上限).
    示例:
        [8,16,24,32,40,48,56,64,80,96,112,128,160,192,224,256,320]
        print(auto_mp3_bitrate(24000, 1))  # 96
        print(auto_mp3_bitrate(24000, 2))  # 128
        print(auto_mp3_bitrate(44100, 1))  # 176->192
        print(auto_mp3_bitrate(44100, 2))  # 352->320
    """
    # 计算推荐比特率
    bitrate = int(round((sample_rate / 1000) * 4 * channels))
    # 限制最高320
    bitrate = min(320, max(8, bitrate))
    # 匹配到最近的标准值(向上取)
    for std in MP3_BITRATE_LIST:
        if bitrate <= std:
            return std
    return MP3_BITRATE_LIST[-1]  # fallback


async def mp3_encoder(
    wav_data: np.ndarray, sample_rate: int, frames_per_chunk: int = 4
) -> AsyncContentStream:
    queue = janus.Queue(maxsize=DEFAULT_LIMIT)

    def encode() -> None:
        nonlocal wav_data

        # 探测元数据
        meta = ndarray_audio_utils.probe_audio_metadata(sample_rate, wav_data)

        # 规整音频数据
        min_frames = MP3_SAMPLES_PER_FRAME * 2
        if meta["num_frames"] < min_frames:
            wav_data = ndarray_audio_utils.pad_to_min_frames(
                wav_data, meta["data_layout"], min_frames=min_frames
            )
        if meta["num_channels"] > 2:
            wav_data = ndarray_audio_utils.trim_channels(wav_data, meta["data_layout"])
        if meta["data_layout"] != "interleaved":
            wav_data = ndarray_audio_utils.convert_data_layout(wav_data, "interleaved")
        if meta["sample_type"] != "s16":
            wav_data = ndarray_audio_utils.convert_pcm16(wav_data, meta["sample_type"])

        # 重新探测并推导比特率
        meta = ndarray_audio_utils.probe_audio_metadata(sample_rate, wav_data)
        bit_rate = infer_mp3_bitrate(sample_rate, meta["num_channels"])

        # 初始化编码器
        encoder = lameenc.Encoder()
        encoder.set_bit_rate(bit_rate)
        encoder.set_in_sample_rate(sample_rate)
        encoder.set_channels(meta["num_channels"])
        encoder.set_quality(2)  # 0=best, 2=fast, 7=fastest/low-quality

        # 建立字节视图
        wav_data = np.ascontiguousarray(wav_data, dtype="<i2")
        mv = memoryview(wav_data).cast("B")
        frame_bytes = meta["num_channels"] * meta["bytes_per_sample"]

        # 开始编码
        step = MP3_SAMPLES_PER_FRAME * frame_bytes * frames_per_chunk
        for start in range(0, mv.nbytes, step):
            if chunk := encoder.encode(mv[start : start + step].tobytes()):
                queue.sync_q.put(memoryview(chunk))

        if chunk := encoder.flush():
            queue.sync_q.put(memoryview(chunk))

        queue.sync_q.put(_EOF)

    task = asyncio.create_task(asyncio.to_thread(encode))
    task.add_done_callback(make_done_callback(queue))
    return queue_iter(queue)


WAV_SUPPORTED = {"u8", "s16", "s24", "s32", "flt", "dbl"}


async def wav_encoder(
    wav_data: np.ndarray, sample_rate: int, chunk_size: int = 1024
) -> AsyncContentStream:
    queue = janus.Queue(maxsize=DEFAULT_LIMIT)

    def encode() -> None:
        nonlocal wav_data

        # 获取元数据
        meta = ndarray_audio_utils.probe_audio_metadata(sample_rate, wav_data)

        # 规整音频数据
        if meta["num_frames"] < 2048:
            wav_data = ndarray_audio_utils.pad_to_min_frames(
                wav_data, meta["data_layout"]
            )
        if meta["data_layout"] != "interleaved":
            wav_data = ndarray_audio_utils.convert_data_layout(wav_data, "interleaved")
        if meta["sample_type"] not in WAV_SUPPORTED:
            wav_data = ndarray_audio_utils.convert_pcm16(wav_data, meta["sample_type"])

        # 重新获取元数据
        meta = ndarray_audio_utils.probe_audio_metadata(sample_rate, wav_data)

        # 全局统一为小端 & C 连续(一次性)
        wav_data = np.ascontiguousarray(
            wav_data,
            dtype=wav_data.dtype.newbyteorder("<"),
        )

        # s24: 推荐在此一步预打包为最终落盘字节流(一次性)
        if meta["sample_type"] == "s24":
            # 确保小端 int32 且 C 连续
            wav_data = np.ascontiguousarray(wav_data, dtype="<i4")
            # 视作 [*, 4] 的字节视图(纯视图, 不复制)
            u8 = wav_data.view(np.uint8).reshape(-1, 4)
            # 直接一次性拷贝到最终 3 字节平铺缓冲
            wav_data = np.empty(u8.shape[0] * 3, dtype=np.uint8)
            wav_data.reshape(-1, 3)[:] = u8[:, :3]  # 单次拷贝

        # 建立字节视图
        mv = memoryview(wav_data).cast("B")
        frame_bytes = meta["num_channels"] * meta["bytes_per_sample"]

        # 写头
        header = wav_header(
            meta["sample_type"],
            meta["sample_rate"],
            meta["num_frames"],
            meta["num_channels"],
        )
        queue.sync_q.put(header)

        # 分块发送(真零拷贝)
        step = chunk_size * frame_bytes
        for start in range(0, mv.nbytes, step):
            queue.sync_q.put(mv[start : start + step])

        # 偶数字节对齐填充
        if mv.nbytes % 2 == 1:
            queue.sync_q.put(b"\x00")

        queue.sync_q.put(_EOF)

    task = asyncio.create_task(asyncio.to_thread(encode))
    task.add_done_callback(make_done_callback(queue))
    return queue_iter(queue)
