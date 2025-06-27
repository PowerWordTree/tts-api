import asyncio
import io

import ffmpeg.audio
import numpy as np
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

app = FastAPI()

"""
哪那么麻烦啊,维护两个有序列表,任务来了,按照进入列表顺序反向遍历,找到标签相同的就取走放进运行中的列表,开始执行.. 如果没找到,就把最后遍历的线程修改标签,取出执行任务...
"""

"""
loop.call_soon_threadsafe(handle_new_data, data)
asyncio.run_coroutine_threadsafe(queue.put(data), loop)


函数	是否阻塞	返回值	适用场景
call_soon_threadsafe()	❌ 非阻塞	None	快速触发轻量操作(无结果)
run_coroutine_threadsafe()	❌ 提交非阻塞<br>✅ .result() 阻塞	Future	跨线程调度协程并取回结果

像 janus.Queue 是双端队列结构,一端是线程安全接口/一端是异步接口,用来做:
高吞吐场景(C扩展更快);
投递状态监控/阻塞检测;
自动清理/完成通知; 对于工业级音视频/日志/行情等系统,是有价值的.

"""

@app.get("/audio_mp3_raw")
async def get_mp3_audio_raw():
    # 假设生成 1 秒钟的单声道 22050Hz 的 int16 PCM 数据
    sampling_rate = 22050
    pcm_data = np.random.randint(-32768, 32767, sampling_rate, dtype=np.int16)

    # 用 BytesIO 包装原始 PCM 数据
    pcm_io = io.BytesIO(pcm_data.tobytes())

    process = await asyncio.create_subprocess_exec(
        "ffmpeg",
        "-f",
        "s16le",  # 输入数据格式:16位小端PCM
        "-ar",
        str(sampling_rate),  # 采样率
        "-ac",
        "1",  # 单声道
        "-i",
        "pipe:0",  # 从标准输入读取数据
        "-f",
        "mp3",  # 输出格式
        "-b:a",
        "192k",  # 比特率
        "pipe:1",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        bufsize=4096,
    )

    async def write_stdin():
        while True:
            chunk = pcm_io.read(4096)
            if not chunk:
                break
            process.stdin.write(chunk)
            await process.stdin.drain()
        process.stdin.close()

    async def read_stdout():
        while True:
            out_chunk = await process.stdout.read(4096)
            if not out_chunk:
                break
            yield out_chunk

    async def stream_generator():
        # 启动写入任务和读取任务并行化执行
        write_task = asyncio.create_task(write_stdin())
        # 读取数据则直接在生成器中 yield 输出
        async for chunk in read_stdout():
            yield chunk
        await write_task  # 确保写入任务完成

    return StreamingResponse(stream_generator(), media_type="audio/mpeg")


def is_valid_audio_format(sample_rate: int, bit_depth: int) -> bool:
    valid_sample_rates = [8000, 16000, 22050, 24000, 32000, 44100, 48000, 96000]
    valid_bit_depths = [8, 16, 24, 32]
    return sample_rate in valid_sample_rates and bit_depth in valid_bit_depths


def parse_gradio_audio(audio_tuple):
    sample_rate, audio = audio_tuple

    # 校验维度
    if audio.ndim == 1:
        channels = 1
    elif audio.ndim == 2:
        channels = audio.shape[1]
    else:
        raise ValueError("Unsupported audio array shape.")

    # 位深推断(这里假定从 dtype 映射)
    bit_depth_map = {
        np.int16: 16,
        np.int32: 32,
        np.uint8: 8,
        np.float32: 32,
        np.float64: 64,
    }
    bit_depth = bit_depth_map.get(audio.dtype.type)
    if bit_depth is None:
        raise ValueError("Unsupported audio dtype.")

    return sample_rate, bit_depth, channels, audio


def maybe_pcm24_in_int32(arr: np.ndarray) -> bool:
    if arr.dtype != np.int32:
        return False
    # 判断高位是否为 0(正值)或 -1(负值扩展)
    upper_bytes = arr >> 24  # 提取最高 8 位(符号位仍保留)
    unique_vals = np.unique(upper_bytes)
    return np.all((unique_vals == 0) | (unique_vals == -1))



def validate_audio_format(sample_rate, wav_data):
    # 合法采样率与位深
    valid_sample_rates = {8000, 16000, 22050, 24000, 32000, 44100, 48000, 96000}
    valid_dtypes = {np.int16, np.int32, np.float32}
    bit_depth_map = {
        np.uint8: 8,
        np.int16: 16,
        np.int32: 32,  # ⚠️ 可能是封装的 24-bit #应该不用处理32-bit,几乎没有播放器支持
        np.float32: 32,
    }

    if not isinstance(wav_data, np.ndarray):
        raise TypeError("音频数据必须是 np.ndarray")

    if sample_rate not in valid_sample_rates:
        raise ValueError(f"不支持的采样率: {sample_rate} Hz")

    if wav_data.dtype.type not in bit_depth_map:
        raise ValueError(f"不支持的数据类型: {wav_data.dtype}")

    channels = (
        1 if wav_data.ndim == 1 else wav_data.shape[1] if wav_data.ndim == 2 else None
    )
    if channels is None:
        raise ValueError(f"不支持的数据维度: {wav_data.ndim}D,必须是1D或2D")

    # 特殊判断:int32 中是否嵌套 24-bit PCM
    suspected_pcm24 = False
    if wav_data.dtype == np.int32:
        high_byte = wav_data >> 24
        high_unique = np.unique(high_byte)
        if np.all((high_unique == 0) | (high_unique == -1)):
            suspected_pcm24 = True

    # 范围检查(仅对 float / int)
    if np.isnan(wav_data).any() or np.isinf(wav_data).any():
        raise ValueError("音频数据中存在 NaN 或 Inf")

    if wav_data.dtype == np.float32:
        if wav_data.max() > 1.0 or wav_data.min() < -1.0:
            raise ValueError("float32 音频数据超出 [-1.0, 1.0] 范围")

    # 结果返回
    return {
        "sample_rate": sample_rate,
        "dtype": str(wav_data.dtype),
        "bit_depth": 24 if suspected_pcm24 else bit_depth_map[wav_data.dtype.type],
        "channels": channels,
        "shape": wav_data.shape,
        "is_pcm24_embedded_in_int32": suspected_pcm24,
    }


def get_audio_duration(wav_data: np.ndarray, sample_rate: int) -> float:
    """
    根据采样率和 NumPy 音频数据计算时长(单位:秒)

    参数:
        wav_data: np.ndarray,支持 1D(单声道)或 2D(samples, channels)
        sample_rate: int,采样率,单位 Hz

    返回:
        float:音频时长,单位为秒
    """
    if not isinstance(wav_data, np.ndarray):
        raise TypeError("输入 wav_data 必须是 np.ndarray 类型")

    if not isinstance(sample_rate, (int, float)) or sample_rate <= 0:
        raise ValueError("采样率必须为正数")

    if wav_data.ndim not in (1, 2):
        raise ValueError(f"不支持的音频维度: {wav_data.ndim}D,必须是 1D 或 2D")

    num_frames = wav_data.shape[0]
    duration = num_frames / sample_rate
    return round(duration, 6)  # 保留微秒精度


import subprocess
import tempfile


def convert_wav_to_mp3(
    input_bytesio: io.BytesIO, use_qsv: bool = False, fallback_cmd: bool = True
) -> io.BytesIO:
    """
    将 WAV 格式的 BytesIO 输入转换为 MP3 格式 BytesIO 输出.
    优先使用 ffmpeg-python,失败时 fallback 到命令行调用.

    参数:
        input_bytesio: io.BytesIO,WAV 音频流
        use_qsv: 是否启用 Intel QSV 加速(目前音频编码无效,预留接口)
        fallback_cmd: 是否允许在 Python 接口失败时使用命令行 fallback

    返回:
        io.BytesIO:输出 MP3 音频流
    """
    input_bytesio.seek(0)
    output_bytesio = io.BytesIO()

    # 尝试使用 ffmpeg-python
    try:
        try:
            import ffmpeg
        except ImportError:
            if not fallback_cmd:
                raise ImportError("未安装 ffmpeg-python,且未启用 fallback 命令调用.")
            raise

        stream = ffmpeg.input("pipe:0")
        stream = ffmpeg.output(
            stream, "pipe:1", format="mp3", acodec="libmp3lame", audio_bitrate="192k"
        )
        out, _ = ffmpeg.run(
            stream, input=input_bytesio.read(), capture_stdout=True, capture_stderr=True
        )
        output_bytesio.write(out)
        output_bytesio.seek(0)
        return output_bytesio

    except Exception:
        if not fallback_cmd:
            raise RuntimeError("ffmpeg-python 转换失败,且未启用命令行 fallback 模式.")

    # fallback 到系统命令行调用 ffmpeg
    with (
        tempfile.NamedTemporaryFile(suffix=".wav") as tmp_in,
        tempfile.NamedTemporaryFile(suffix=".mp3") as tmp_out,
    ):
        tmp_in.write(input_bytesio.getvalue())
        tmp_in.flush()

        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            tmp_in.name,
            "-vn",
            "-acodec",
            "libmp3lame",
            "-b:a",
            "192k",
        ]

        if use_qsv:
            # QSV 加速暂不支持音频编码,占位保留
            pass

        cmd.append(tmp_out.name)

        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode != 0:
            raise RuntimeError(
                f"命令行 ffmpeg 转换失败: {result.stderr.decode(errors='ignore')}"
            )

        output_bytesio.write(tmp_out.read())
        output_bytesio.seek(0)
        return output_bytesio

def convert_wav_to_mp3(...):
    try:
        import ffmpeg
        # 使用 ffmpeg-python 进行转换...
    except ImportError:
        if not fallback_cmd:
            raise ImportError("未安装 ffmpeg-python,且未启用 fallback")
        # fallback 到命令行 ffmpeg
    except Exception as e:
        if not fallback_cmd:
            raise RuntimeError(f"ffmpeg-python 转换失败: {e}")
        # fallback 到命令行 ffmpeg


import asyncio
from collections import deque

class LightweightQueue:
    def __init__(self):
        self.buffer = deque()
        self.cond = asyncio.Condition()

    def put(self, item):  # 可被线程调用,用 call_soon_threadsafe 包裹
        self.buffer.append(item)
        asyncio.get_event_loop().call_soon_threadsafe(self._notify)

    def _notify(self):
        # 唤醒等待的消费者
        asyncio.create_task(self._wake())

    async def _wake(self):
        async with self.cond:
            self.cond.notify()

    async def get(self):
        async with self.cond:
            while not self.buffer:
                await self.cond.wait()
            return self.buffer.popleft()


import asyncio
import time
from sortedcontainers import SortedList
from dataclasses import dataclass, field

@dataclass(order=True)
class Item:
    expire_time: float
    key: str = field(compare=False)
    value: any = field(compare=False)

class AsyncSortedTTLMap:
    def __init__(self, cleanup_interval=5):
        self._dict = {}
        self._sorted = SortedList()
        self._lock = asyncio.Lock()
        self._cleanup_interval = cleanup_interval
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())

    async def set(self, key: str, value: any, ttl: float):
        expire_time = time.time() + ttl
        async with self._lock:
            if key in self._dict:
                old = self._dict[key]
                self._sorted.discard(old)
            item = Item(expire_time, key, value)
            self._dict[key] = item
            self._sorted.add(item)

    async def get(self, key: str):
        async with self._lock:
            item = self._dict.get(key)
            if item and item.expire_time > time.time():
                return item.value
            elif item:
                await self._remove(key)
        return None

    async def pop_earliest(self):
        async with self._lock:
            while self._sorted:
                item = self._sorted[0]
                if item.expire_time > time.time():
                    self._sorted.pop(0)
                    del self._dict[item.key]
                    return item.key, item.value
                else:
                    await self._remove(item.key)
        return None

    async def _remove(self, key):
        item = self._dict.pop(key, None)
        if item:
            self._sorted.discard(item)

    async def _cleanup_loop(self):
        while True:
            await asyncio.sleep(self._cleanup_interval)
            async with self._lock:
                now = time.time()
                while self._sorted and self._sorted[0].expire_time <= now:
                    expired = self._sorted.pop(0)
                    self._dict.pop(expired.key, None)



import sys, dill

# 接收端
while True:
    obj = dill.load(sys.stdin.buffer)
    handle(obj)

# 发送端
dill.dump({"wav_data": b"...", "sampling_rate": 24000}, sys.stdout.buffer)
sys.stdout.flush()

import sys
import dill

# 保存原始 stdout(用于发送数据)
data_out = sys.stdout.buffer

# 重定向 print/log 到 stderr,避免干扰 stdout
sys.stdout = sys.stderr

# 子进程内发送数据时:使用 data_out 明确输出
dill.dump({"sampling_rate": 24000, "wav_data": b"..."}, data_out)
data_out.flush()
