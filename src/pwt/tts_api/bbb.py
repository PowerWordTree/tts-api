import asyncio
import io
import threading
import queue
import numpy as np
import av
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
import soundfile as sf

app = FastAPI()

class StreamingBuffer:
    """
    一个文件-like 对象,用于记录写入的数据块,
    内部通过线程安全的 Queue 保存数据,提供生成器接口给 StreamingResponse.
    """
    def __init__(self):
        self._queue = queue.Queue()
        self.closed = False

    def write(self, data: bytes):
        self._queue.put(data)

    def flush(self):
        # 本例中无需额外处理 flush
        pass

    def close(self):
        self.closed = True
        # 用 None 标记已完成写入
        self._queue.put(None)

    def read_generator(self):
        """
        通过生成器持续 yield 队列中的数据块,实现边编码边输出
        """
        while True:
            chunk = self._queue.get()
            if chunk is None:
                break
            yield chunk

async def generate_encoded_stream_async(
    sampling_rate: int,
    wav_data: np.ndarray,
    output_format: str = "mp3"
):
    """
    异步封装的生成器:
    将传入的采样率与 wav_data(numpy 数组)转码为指定格式(mp3 或 ogg)的音频,
    利用 run_in_executor 调用阻塞的编码函数,
    并返回一个生成器,该生成器实时产出编码数据.
    """
    # 根据 output_format 确定容器/编码器和 MIME 类型
    if output_format == "mp3":
        container_format = "mp3"
        codec_name = "libmp3lame"
        mime_type = "audio/mpeg"
    elif output_format == "ogg":
        container_format = "ogg"
        codec_name = "vorbis"
        mime_type = "audio/ogg"
    else:
        raise ValueError(f"Unsupported output format: {output_format}")

    # 处理音频数据
    # 如果 wav_data 为一维,则扩展为二维
    if wav_data.ndim == 1:
        wav_data = np.expand_dims(wav_data, axis=1)
    if wav_data.ndim != 2:
        raise ValueError("Unsupported wav_data shape; 应为 (samples, channels)")
    channels = wav_data.shape[1]
    # PyAV 要求 shape 为 (channels, samples)
    wav_data = wav_data.T
    # 确保数据类型为 int16(如果不是,可自动转换)
    if wav_data.dtype != np.int16:
        wav_data = wav_data.astype(np.int16)
    layout = "mono" if channels == 1 else "stereo"

    # 创建自定义流式缓冲区,作为输出目标
    stream_buffer = StreamingBuffer()

    def blocking_encoding():
        """
        阻塞的编码逻辑,运行在 executor 线程池中.
        使用 PyAV 将 numpy 数组转成 AudioFrame 后编码,并写入 stream_buffer.
        """
        try:
            # 以写模式打开容器,将输出定向到我们的自定义 stream_buffer
            output_container = av.open(stream_buffer, mode="w", format=container_format)
            stream = output_container.add_stream(codec_name, rate=sampling_rate)
            stream.channels = channels

            # 从 numpy 数据构造音频帧
            frame = av.AudioFrame.from_ndarray(wav_data, format="s16", layout=layout)
            frame.rate = sampling_rate

            # 编码帧并实时 mux 生成 packets
            for packet in stream.encode(frame):
                output_container.mux(packet)
            # 冲刷编码器,将延迟数据写出
            for packet in stream.encode():
                output_container.mux(packet)

            output_container.close()  # 完整写入 trailer 后会调用 stream_buffer.close()
        except Exception as e:
            print("编码过程中发生异常:", e)
            stream_buffer.close()

    # 调用 run_in_executor 将阻塞编码任务放入线程池执行
    loop = asyncio.get_running_loop()
    loop.run_in_executor(None, blocking_encoding)

    # 返回生成器对象和 MIME 类型
    return stream_buffer.read_generator(), mime_type


@app.post("/stream_encode")
async def stream_encode(
    sampling_rate: int,
    file: UploadFile = File(...),
    output_format: str = "mp3"
):
    """
    FastAPI 路由:
    接收上传的音频文件(例如 WAV),使用 soundfile 读取数据,
    然后调用异步生成器函数实现边编码边流式返回.
    """
    # 使用 soundfile 读取上传的文件数据,得到 numpy 数组及采样率
    data, sr = sf.read(file.file)
    if sr != sampling_rate:
        # 如有需要可添加重采样,下面仅输出警告
        print(f"警告:上传文件采样率为 {sr},与目标采样率 {sampling_rate} 不一致.")

    generator, mime = await generate_encoded_stream_async(sampling_rate, data, output_format)
    # StreamingResponse 会逐块拉取生成器的数据,实现边传输边编码的效果
    return StreamingResponse(generator, media_type=mime)


import queue

class StreamingBuffer(queue.Queue):
    """
    继承自 queue.Queue,实现文件-like 接口,并重载 __iter__,
    使得该对象本身就是一个生成器,可以直接用于 StreamingResponse.
    """
    def write(self, data: bytes):
        """写入数据到队列"""
        self.put(data)

    def flush(self):
        """本例中无需额外处理 flush"""
        pass

    def close(self):
        """结束写入时,将结束标识 None 放入队列"""
        self.put(None)

    def __iter__(self):
        """
        实现迭代器协议:直接对队列进行迭代,
        当遇到 None 时退出迭代.
        """
        while True:
            item = self.get()
            if item is None:
                break
            yield item

# 使用示例:
if __name__ == "__main__":
    sb = StreamingBuffer()
    # 模拟写入过程
    sb.write(b"数据块1\n")
    sb.write(b"数据块2\n")
    sb.close()

    # 直接迭代 sb 对象
    for chunk in sb:
        print(chunk)
