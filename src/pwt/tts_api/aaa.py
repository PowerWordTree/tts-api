import io
import numpy as np
import av
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse

app = FastAPI()


def encode_audio_to_streaming_response(
    sampling_rate: int,
    wav_data: np.ndarray,
    output_format: str = "mp3"
) -> StreamingResponse:
    """
    将传入的采样率和 wav_data (numpy 数组) 编码成 MP3 或 OGG,
    并返回 FastAPI 的 StreamingResponse.
    """
    # 根据 output_format 判定容器格式/编码器和 MIME 类型
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

    # 使用 BytesIO 作为内存缓冲区,构建输出容器
    output_buffer = io.BytesIO()
    output_container = av.open(output_buffer, mode="w", format=container_format)

    # 判断 wav_data 的形状,如果是一维则视为单声道,
    # 如果是二维则假设为 (samples, channels)
    if wav_data.ndim == 1:
        channels = 1
        layout = "mono"
        # 转换为二维 (samples, 1) 方便后续处理
        wav_data = np.expand_dims(wav_data, axis=1)
    elif wav_data.ndim == 2:
        channels = wav_data.shape[1]
        if channels == 1:
            layout = "mono"
        elif channels == 2:
            layout = "stereo"
        else:
            raise ValueError("仅支持单声道或双声道音频")
    else:
        raise ValueError(f"不支持的 wav_data 维度: {wav_data.ndim}")

    # PyAV 要求数据格式为 (channels, samples),因此对二维数据做转置
    wav_data = wav_data.T  # 新形状: (channels, samples)

    # 确保数据类型为 int16,即对应 's16' 格式
    if wav_data.dtype != np.int16:
        wav_data = wav_data.astype(np.int16)

    # 添加音频流,设置采样率和通道数
    stream = output_container.add_stream(codec_name, rate=sampling_rate)
    stream.channels = channels

    # 创建 AudioFrame,指定数据格式与通道布局
    frame = av.AudioFrame.from_ndarray(wav_data, format="s16", layout=layout)
    frame.rate = sampling_rate

    # 编码 AudioFrame,并将生成的 packet 写入容器
    for packet in stream.encode(frame):
        output_container.mux(packet)
    # 再次调用 encode(None) 以冲刷编码器中的缓冲数据
    for packet in stream.encode():
        output_container.mux(packet)

    output_container.close()  # 此时 BytesIO 中已包含完整生成的数据
    output_buffer.seek(0)
    return StreamingResponse(output_buffer, media_type=mime_type)


# 一个 FastAPI 接口示例,假设上传文件是 WAV 格式(与 Gradio 输出类似)
@app.post("/upload_and_encode")
async def upload_and_encode(
    sampling_rate: int,
    file: UploadFile = File(...),
    output_format: str = "mp3"
):
    """
    上传一个 WAV 文件,并按指定采样率及格式进行编码,返回流形式的音频数据.
    使用 soundfile 库读取文件数据,你也可以根据业务场景调整获取 wav_data 的方式.
    """
    import soundfile as sf

    # 使用 soundfile 读取上传的文件
    data, sr = sf.read(file.file)
    # 若上传文件实际采样率与指定的 sampling_rate 不一致,
    # 可在此处加入重采样步骤(例如使用 librosa 或 scipy.signal.resample)
    if sr != sampling_rate:
        # 此处仅提醒,可添加重采样代码
        print(f"Warning: 文件的采样率为 {sr},与指定值 {sampling_rate} 不同.")

    # 调用上面封装的编码函数,返回 StreamingResponse
    return encode_audio_to_streaming_response(sampling_rate, data, output_format)
