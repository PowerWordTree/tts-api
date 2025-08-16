import argparse
import asyncio
import io
import wave
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Query
from fastapi.concurrency import asynccontextmanager
from fastapi.responses import StreamingResponse

from pwt.tts_api.model_runner import ModelRunner
from pwt.utils import ndarray_audio_utils
from pwt.utils.tag_pool import TagData, TagPool


async def init_model() -> TagData[ModelRunner]:
    runner = await asyncio.to_thread(ModelRunner)
    return TagData(runner)


async def close_model(tag_data: TagData[ModelRunner]) -> None:
    if tag_data.data is None:
        return
    await asyncio.to_thread(tag_data.data.close)


@asynccontextmanager
async def lifespan(app: FastAPI):
    tag_pool = await TagPool.create(
        max_size=app.state.args.jobs,
        register_factory=init_model,
        destroy_factory=close_model,
    )
    app.state.model_pool = tag_pool
    yield  # 应用运行中
    await tag_pool.close()


# # FastAPI
app = FastAPI(lifespan=lifespan)


@app.get("/tts")
async def tts_endpoint(audio_prompt: str = Query(...), text: str = Query(...)):
    # 获取完整路径
    audio_prompt = str(Path(audio_prompt).resolve())
    # 模型推理
    async with app.state.model_pool.lease(audio_prompt) as tag_data:
        tag_data.tag = audio_prompt
        sample_rate, wav_data = await asyncio.to_thread(
            tag_data.data.infer, audio_prompt, text
        )
    # 获取音频信息
    info = ndarray_audio_utils.probe_audio_metadata(sample_rate, wav_data)
    print("音频信息:", info)

    # 写入内存 WAV 流
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wf:
        wf.setnchannels(info["num_channels"])
        wf.setsampwidth(info["bytes_per_sample"])  # 通常 int16 -> 2 字节
        wf.setframerate(info["sample_rate"])
        wf.writeframes(wav_data.tobytes())

    buffer.seek(0)
    return StreamingResponse(buffer, media_type="audio/wav")


def parse_args():
    parser = argparse.ArgumentParser(description="IndexTTS API")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="监听地址")
    parser.add_argument("--port", type=int, default=7860, help="监听端口")
    parser.add_argument("--jobs", type=int, default=1, help="模型并发数")
    return parser.parse_args()


def main():
    # 解析命令行
    args = parse_args()
    app.state.args = args
    # 启动API服务
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
