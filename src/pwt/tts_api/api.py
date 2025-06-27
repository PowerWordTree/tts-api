import argparse
import asyncio
import io
import wave

import uvicorn
from fastapi import FastAPI, Query
from fastapi.concurrency import asynccontextmanager
from fastapi.responses import StreamingResponse

from pwt.tts_api import audio_utils
from pwt.tts_api.model_runner import ModelRunner
from pwt.tts_api.tag_resource_pool import TagResource, TagResourcePool


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model_pool = TagResourcePool()
    jobs = app.state.args.jobs
    resouces = [TagResource(None, ModelRunner()) for i in range(0, jobs, 1)]
    await app.state.model_pool.register_Resources(resouces)
    yield  # 应用运行中
    for resouce in resouces:
        resouce.model.close()


# # FastAPI
app = FastAPI(lifespan=lifespan)


@app.get("/tts")
async def tts_endpoint(audio_prompt: str = Query(...), text: str = Query(...)):
    # 模型推理
    async with app.state.model_pool.acquire_context(audio_prompt) as resource:
        sampling_rate, wav_data = await asyncio.to_thread(
            resource.model.infer, audio_prompt, text
        )
    # 获取音频信息
    info = audio_utils.get_audio_info_from_gradio(sampling_rate, wav_data)
    print("音频信息:", info)

    # 写入内存 WAV 流
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wf:
        wf.setnchannels(info["channels"])
        wf.setsampwidth(info["sample_width"])  # 通常 int16 -> 2 字节
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
