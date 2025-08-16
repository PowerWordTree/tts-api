import argparse
import asyncio
import contextlib
import os
import re
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any, AsyncGenerator, Literal

import numpy as np
import uvicorn
from fastapi.responses import StreamingResponse
from pydantic import Field, ValidationError
from starlette.applications import Starlette
from starlette.datastructures import State
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from pwt.tts_api import audio_encode
from pwt.utils.pydantic_utils import (
    BaseModelEx,
    check,
    convert,
    format_validation_error,
)

FORM_ENCODE = "application/x-www-form-urlencoded"
FORM_MULTI = "multipart/form-data"
APP_JSON = "application/json"


@asynccontextmanager
async def lifespan(app: Starlette) -> AsyncGenerator[None, Any]:
    import indextts
    from indextts.infer import IndexTTS

    # 运行前
    print(">>", "开始载入IndexTTS模型")
    base_path = Path(indextts.__file__).resolve().parent.parent
    model_path = base_path / "checkpoints"
    model_config = model_path / "config.yaml"
    app.state.tts = IndexTTS(model_dir=str(model_path), cfg_path=str(model_config))
    if app.state.concurrency != 1:
        raise ValueError("不支持并发数量大于1")
    app.state.semaphore = asyncio.BoundedSemaphore(app.state.concurrency)
    yield  # 应用运行中
    # 运行后
    pass


async def parse_params(request: Request) -> dict[str, Any]:
    result = {}

    await request.body()
    with contextlib.suppress(Exception):
        query_params = request.query_params
        for name in query_params.keys():
            param = query_params.getlist(name)
            result[name] = param[0] if len(param) == 1 else param

        content_type = request.headers.get("content-type", "").lower()
        if FORM_ENCODE in content_type or FORM_MULTI in content_type:
            form_params = await request.form()
            for name in form_params.keys():
                param = form_params.getlist(name)
                result[name] = param[0] if len(param) == 1 else param
        elif APP_JSON in content_type:
            json_params = await request.json()
            if isinstance(json_params, dict):
                result.update(json_params)

    return result


class TtsParams(BaseModelEx, extra="ignore"):
    audio_prompt: Annotated[
        str,
        convert(str.strip),
        Field(min_length=1),
        convert(lambda x: str(Path(x).resolve())),
        check(os.path.exists, check_result=True, description="File not found"),
    ]
    text: Annotated[
        str,
        convert(str.strip),
        Field(min_length=1),
    ]
    output_format: Annotated[
        Literal["wav", "mp3"], convert(lambda x: x.strip().lower())
    ] = "mp3"
    queue_timeout: Annotated[
        float | None,
        Field(ge=0),
        convert(lambda x: 0.005 if x < 0.005 else x),
    ] = None
    execution_timeout: Annotated[
        float | None,
        Field(ge=0),
        convert(lambda x: 0.005 if x < 0.005 else x),
    ] = None


async def tts_infer(state: State, tts_params: TtsParams) -> tuple[int, np.ndarray]:
    """
    Raises:
        asyncio.TimeoutError: 服务繁忙
        RuntimeError: 推理失败
    """
    # 并发控制
    try:
        await asyncio.wait_for(
            state.semaphore.acquire(), timeout=tts_params.queue_timeout
        )
    except asyncio.TimeoutError:
        raise asyncio.TimeoutError("服务繁忙, 请稍后再试")
    # 模型推理
    try:
        # 开始推理
        sampling_rate, wav_data = await asyncio.wait_for(
            asyncio.to_thread(
                state.tts.infer,
                tts_params.audio_prompt,
                tts_params.text,
                output_path=None,
                verbose=False,
            ),
            tts_params.execution_timeout,
        )
    except Exception as exc:
        raise RuntimeError(*exc.args) from exc
    finally:
        state.semaphore.release()
    # 返回结果
    return sampling_rate, wav_data


async def tts(request: Request) -> Response:
    # 解析参数
    params = await parse_params(request)
    try:
        tts_params = TtsParams.model_validate(params)
    except ValidationError as exc:
        details = format_validation_error(exc)
        return JSONResponse(
            {"code": 400, "message": "参数错误", "details": details}, 400
        )
    # 模型推理
    try:
        sampling_rate, wav_data = await tts_infer(request.app.state, tts_params)
    except asyncio.TimeoutError as exc:
        return JSONResponse({"code": 429, "message": str(exc)}, 429)
    except RuntimeError as exc:
        return JSONResponse({"code": 500, "message": str(exc)}, 500)
    # 返回音频流
    if tts_params.output_format == "mp3":
        buffer = await audio_encode.mp3_encoder(wav_data, sampling_rate)
        media_type = "audio/mpeg"
    else:
        buffer = await audio_encode.wav_encoder(wav_data, sampling_rate)
        media_type = "audio/wav"
    return StreamingResponse(buffer, media_type=media_type)


def parse_args():
    parser = argparse.ArgumentParser(description="IndexTTS API")
    parser.add_argument(
        "--bind",
        "-b",
        type=str,
        default="0.0.0.0:7860",
        help="监听地址与端口, 支持Unix套接字. 默认: 0.0.0.0:7860",
    )
    parser.add_argument(
        "--concurrency",
        "-c",
        type=int,
        default=1,
        help="最大并发数, 默认: 1",
    )
    return parser.parse_args()


def parse_bind(bind: str) -> dict[str, str]:
    match = (
        re.match(r"(?i)unix:(?P<unix>.+)", bind)  # unix:/path/file.sock
        or re.match(r"\[(?P<host>.+)\]:(?P<port>\d+)", bind)  # [::]:8000
        or re.match(r"\[(?P<host>.+)\]", bind)  # [::]
        or re.match(r"(?P<host>.+):(?P<port>\d+)", bind)  # 0.0.0.0:8000
        or re.match(r":(?P<port>\d+)", bind)  # :8000
        or re.match(r"(?P<host>.+)", bind)  # 0.0.0.0/localhost
    )
    if not match:
        raise argparse.ArgumentTypeError(f"Invalid bind address: {bind}")

    result = match.groupdict()
    if "unix" not in result:
        if "host" not in result:
            result["host"] = "0.0.0.0"
        if "port" not in result:
            result["port"] = "7860"

    return result


def main() -> None:
    args = parse_args()
    print(">>", "监听地址:", args.bind, "最大并发数:", args.concurrency)
    app = Starlette(
        lifespan=lifespan,
        routes=[Route("/tts", tts, methods=["GET", "POST"])],
    )
    app.state.concurrency = args.concurrency
    bind = parse_bind(args.bind)
    if "unix" in bind:
        uvicorn.run(app, uds=bind["unix"])
    else:
        uvicorn.run(app, host=bind["host"], port=int(bind["port"]))


if __name__ == "__main__":
    main()
