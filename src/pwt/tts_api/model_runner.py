import contextlib
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import dill
import indextts
import numpy as np
from indextts.infer import IndexTTS


class ModelRunner:
    def __init__(self) -> None:
        self.process = subprocess.Popen(
            [sys.executable, os.path.abspath(__file__)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=sys.stdout,
            text=False,
        )
        if self.process.stdin is None or self.process.stdout is None:
            raise ValueError("进程 stdin 或 stdout 未正确初始化")
        self._stdin = self.process.stdin
        self._stdout = self.process.stdout

    # @override
    def infer(
        self, audio_prompt: str, text: str, verbose: bool = False
    ) -> tuple[int, np.ndarray]:
        send_data: dict[str, Any] = {
            "audio_prompt": audio_prompt,
            "text": text,
            "verbose": verbose,
        }
        dill.dump(send_data, self._stdin)
        self._stdin.flush()
        recv_data: tuple[int, np.ndarray] | None = dill.load(self._stdout)
        if recv_data is None:
            raise ValueError("模型执行失败")  # TODO: 需封装返回,以适应异常情况
        return recv_data

    def close(self) -> None:
        try:
            with contextlib.suppress(Exception):
                dill.dump(None, self._stdin)
                self._stdin.flush()
            with contextlib.suppress(Exception):
                self._stdin.close()
            with contextlib.suppress(Exception):
                self._stdout.close()
            self.process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            self.process.kill()
            with contextlib.suppress(Exception):
                self.process.wait(timeout=5)


class ModelRunnerWorker:
    def __init__(self) -> None:
        self.stdin = sys.stdin.buffer
        self.stdout = sys.stdout.buffer
        sys.stdout = sys.stderr
        print(">>", "开始载入IndexTTS模型")
        self._base_path = Path(os.path.dirname(indextts.__file__)).parent
        self._model_path = os.path.join(self._base_path, "checkpoints")
        self._model_config = os.path.join(self._model_path, "config.yaml")
        self._tts = IndexTTS(model_dir=self._model_path, cfg_path=self._model_config)

    def start(self) -> None:
        while True:
            recv_data: dict[str, Any] = dill.load(self.stdin)
            if recv_data is None:
                break
            try:
                send_data: tuple[int, np.ndarray] | None = self._tts.infer(
                    **recv_data, output_path=None
                )
            except Exception:
                send_data = None
            dill.dump(send_data, self.stdout)
            self.stdout.flush()


if __name__ == "__main__":
    worker = ModelRunnerWorker()
    worker.start()
