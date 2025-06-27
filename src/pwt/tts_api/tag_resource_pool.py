import asyncio
from dataclasses import dataclass
from typing import Iterable

from pwt.tts_api.model_runner import ModelRunner


@dataclass(frozen=True, eq=True)
class TagResource:
    tag: str | None
    model: ModelRunner


class TagResourcePool:
    def __init__(self) -> None:
        # 可分配的空闲资源列表(活跃度从低到高排序, 末尾最活跃)
        self._idle_Resources = []
        # 当前占用中的资源集合(快速查重)
        self._active_Resources = set()
        self._condition = asyncio.Condition()

    async def register_Resources(self, Resources: Iterable[TagResource]) -> None:
        """注册新的资源到池

        新注册的资源会添加到空闲列表末尾, 视为最活跃资源

        Args:
            Resources: 待注册的资源迭代器
        """
        async with self._condition:
            self._idle_Resources.extend(Resources)
            self._condition.notify_all()

    async def acquire(self, tag: str | None = None) -> TagResource:
        """在池中获取一个资源

        优先从最活跃(列表末尾)到最不活跃(列表头部)查找匹配tag的资源
        若未找到tag匹配, 则返回最不活跃的资源

        Args:
            tag: 可选的标签过滤条件

        Returns:
            获取到的资源
        """
        async with self._condition:
            while not self._idle_Resources:
                await self._condition.wait()
            # 从最活跃(末尾)到最不活跃(头部)遍历查找匹配tag的资源
            for i in range(len(self._idle_Resources) - 1, -1, -1):
                if self._idle_Resources[i].tag == tag:
                    Resource = self._idle_Resources.pop(i)
                    break
            else:
                # 未找到匹配时,获取最不活跃的资源
                Resource = self._idle_Resources.pop(0)
            self._active_Resources.add(Resource)
            return Resource

    async def release(self, Resource: TagResource) -> None:
        """释放已占用的资源

        释放的资源会从占用中的集合移除,并重新添加到空闲列表末尾(最活跃状态)

        Args:
            Resource: 待释放的资源
        """
        async with self._condition:
            if Resource in self._active_Resources:
                self._active_Resources.remove(Resource)
                self._idle_Resources.append(Resource)
                self._condition.notify_all()

    def acquire_context(self, tag: str | None = None) -> "TagResourceContext":
        return TagResourceContext(self, tag)

class TagResourceContext:
    def __init__(self, pool: TagResourcePool, tag: str | None = None):
        self._pool = pool
        self._tag = tag
        self.resource: TagResource | None = None

    async def __aenter__(self) -> TagResource:
        self.resource = await self._pool.acquire(self._tag)
        return self.resource

    async def __aexit__(self, exc_type, exc, tb):
        if self.resource is not None:
            await self._pool.release(self.resource)
