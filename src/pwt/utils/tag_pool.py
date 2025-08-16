"""
tag_pool.py 模块

提供 TagPool 类, 用于管理带标签的异步资源池,
支持并发获取/释放/注册/销毁等操作, 并具备自动回收空闲资源的机制.

核心组件:
- TagData: 资源包装类, 包含 data/tag/timestamp 和 stale 标记, 用于描述池中单个资源状态.
- TagPool: 资源池管理类, 负责资源的生命周期管理, 包括注册/获取/释放/销毁及关闭.
- 工厂方法: register_factory 用于动态创建资源; destroy_factory 用于清理回收资源.
- 空闲回收机制: 内置 `_idle_timer` 定时任务, 自动销毁超过 `max_idle_time` 的空闲资源.

使用示例:
    pool = await TagPool.create(max_size=5, min_size=1)
    async with pool.lease() as td:
        # 使用资源
        _ = td.data
        td.tag = "new_tag"
    await pool.close()
"""

from __future__ import annotations

import asyncio
import contextlib
import os
from collections import OrderedDict
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from itertools import islice
from typing import (
    Any,
    AsyncContextManager,
    AsyncGenerator,
    Callable,
    Concatenate,
    Coroutine,
    Generic,
    Iterable,
    ParamSpec,
    TypeVar,
)

from pwt.utils.mapped_list import MappedLinkedList, QueryField

T = TypeVar("T")
P = ParamSpec("P")

_RegisterFactory = Callable[[], Coroutine[Any, Any, "TagData"]]
_DestroyFactory = Callable[["TagData"], Coroutine[Any, Any, None]]
_Task = Callable[Concatenate["TagData", P], Coroutine[Any, Any, T]]


class TagPoolError(Exception):
    """TagPool 基础异常"""


class PoolClosedError(TagPoolError):
    """在已关闭的 TagPool 上执行操作时抛出"""


class ResourceUnavailableError(TagPoolError):
    """资源不可用时抛出"""


class FactoryInvocationError(TagPoolError):
    """工厂方法调用失败时抛出"""


@dataclass
class TagData(Generic[T]):
    """
    表示池中单个资源的封装结构.

    Attributes:
        data: 实际资源对象.
        tag: 可选标签, 用于资源筛选.
        timestamp: 最近一次释放或创建的时间戳, 用于判断空闲过期.
        stale: 标记资源是否已被销毁, 避免重复清理.
    """

    data: T | None = None
    tag: str | None = None
    timestamp: float = field(default=0.0, init=False)
    stale: bool = field(default=False, init=False)

    __hash__ = object.__hash__


class TagPool:
    """
    异步资源池, 支持标签筛选/并发获取与自动回收.

    特性:
        - 最大并发限制(max_size)
        - 最小池容量自动填充(min_size)
        - 空闲资源超时销毁(max_idle_time)
        - 可插拔的注册与销毁工厂方法
    """

    _loop: asyncio.AbstractEventLoop
    _idle_pool: MappedLinkedList[TagData]
    _active_pool: OrderedDict[TagData, float]
    _quota: asyncio.Semaphore
    _lock: asyncio.Lock
    _register_factory: _RegisterFactory | None
    _destroy_factory: _DestroyFactory | None
    _max_size: int
    _min_size: int
    _max_idle_time: float
    _idle_timer_event: asyncio.Event
    _idle_timer_task: asyncio.Task[None]
    _closed: bool

    @staticmethod
    async def create(
        loop: asyncio.AbstractEventLoop | None = None,
        max_size: int = 100,
        min_size: int = 1,
        max_idle_time: float = 600,
        register_factory: _RegisterFactory | None = None,
        destroy_factory: _DestroyFactory | None = None,
    ) -> TagPool:
        """🔒
        创建并初始化 TagPool 实例.

        Args:
            loop: 指定事件循环, 默认使用当前运行循环.
            max_size: 最大资源数, 超出后获取将阻塞.
            min_size: 最小资源数, 初始化时自动填充.
            max_idle_time: 空闲资源最大存活时间(秒).
            register_factory: 创建资源的协程工厂.
            destroy_factory: 销毁资源的协程工厂.

        Returns:
            TagPool: 初始化完成的资源池实例.
        """
        self = object.__new__(TagPool)
        self._loop = loop or asyncio.get_running_loop()
        self._idle_pool = MappedLinkedList(TagData, ["tag"])
        self._active_pool = OrderedDict()
        self._quota = asyncio.Semaphore(max_size)
        self._lock = asyncio.Lock()

        self._register_factory = register_factory
        self._destroy_factory = destroy_factory
        self._max_size = max_size
        self._min_size = min_size
        self._max_idle_time = max_idle_time

        self._closed = False
        self._idle_timer_event = asyncio.Event()
        self._idle_timer_task = self._loop.create_task(self._idle_timer())

        if self._register_factory:
            await self._run_all(self.register() for _ in range(self._min_size))

        return self

    def __init__(self) -> None:
        """⚠️
        禁止直接实例化, 必须通过 `create` 方法创建.

        Raises:
            TypeError: 总是抛出, 阻止直接调用构造函数.
        """
        raise TypeError("Direct instantiation is not allowed.")

    def __call__(self, tag: str | None = None) -> AsyncContextManager[TagData]:
        """
        结合上下文管理器语法, 简化lease调用.

        Args:
            tag: 可选标签, 指定获取特定标签的资源.

        Returns:
            AsyncContextManager[TagData]: 可用于 `async with` 的资源上下文.
        """
        return self.lease(tag)

    async def register(self, tag_data: TagData | None = None) -> None:
        """🔒⚠️
        向池中注册一个新资源.

        如果未提供 `tag_data` 则调用注册工厂方法创建.

        Args:
            tag_data: 可选的预创建资源包装, 若为 None 则动态创建.

        Raises:
            PoolClosedError: 池已关闭时抛出.
            ResourceUnavailableError: 无法创建资源时抛出.
            FactoryInvocationError: 调用注册工厂失败时抛出.
        """
        if self._closed:
            raise PoolClosedError("TagPool is closed.")

        if tag_data is None:
            tag_data = await self._call_register_factory()

        if tag_data is None:
            raise ResourceUnavailableError("Failed to create resource.")

        async with self._lock:
            await self._add(tag_data)
            self._idle_timer_event.set()

    async def destroy(self, tag_data: TagData) -> None:
        """🔒⚠️
        从池中移除并销毁指定资源.

        Args:
            tag_data: 目标资源包装.

        Raises:
            PoolClosedError: 池已关闭时抛出.
            ResourceUnavailableError: 资源不存在时抛出.
            FactoryInvocationError: 调用销毁工厂失败时抛出.
        """
        if self._closed:
            raise PoolClosedError("TagPool is closed.")

        async with self._lock:
            await self._remove(tag_data)
            self._idle_timer_event.set()

        if not tag_data.stale:
            await self._call_destroy_factory(tag_data)

    async def acquire(self, tag: str | None = None) -> TagData:
        """🔒⚠️
        获取一个可用资源(可能阻塞).

        Args:
            tag: 可选标签, 优先匹配对应资源.

        Returns:
            TagData: 已分配的资源包装.

        Raises:
            PoolClosedError: 池已关闭.
            ResourceUnavailableError: 无可用资源.
            FactoryInvocationError: 注册工厂调用失败.
        """
        await self._quota.acquire()

        if self._closed:
            raise PoolClosedError("TagPool is closed.")

        if not await self.idle_length():
            await self.register()

        async with self._lock:
            tag_data = self._idle_pool.pop(QueryField("tag", tag))
            if tag_data is None:
                tag_data = self._idle_pool.pop(first=True)

            if tag_data is None:
                raise ResourceUnavailableError("No idle resources available.")

            self._active_pool[tag_data] = self._loop.time()
            self._idle_timer_event.set()

        return tag_data

    async def release(self, tag_data: TagData) -> None:
        """🔒⚠️
        释放已获取的资源, 返回空闲池.

        Args:
            tag_data: 之前由 `acquire` 获取的资源包装.

        Raises:
            PoolClosedError: 池已关闭时抛出.
            ResourceUnavailableError: 资源不在活动池或已过时时抛出.
        """
        if self._closed:
            raise PoolClosedError("TagPool is closed.")

        async with self._lock:
            if tag_data not in self._active_pool:
                raise ResourceUnavailableError("Resource is not in the active pool.")

            if tag_data.stale:
                raise ResourceUnavailableError("Resource is stale.")

            self._active_pool.pop(tag_data, None)
            tag_data.timestamp = self._loop.time()
            self._idle_pool.push(tag_data)
            self._idle_timer_event.set()

        self._quota.release()

    async def submit(
        self, task: _Task[P, T], tag: str | None, *args: P.args, **kwargs: P.kwargs
    ) -> T:
        """🔒⚠️
        在获取并释放资源的上下文中执行指定任务.

        Args:
            task: 接收 TagData 的协程任务函数.
            tag: 可选标签, 指定获取相应标签的资源.
            *args: 传递给任务的其他位置参数.
            **kwargs: 传递给任务的关键字参数.

        Returns:
            T: 任务返回值.

        Raises:
            PoolClosedError: 池已关闭时抛出.
            ResourceUnavailableError: 资源获取失败时抛出.
            Exception: 任务执行过程中抛出的任何异常会向上冒泡.
        """
        async with self.lease(tag) as tag_data:
            return await task(tag_data, *args, **kwargs)

    @asynccontextmanager
    async def lease(self, tag: str | None = None) -> AsyncGenerator[TagData, Any]:
        """🔒⚠️
        资源上下文管理器, 获取后自动释放.

        Args:
            tag: 可选标签, 指定获取相应标签的资源.

        Yields:
            TagData: 上下文中可用的资源包装.

        Raises:
            PoolClosedError: 池已关闭时抛出.
            ResourceUnavailableError: 资源获取失败时抛出.
        """
        tag_data = await self.acquire(tag)
        try:
            yield tag_data
        finally:
            await self.release(tag_data)

    async def close(self) -> None:
        """🔒
        关闭资源池并销毁所有空闲及活动资源.

        取消空闲定时器任务, 释放信号量, 并调用销毁工厂清理所有资源.
        """
        self._closed = True

        while self._quota.locked():
            self._quota.release()

        if not self._idle_timer_task.done():
            self._idle_timer_task.cancel()
            with contextlib.suppress(Exception, asyncio.CancelledError):
                await self._idle_timer_task

        items = self._idle_pool.pop_all()
        items.extend(self._active_pool.keys())
        self._active_pool.clear()
        await self._run_all(self._call_destroy_factory(item) for item in items)

    async def set_min_size(self, value: int) -> None:
        """🔒
        调整池的最小容量.

        Args:
            value: 新的最小资源数.
        """
        async with self._lock:
            self._min_size = value
            self._idle_timer_event.set()

    async def set_max_idle_time(self, value: float) -> None:
        """🔒
        调整空闲资源超时时长.

        Args:
            value: 新的空闲超时(秒).
        """
        async with self._lock:
            self._max_idle_time = value
            self._idle_timer_event.set()

    async def set_register_factory(self, value: _RegisterFactory | None) -> None:
        """🔒
        设置自定义注册工厂方法.

        Args:
            value: 新的注册协程工厂函数, 或 None 取消自定义行为.
        """
        async with self._lock:
            self._register_factory = value

    async def set_destroy_factory(self, value: _DestroyFactory | None) -> None:
        """🔒
        设置自定义销毁工厂方法.

        Args:
            value: 新的销毁协程工厂函数, 或 None 取消自定义行为.
        """
        async with self._lock:
            self._destroy_factory = value

    async def length(self) -> int:
        """🔒
        获取池中所有资源的总数(活动 + 空闲).

        Returns:
            int: 当前资源总数.
        """
        async with self._lock:
            return len(self._active_pool) + len(self._idle_pool)

    async def active_length(self) -> int:
        """🔒
        获取活动(已分配)资源的数量.

        Returns:
            int: 当前活动资源数.
        """
        async with self._lock:
            return len(self._active_pool)

    async def idle_length(self) -> int:
        """🔒
        获取空闲资源的数量.

        Returns:
            int: 当前空闲资源数.
        """
        async with self._lock:
            return len(self._idle_pool)

    async def is_active(self, tag_data: TagData) -> bool:
        """🔒
        检查指定资源是否在活动池中.

        Args:
            tag_data: 要检查的资源包装.

        Returns:
            bool: 在活动池中返回 True, 否则 False.
        """
        async with self._lock:
            return tag_data in self._active_pool

    async def is_idle(self, tag_data: TagData) -> bool:
        """🔒
        检查指定资源是否在空闲池中.

        Args:
            tag_data: 要检查的资源包装.

        Returns:
            bool: 在空闲池中返回 True, 否则 False.
        """
        async with self._lock:
            return tag_data in self._idle_pool

    async def _idle_timer(self) -> None:
        """🔒
        空闲资源回收定时器.

        等待空闲事件并定期检查最早释放的资源, 超时则调用销毁工厂回收.
        """
        while True:
            await self._idle_timer_event.wait()
            if self._closed:
                break

            async with self._lock:
                now = self._loop.time()
                if len(self._idle_pool) + len(self._active_pool) <= self._min_size:
                    self._idle_timer_event.clear()
                    continue
                tag_data = self._idle_pool.peek(first=True)
                if tag_data is None:
                    self._idle_timer_event.clear()
                    continue
                expire = tag_data.timestamp + self._max_idle_time
                if now < expire:
                    tag_data = None
                else:
                    await self._remove(tag_data)

            if tag_data is not None:
                with contextlib.suppress(Exception):
                    await self._call_destroy_factory(tag_data)

            delay = max(expire - now + 0.05, 1)
            await asyncio.sleep(delay)

    async def _call_register_factory(self) -> TagData | None:
        """⚠️
        调用注册工厂方法创建资源.

        Returns:
            TagData | None: 工厂返回的资源包装, 或 None(无工厂时).

        Raises:
            FactoryInvocationError: 工厂调用失败时抛出.
        """
        if not self._register_factory:
            return None

        try:
            return await self._register_factory()
        except Exception as exc:
            raise FactoryInvocationError(*exc.args) from exc

    async def _call_destroy_factory(self, tag_data: TagData) -> None:
        """⚠️
        调用销毁工厂方法清理资源.

        标记资源为过时(stale), 并调用自定义销毁逻辑.

        Args:
            tag_data: 要销毁的资源包装.

        Raises:
            FactoryInvocationError: 工厂调用失败时抛出.
        """
        tag_data.stale = True
        if not self._destroy_factory:
            return

        try:
            await self._destroy_factory(tag_data)
        except Exception as exc:
            raise FactoryInvocationError(*exc.args) from exc

    async def _add(self, tag_data: TagData) -> None:
        """
        将新资源推入空闲池并更新时间戳.

        Args:
            tag_data: 待入池的资源包装.
        """
        tag_data.timestamp = self._loop.time()
        tag_data.stale = False
        self._idle_pool.push(tag_data)

    async def _remove(self, tag_data: TagData) -> None:
        """⚠️
        从池中移除资源, 可能抛出 ResourceUnavailableError.

        Args:
            tag_data: 待移除的资源包装.

        Raises:
            ResourceUnavailableError: 资源在空闲池和活动池均未找到时抛出.
        """
        if self._idle_pool.pop(QueryField(None, tag_data)) is None:
            if self._active_pool.pop(tag_data, None) is None:
                raise ResourceUnavailableError("Resource not found in the pool.")

    async def _run_all(
        self,
        coros: Iterable[Coroutine[Any, Any, T]],
        concurrency: int = 0,
        raise_exception: bool = False,
    ) -> list[asyncio.Task[T]]:
        """
        并发执行一组协程, 并根据配置限制并发数量.

        Args:
            coros: 要执行的协程可迭代对象.
            concurrency: 最大并发任务数(<=0 则自动根据 CPU 核心数设置).
            raise_exception: 若为 True, 首个出现异常时立即终止并抛出.

        Returns:
            list[asyncio.Task[T]]: 已创建并执行的任务列表.

        Raises:
            Exception: 当 raise_exception 为 True 且有任务抛出时, 重抛该异常.
        """
        coro_iter = iter(coros)
        if concurrency <= 0:
            concurrency = min(32, (os.cpu_count() or 1) + 4)

        tasks = [asyncio.create_task(coro) for coro in islice(coro_iter, concurrency)]

        pending = set(tasks)
        while pending:
            done, pending = await asyncio.wait(
                pending, return_when=asyncio.FIRST_COMPLETED
            )

            for finished_task in done:
                if raise_exception and (exc := finished_task.exception()):
                    raise exc from exc

            for next_coro in islice(coro_iter, len(done)):
                next_task = asyncio.create_task(next_coro)
                tasks.append(next_task)
                pending.add(next_task)

        return tasks
