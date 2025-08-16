# tests/test_tag_pool.py
import asyncio

import pytest

from pwt.utils.tag_pool import PoolClosedError, ResourceUnavailableError, TagData, TagPool

pytestmark = pytest.mark.asyncio


@pytest.fixture
def pool_factory():
    """
    Returns an async factory function: await factory(...) -> (pool, record)
    `record` is a list of destroyed TagData, for post-mortem assertions.
    """

    async def _make_pool(
        *,
        max_size: int = 3,
        min_size: int = 1,
        max_idle_time: float = 600.0,
        register_factory=None,
        destroy_factory=None,
    ):
        record: list[TagData] = []

        async def reg() -> TagData:
            td = TagData(data="payload", tag=f"tag-{len(record)}")
            return td

        async def dest(td: TagData) -> None:
            record.append(td)

        pool = await TagPool.create(
            loop=asyncio.get_running_loop(),
            max_size=max_size,
            min_size=min_size,
            max_idle_time=max_idle_time,
            register_factory=register_factory or reg,
            destroy_factory=destroy_factory or dest,
        )
        # attach record for tests
        pool._destroyed = record  # type: ignore
        return pool, record

    return _make_pool


async def test_initial_min_size(pool_factory):
    pool, record = await pool_factory(min_size=2)
    assert await pool.idle_length() == 2
    assert await pool.active_length() == 0
    # 尚未执行任何销毁操作
    assert record == []
    await pool.close()


async def test_manual_register_and_length(pool_factory):
    pool, record = await pool_factory(min_size=0)
    assert await pool.idle_length() == 0
    await pool.register(TagData(data=1, tag="x"))
    assert await pool.idle_length() == 1
    # 手动销毁该资源
    td = await pool.acquire()
    await pool.release(td)
    await pool.destroy(td)
    assert record == [td]
    await pool.close()


async def test_acquire_release_roundtrip(pool_factory):
    pool, record = await pool_factory(min_size=1)
    # 获取一个资源
    td = await pool.acquire()
    assert await pool.active_length() == 1
    assert await pool.idle_length() == 0
    # 释放该资源
    await pool.release(td)
    assert await pool.active_length() == 0
    assert await pool.idle_length() == 1
    # 确认记录仍然为空
    assert record == []
    await pool.close()


async def test_lease_context_manager(pool_factory):
    pool, record = await pool_factory(min_size=1)
    async with pool("any-tag") as td:
        # 在上下文中，资源处于激活状态
        assert await pool.is_active(td)
        assert td.tag is not None
    # 退出上下文后，资源应为空闲状态
    assert not (await pool.is_active(td))
    assert await pool.idle_length() == 1
    await pool.close()


async def test_submit_executes_task(pool_factory):
    pool, record = await pool_factory(min_size=1)

    async def echo(td: TagData, x: int):
        # 模拟执行任务
        await asyncio.sleep(0)
        return (td.tag, x)

    result = await pool.submit(echo, None, 42)
    # echo 应返回 (tag, 42)
    assert isinstance(result, tuple) and result[1] == 42
    await pool.close()


async def test_invalid_release_and_acquire_errors(pool_factory):
    pool, record = await pool_factory(min_size=1)
    bad = TagData(data=0, tag="bad")
    with pytest.raises(ResourceUnavailableError):
        await pool.release(bad)
    await pool.close()
    # 池关闭后，再调用 acquire 应抛出 RuntimeError
    with pytest.raises(PoolClosedError):
        await pool.acquire()


async def test_destroy_and_double_destroy(pool_factory):
    pool, record = await pool_factory(min_size=1)
    td = await pool.acquire()
    await pool.release(td)
    # 第一次销毁应成功
    await pool.destroy(td)
    assert record == [td]
    # 检查 stale 标志已设置
    assert td.stale
    # 第二次销毁应抛出 ValueError
    with pytest.raises(ResourceUnavailableError):
        await pool.destroy(td)
    await pool.close()


async def test_close_cleans_up(pool_factory):
    pool, record = await pool_factory(min_size=3)
    # close 应销毁所有三个空闲资源
    await pool.close()
    assert len(record) == 3
    # 再次 close 应无任何操作
    await pool.close()
    assert len(record) == 3


pytest.mark.asyncio


async def test_max_size_concurrency(pool_factory):
    pool, record = await pool_factory(max_size=2, min_size=0)

    # 同时获取两次，应当都能成功
    t1, t2 = await asyncio.gather(pool.acquire(), pool.acquire())
    assert await pool.active_length() == 2

    # 第三次 acquire 会被挂起
    acquire_third = asyncio.create_task(pool.acquire())

    # 让事件循环跑一下，确保第三个任务尚未完成
    await asyncio.sleep(0.05)
    assert not acquire_third.done()

    # 释放一个资源，让第三个任务继续进行
    await pool.release(t1)
    t3 = await acquire_third
    assert isinstance(t3, TagData)

    # 收尾：释放所有资源并关闭池
    await pool.release(t2)
    await pool.release(t3)
    await pool.close()
