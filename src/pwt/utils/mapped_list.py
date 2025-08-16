"""
MappedLinkedList 模块

在内部使用 OrderedDict 维护元素的插入顺序, 并基于指定字段构建索引映射.
支持按字段值或按实例的快速查询/遍历和删除操作.

核心组件:
- MappedLinkedList[T]: 主类, 管理元素顺序与字段索引.
- QueryField: 字段查询包装器, 用于构造按字段值或实例匹配的查询条件.
- DataProxy: 运行时动态生成的代理类型, 锁定索引字段, 不允许修改.

基本示例:
    >>> class Person:
    ...     def __init__(self, name, age):
    ...         self.name, self.age = name, age

    >>> mlist = MappedLinkedList(Person, ["name", "age"])
    >>> mlist.push(Person("Alice", 30))
    >>> mlist.push(Person("Bob", 25))
    >>> mlist.pull(QueryField("name")("Alice"))
    <Person name='Alice' age=30>
"""

from __future__ import annotations

from collections import OrderedDict
from typing import (
    Any,
    Generic,
    Iterable,
    Iterator,
    Reversible,
    Type,
    TypeVar,
    cast,
)

T = TypeVar("T")

_ID = int
_Field = str
_Data = Any
_ItemMap = OrderedDict[_ID, T]
_DataMap = dict[_Data, _ItemMap[T]]
_FieldMap = dict[_Field, _DataMap[T]]


class QueryField:
    """
    封装查询字段名称与匹配值, 用于在 MappedLinkedList 中按字段或按实例检索元素.

    Attributes:
        field (str | None): 要匹配的属性名; 为 None 表示按实例匹配.
        data (Any): 字段值或要匹配/移除的实例对象.
    """

    def __init__(self, field: _Field | None, data: _Data | None = None) -> None:
        """
        初始化 QueryField.

        Args:
            field (str | None): 要查询的属性名; 为 None 则直接按实例匹配.
            data (Any, optional): 要匹配的字段值或实例; 默认为 None.
        """
        self.field: _Field | None = field
        self.data: _Data | None = data

    def __call__(self, data: _Data | None) -> QueryField:
        """
        设置匹配值并返回自身, 支持链式调用.

        Args:
            data (Any): 要匹配的字段值或实例.

        Returns:
            QueryField: self实例, 且 data 属性已更新.
        """
        self.data = data
        return self


class MappedLinkedList(Generic[T]):
    """
    支持字段索引与有序访问的数据结构.

    内部通过 `_items` (OrderedDict)维护元素的插入顺序,
    并通过 `_mapping`(dict)实现字段名 → 字段值 → OrderedDict(id→实例) 的多级映射,
    用于快速按字段或按实例定位与移除节点.

    主要操作:
      - push(item, first): 在头部或尾部插入新元素, 并更新索引.
      - peek/peek_all(query): 返回匹配元素但不删除.
      - pull/pull_all(query): 返回并删除匹配元素.
      - clear(): 清空所有元素与索引映射.
    """

    def __init__(
        self,
        data_type: Type[T],
        query_fields: Iterable[_Field],
        items: Iterable[T] | None = None,
    ) -> None:
        """
        初始化 MappedLinkedList.

        Args:
            data_type (Type[T]): 允许入列的元素类型, 仅支持基类实例, 不允许子类.
            query_fields (Iterable[str]): 要建立索引的属性名列表.
            items (Iterable[T], optional): 初始批量插入的元素序列.
        """
        self.data_type: Type[T] = data_type
        self.query_fields: set[str] = set(query_fields)
        self.DataProxy: type[T] = self._make_dataproxy()
        self._items: _ItemMap = OrderedDict()
        self._mapping: _FieldMap[T] = {}
        if items is not None:
            self.push_all(items)

    def __len__(self) -> int:
        """返回数据结构中元素总数."""
        return len(self._items)

    def __contains__(self, item: T) -> bool:
        """按实例判断元素是否在数据结构中."""
        return id(item) in self._items

    def __iter__(self) -> Iterator[T]:
        """顺序迭代所有数据实例."""
        return iter(self._items.values())

    def __reversed__(self) -> Iterator[T]:
        """逆序迭代所有数据实例."""
        return reversed(self._items.values())

    def __repr__(self) -> str:
        """返回当前数据结构状态的字符串表示."""
        vals = list(self._items.values())
        return f"{self.__class__.__name__}({self.data_type.__name__}, {self.query_fields}, {vals})"

    def _make_dataproxy(self) -> type[T]:
        """
        动态创建继承自 data_type 的代理类.

        特性:
          - 禁用 __init__, 禁止直接实例化.
          - 禁止对索引字段执行赋值／删除操作.
          - 在类属性层面为每个索引字段绑定一个 QueryField 占位.

        Returns:
            新生成的代理类型(DataProxy).
        """

        class DataProxy(self.data_type):
            __dataproxy_owner__: MappedLinkedList = self

            def __init__(self) -> None:
                raise TypeError("Cannot instantiate DataProxy directly")

            def __setattr__(self, name: str, value: Any, /) -> None:
                if name in self.__dataproxy_owner__.query_fields:
                    raise AttributeError(f"Cannot modify indexed field '{name}'")
                super().__setattr__(name, value)

            def __delattr__(self, name: str, /) -> None:
                if name in self.__dataproxy_owner__.query_fields:
                    raise AttributeError(f"Cannot delete indexed field '{name}'")
                super().__delattr__(name)

        for field in self.query_fields:
            setattr(DataProxy, field, QueryField(field))

        return cast(type[T], DataProxy)

    def _table_add(self, item: T, first: bool = False) -> None:
        """
        将数据实例加入 _items 和 _mapping 索引.

        Args:
            item (T): 已转换为 DataProxy 的实例.
            first (bool): True 插入至头部; False 追加到尾部.
        """
        item_id = id(item)
        self._items[item_id] = item
        if first:
            self._items.move_to_end(item_id, last=False)

        for field in self.query_fields:
            data_map = self._mapping.setdefault(field, {})
            try:
                key = getattr(item, field)
            except (AttributeError, TypeError):
                continue

            bucket = data_map.setdefault(key, OrderedDict())
            bucket[item_id] = item
            if first:
                bucket.move_to_end(item_id, last=False)

    def _table_get_all(self, field: _Field | None, data: _Data) -> Reversible[T]:
        """
        按字段名与字段值或按实例, 返回所有匹配的数据实例序列.

        Args:
            field (str | None): None 表示按实例匹配; 否则按该字段.
            data (Any): 字段值或实例对象.

        Returns:
            Reversible[T]: 匹配的数据实例集合(可逆迭代器).
        """
        if field is None:
            inst = self._items.get(id(data))
            return [inst] if inst is not None else []

        data_map = self._mapping.get(field)
        if not data_map:
            return []
        bucket = data_map.get(data)
        return bucket.values() if bucket is not None else []

    def _table_remove(self, item: T) -> None:
        """
        从 _items 与 _mapping 中移除指定代理实例的所有记录.

        Args:
            item (T): 待删除的 DataProxy 实例.
        """
        item_id = id(item)
        self._items.pop(item_id, None)

        for field in self.query_fields:
            data_map = self._mapping.get(field)
            if not data_map:
                continue
            try:
                key = getattr(item, field)
            except (AttributeError, TypeError):
                continue

            bucket = data_map.get(key)
            if not bucket:
                continue

            bucket.pop(item_id, None)
            if not bucket:
                data_map.pop(key, None)
            if not data_map:
                self._mapping.pop(field, None)

    def push(self, item: T, first: bool = False) -> None:
        """
        向数据结构插入元素并更新索引.

        Args:
            item (T): 待插入的基类实例(不接受子类).
            first (bool): True 插入到头部; False 插入到尾部.

        Raises:
            TypeError: item 类型不符.
            ValueError: 实例已存在.
        """
        if type(item) is not self.data_type:
            raise TypeError("Only base type allowed, no subclasses.")
        if id(item) in self._items:
            raise ValueError(f"Item {item} already in list.")

        item.__class__ = self.DataProxy
        self._table_add(item, first)

    def push_all(self, items: Iterable[T], first: bool = False) -> None:
        """
        批量插入元素.

        Args:
            items (Iterable[T]): 待插入实例序列.
            first (bool): True 插入到头部; False 插入到尾部.
        """
        for item in items:
            self.push(item, first)

    def peek(self, query: QueryField | None = None, first: bool = False) -> T | None:
        """
        返回匹配元素但不删除.

        Args:
            query (QueryField | None): 查询条件; None 表示顺序出队.
            first (bool): True 从头部取; False 从尾部取.

        Returns:
            T | None: 匹配实例; 无匹配时返回 None.
        """
        if not self._items:
            return None

        container = (
            self._items.values()
            if query is None
            else self._table_get_all(query.field, query.data)
        )
        return next(iter(container) if first else reversed(container), None)

    def peek_all(self, query: QueryField | None = None) -> list[T]:
        """
        返回所有匹配元素但不删除.

        Args:
            query (QueryField | None): 查询条件; None 表示全部元素.

        Returns:
            list[T]: 匹配实例列表.
        """
        if not self._items:
            return []

        return (
            list(self._items.values())
            if query is None
            else list(self._table_get_all(query.field, query.data))
        )

    def pop(self, query: QueryField | None = None, first: bool = False) -> T | None:
        """
        移除并返回单个匹配元素.

        Args:
            query (QueryField | None): 查询条件; None 表示顺序出队.
            first (bool): True 从头部移除; False 从尾部移除.

        Returns:
            T | None: 被移除实例; 无匹配时返回 None.
        """
        item = self.peek(query, first)
        if item is None:
            return None

        self._table_remove(item)
        item.__class__ = self.data_type
        return item

    def pop_all(self, query: QueryField | None = None) -> list[T]:
        """
        移除并返回所有匹配元素.

        Args:
            query (QueryField | None): 查询条件; None 表示移除所有.

        Returns:
            list[T]: 被移除实例列表.
        """
        items = self.peek_all(query)
        for item in items:
            self._table_remove(item)
            item.__class__ = self.data_type
        return items

    def clear(self) -> None:
        """清空所有元素及索引映射."""
        self._items.clear()
        self._mapping.clear()
