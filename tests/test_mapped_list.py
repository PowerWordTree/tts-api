import pytest

from pwt.utils.mapped_list import MappedLinkedList, QueryField


class Person:
    def __init__(self, name: str, age: int, note: str = ""):
        self.name = name
        self.age = age
        self.note = note

    def __repr__(self):
        return f"<Person name={self.name!r} age={self.age!r} note={self.note!r}>"


@pytest.fixture
def sample_list():
    people = [Person("AA", 11), Person("BB", 22), Person("CC", 33)]
    m = MappedLinkedList(Person, ["name", "age"])
    m.push_all(people)
    return m, people


def test_len_and_contains_and_iteration(sample_list):
    mlist, people = sample_list
    # length
    assert len(mlist) == 3
    # contains
    assert people[1] in mlist
    assert Person("XX", 99) not in mlist
    # 正向迭代 yields proxies wrapping original objects
    seen_names = [p.name for p in mlist]
    assert seen_names == ["AA", "BB", "CC"]
    # 反向迭代
    seen_ages = [p.age for p in reversed(mlist)]
    assert seen_ages == [33, 22, 11]


def test_peek_and_peek_all(sample_list):
    mlist, people = sample_list
    # peek 不带 query: first=True(头), first=False(尾)
    assert mlist.peek(first=True).name == "AA"
    assert mlist.peek(first=False).name == "CC"
    # peek_all 不带 query 返回所有
    all_names = [p.name for p in mlist.peek_all()]
    assert all_names == ["AA", "BB", "CC"]

    # 按字段 peek
    q = QueryField("age")(22)
    assert mlist.peek(q).age == 22
    assert mlist.peek_all(q)[0].name == "BB"

    # 不匹配时返回 None / 空列表
    assert mlist.peek(QueryField("name")("ZZZ")) is None
    assert mlist.peek_all(QueryField("age")(999)) == []


def test_pop_all_and_clear(sample_list):
    mlist, people = sample_list
    # pull_all 按字段移除所有 age>0(也可以用 None 匹配实例)
    removed = mlist.pop_all(QueryField("name")("AA"))
    assert isinstance(removed, list) and len(removed) == 1
    assert removed[0].name == "AA"
    assert len(mlist) == 2

    # 再清空
    mlist.clear()
    assert len(mlist) == 0
    # 任何 peek/pull now 都返回空
    assert mlist.peek() is None
    assert mlist.peek_all() == []
    assert mlist.pop() is None
    assert mlist.pop_all() == []


def test_push_all_behaviour():
    mlist = MappedLinkedList(Person, ["name"])
    batch = [Person("X", 1), Person("Y", 2)]
    mlist.push_all(batch, first=True)
    # first=True 时批量插入后, 顺序应为 Y, X
    names = [p.name for p in mlist]  # type: ignore
    assert names == ["Y", "X"]


def test_repr_contains_key_parts():
    mlist = MappedLinkedList(Person, ["name"])
    mlist.push(Person("R", 9))
    rep = repr(mlist)
    # 简单断言类名/字段名和元素 repr 都出现在输出里
    assert "MappedLinkedList" in rep
    assert "{'name'}" in rep or "name" in rep
    assert "<Person name='R' age=9" in rep


def test_push_and_pull_sequential():
    mlist = MappedLinkedList(Person, ["name", "age"])
    p1 = Person("A", 10)
    p2 = Person("B", 20)
    p3 = Person("C", 30)

    # 顺序 push, 然后按头/尾顺序 pull
    mlist.push(p1)
    mlist.push(p2)
    mlist.push(p3)
    assert mlist.pop() is p3
    assert mlist.pop(first=False) is p2
    assert mlist.pop(first=True) is p1
    # 空时返回 None
    assert mlist.pop() is None


def test_push_first_and_order_mix():
    mlist = MappedLinkedList(Person, ["name", "age"])
    p1 = Person("X", 1)
    p2 = Person("Y", 2)
    p3 = Person("Z", 3)

    mlist.push(p1)  # [X]
    mlist.push(p2, first=True)  # [Y, X]
    mlist.push(p3)  # [Y, X, Z]

    # 按顺序出队
    assert mlist.pop(first=True) is p2  # 从头取 Y
    assert mlist.pop(first=False) is p3  # 从尾取 Z
    assert mlist.pop() is p1  # 余下 X


def test_queryfield_single_match():
    mlist = MappedLinkedList(Person, ["name", "age"])
    p1 = Person("Alice", 30)
    p2 = Person("Bob", 25)
    p3 = Person("Alice", 40)

    mlist.push(p1)
    mlist.push(p2)
    mlist.push(p3)

    # QueryField 链式调用设置匹配值
    q_name = QueryField("name")("Alice")
    # first=True 返回最先插入的 Alice
    first_alice = mlist.pop(q_name, first=True)
    assert isinstance(first_alice, Person)
    assert first_alice.age == 30

    # 再次按同样 query first=False 返回最后插入的 Alice
    last_alice = mlist.pop(q_name, first=False)
    assert last_alice.age == 40  # type: ignore

    # Bob 仍在队列
    assert mlist.pop(QueryField("name")("Bob")) is p2
    assert mlist.pop() is None


def test_multiple_same_field_and_fifo_lifo():
    mlist = MappedLinkedList(Person, ["name"])
    ps = [Person("dup", i) for i in range(5)]
    for p in ps:
        mlist.push(p)

    # names 全部相同的节点, first=False 相当于 LIFO
    seen = [mlist.pop(QueryField("name")("dup")) for _ in range(5)]
    assert seen == ps[::-1]

    # 再次取空
    assert mlist.pop(QueryField("name")("dup")) is None


def test_data_proxy_immutable_fields():
    mlist = MappedLinkedList(Person, ["name", "age"])
    p = Person("Tom", 50)
    mlist.push(p)
    # 推入后 p.__class__ 已切换为 DataProxy
    with pytest.raises(AttributeError):
        p.name = "Jerry"
    with pytest.raises(AttributeError):
        p.age = 60

    # 非索引字段允许修改
    p.note = "new note"
    assert p.note == "new note"


def test_data_proxy_restored_after_pull():
    mlist = MappedLinkedList(Person, ["name"])
    p = Person("Zoe", 99)
    mlist.push(p)
    result = mlist.pop(QueryField("name")("Zoe"))
    # pull 返回后 __class__ 恢复到 Person
    assert isinstance(result, Person)
    # 可以自由修改索引字段
    result.name = "Z"
    assert result.name == "Z"


def test_push_wrong_type_raises():
    mlist = MappedLinkedList(Person, ["name"])

    class Student(Person):
        pass

    with pytest.raises(TypeError):
        mlist.push(Student("S", 18))


def test_empty_queries_and_nonindexed_field():
    mlist = MappedLinkedList(Person, ["name"])
    # 空列表 pull 返回 None
    assert mlist.pop() is None
    assert mlist.pop(QueryField("name")("Anything")) is None

    p = Person("Eve", 22)
    mlist.push(p)
    # 使用非索引字段查询直接按 None 返回
    assert mlist.pop(QueryField("note")("whatever")) is None


def test_mapping_table_cleanup():
    mlist = MappedLinkedList(Person, ["name"])
    people = [Person("A", 1), Person("B", 2), Person("A", 3)]
    for p in people:
        mlist.push(p)

    # 先取掉所有 A
    mlist.pop(QueryField("name")("A"), first=True)
    mlist.pop(QueryField("name")("A"), first=True)
    # 再取 B
    assert mlist.pop(QueryField("name")("B")) is people[1]
    # 此时链表与映射表已空
    assert mlist.pop() is None
    assert mlist.pop(QueryField("name")("A")) is None
