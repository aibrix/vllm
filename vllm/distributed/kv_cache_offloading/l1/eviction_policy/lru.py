from typing import (Hashable, Tuple)
from ...memory import RefCountedObj
from ...status import Status, StatusCodes
from .base_eviction_policy import (
    BaseEvictionPolicy,
    BaseEvictionPolicyNode,
    Functor,
    V,
)


class LRUNode(BaseEvictionPolicyNode[V]):
    __slots__ = ("next", "prev")

    def __init__(self, key: Hashable, value: V):
        super().__init__(key, value)
        self.next: 'LRUNode' = None
        self.prev: 'LRUNode' = None


class LRU(BaseEvictionPolicy[LRUNode, V]):

    def __init__(
        self,
        capacity: int,
        evict_size: int = 1,
        on_evict: Functor | None = None,
        on_hot_access: Functor | None = None,
    ) -> None:
        super().__init__("LRU", capacity, evict_size, on_evict, on_hot_access)
        self._head: LRUNode = None
        self._tail: LRUNode = None

    def put(
        self,
        key: Hashable,
        value: V,
    ) -> Status:
        if key in self._hashmap:
            node = self._hashmap[key]

            if node.value is not None:
                if isinstance(node.value, RefCountedObj):
                    node.value.ref_down()
                node.value = None

            node.value = value
            node.hotness = 0

            self._remove_from_list(node)
            self._prepend_to_head(node)
        else:
            node = LRUNode(key, value)
            self._hashmap[key] = node
            self._prepend_to_head(node)

        if len(self) > self._capacity:
            self.evict(self.evict_size)

        return Status(StatusCodes.OK)

    def get(
        self,
        key: Hashable,
    ) -> Status[V]:
        if key not in self._hashmap:
            return Status(StatusCodes.NOT_FOUND)

        node = self._hashmap[key]

        # The item becomes hot after the first access
        if node.hotness == 0 and self._on_hot_access:
            if isinstance(node.value, RefCountedObj):
                node.value.ref_up()
            self._on_hot_access(node.key, node.value)

        self._remove_from_list(node)
        self._prepend_to_head(node)

        node.hotness = 1
        if isinstance(node.value, RefCountedObj):
            node.value.ref_up()
        return Status(value=node.value)

    def delete(self, key: Hashable) -> Status:
        node = self._hashmap.pop(key, None)
        if node:
            self._remove_from_list(node)

            if node.value is not None:
                if isinstance(node.value, RefCountedObj):
                    node.value.ref_down()
                node.value = None

        return Status(StatusCodes.OK)

    def evict(self, size: int = 1) -> Status:
        for _ in range(size):
            if not self._tail:
                break
            if self._on_evict:
                if isinstance(self._tail.value, RefCountedObj):
                    self._tail.value.ref_up()
                self._on_evict(self._tail.key, self._tail.value)
            evicted_node = self._tail
            self.delete(evicted_node.key)

        return Status(StatusCodes.OK)

    def assert_consistency(self) -> None:
        total_in_list = 0
        curr = self._head
        while curr is not None and curr.next != self._head:
            total_in_list += 1
            assert self._hashmap.get(curr.key, None) == curr
            curr = curr.next
        assert total_in_list == len(
            self._hashmap), f"{total_in_list} != {len(self._hashmap)}"

    def _prepend_to_head(self, node: LRUNode) -> None:
        node.next = self._head
        node.prev = None
        if self._head:
            self._head.prev = node
        self._head = node
        if self._tail is None:
            self._tail = node

    def _remove_from_list(self, node: LRUNode) -> None:
        if node.prev:
            node.prev.next = node.next
        if node.next:
            node.next.prev = node.prev
        if self._head == node:
            self._head = node.next
        if self._tail == node:
            self._tail = node.prev
