"""solution.py must define a decorator factory lru_ttl_cache(maxsize, ttl_seconds, clock=None).
- lru_ttl_cache(maxsize: int, ttl_seconds: float, clock=None) returns a decorator.
- clock is a zero-argument callable returning seconds as a float; when None, use time.monotonic.
- The decorated function keeps its name, signature and return type (functools.wraps or equivalent).
- Cache keys are built from the positional and keyword arguments; all test arguments are hashable.
- An entry older than ttl_seconds (by clock) is treated as missing on access.
- The cache never holds more than maxsize entries; eviction is least-recently-used.
- Concurrent calls from several threads must be safe.
"""

import threading


def _counting(fn):
    calls = {"n": 0}

    def inner(*a, **k):
        calls["n"] += 1
        return fn(*a, **k)

    inner.__name__ = fn.__name__
    return inner, calls


class FakeClock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def test_hit_avoids_recomputation():
    from solution import lru_ttl_cache
    inner, calls = _counting(lambda x: x * 2)
    f = lru_ttl_cache(maxsize=10, ttl_seconds=60)(inner)
    assert f(3) == 6 and f(3) == 6 and f(3) == 6
    assert calls["n"] == 1


def test_wrapped_name_and_return_type_unchanged():
    from solution import lru_ttl_cache

    def square(x):
        return x * x

    f = lru_ttl_cache(maxsize=4, ttl_seconds=60)(square)
    assert f.__name__ == "square"
    assert f(4) == 16 and isinstance(f(4), int)


def test_lru_eviction_order():
    from solution import lru_ttl_cache
    inner, calls = _counting(lambda x: x)
    f = lru_ttl_cache(maxsize=2, ttl_seconds=60)(inner)
    f("a"); f("b"); f("a")   # a is most recently used
    f("c")                   # evicts b
    n = calls["n"]
    f("a")
    assert calls["n"] == n, "a should still be cached"
    f("b")
    assert calls["n"] == n + 1, "b should have been evicted"


def test_size_never_exceeds_maxsize():
    from solution import lru_ttl_cache
    inner, calls = _counting(lambda x: x)
    f = lru_ttl_cache(maxsize=3, ttl_seconds=60)(inner)
    for i in range(10):
        f(i)
    assert calls["n"] == 10
    f(0)
    assert calls["n"] == 11, "oldest entry must have been evicted"
    f(9)
    assert calls["n"] == 11, "newest entry must still be cached"


def test_ttl_expiry_on_access():
    from solution import lru_ttl_cache
    clock = FakeClock()
    inner, calls = _counting(lambda x: x)
    f = lru_ttl_cache(maxsize=10, ttl_seconds=5.0, clock=clock)(inner)
    f(1)
    clock.t += 4.0
    f(1)
    assert calls["n"] == 1
    clock.t += 2.0
    f(1)
    assert calls["n"] == 2


def test_thread_safety_smoke():
    from solution import lru_ttl_cache
    inner, calls = _counting(lambda x: x + 1)
    f = lru_ttl_cache(maxsize=50, ttl_seconds=60)(inner)
    errors = []

    def worker():
        try:
            for i in range(200):
                assert f(i % 20) == (i % 20) + 1
        except Exception as e:  # pragma: no cover - reported below
            errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(10)
    assert not errors
    assert calls["n"] <= 8 * 20
