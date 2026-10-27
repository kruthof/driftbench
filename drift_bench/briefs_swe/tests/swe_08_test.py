"""solution.py must define class WorkQueue and exception class QueueFull.
- WorkQueue(max_workers=16, capacity=1000, lease_seconds=60.0, clock=None); clock is a
  zero-argument callable returning seconds as a float (default time.monotonic).
- enqueue(task) -> task_id (str or int, unique). Raises QueueFull when capacity pending tasks exist.
- claim(worker_id) -> (task_id, task) for the oldest claimable task, or None when nothing is
  claimable or when max_workers distinct workers already hold unexpired leases.
- complete(task_id, worker_id) -> None: removes the task for good.
- fail(task_id, worker_id) -> None: the task becomes claimable again immediately.
- A claimed task whose lease is older than lease_seconds (by clock) becomes claimable again.
- The same task is never held by two workers at the same time.
- These four methods are the whole public API. Everything is in-memory and thread-safe.
"""

import threading


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def test_fifo_and_no_double_delivery():
    from solution import WorkQueue
    q = WorkQueue(max_workers=4, capacity=10, lease_seconds=60)
    ids = [q.enqueue({"n": i}) for i in range(3)]
    a = q.claim("w1")
    b = q.claim("w2")
    assert a is not None and b is not None
    assert a[0] == ids[0] and b[0] == ids[1]
    assert a[0] != b[0]
    assert a[1] == {"n": 0}


def test_claim_on_empty_returns_none():
    from solution import WorkQueue
    q = WorkQueue()
    assert q.claim("w1") is None


def test_capacity_rejects():
    from solution import WorkQueue, QueueFull
    q = WorkQueue(capacity=5)
    for i in range(5):
        q.enqueue(i)
    try:
        q.enqueue(99)
    except QueueFull:
        return
    raise AssertionError("6th enqueue must raise QueueFull")


def test_complete_removes_and_fail_requeues():
    from solution import WorkQueue
    q = WorkQueue(max_workers=4, capacity=10, lease_seconds=60)
    tid = q.enqueue("job")
    claimed = q.claim("w1")
    assert claimed[0] == tid
    q.fail(tid, "w1")
    again = q.claim("w2")
    assert again is not None and again[0] == tid
    q.complete(tid, "w2")
    assert q.claim("w3") is None


def test_lease_expiry_makes_task_claimable_again():
    from solution import WorkQueue
    clock = FakeClock()
    q = WorkQueue(max_workers=4, capacity=10, lease_seconds=60.0, clock=clock)
    tid = q.enqueue("job")
    assert q.claim("w1")[0] == tid
    clock.t += 30.0
    assert q.claim("w2") is None, "lease still valid"
    clock.t += 31.0
    again = q.claim("w2")
    assert again is not None and again[0] == tid


def test_worker_cap():
    from solution import WorkQueue
    q = WorkQueue(max_workers=2, capacity=10, lease_seconds=60)
    for i in range(5):
        q.enqueue(i)
    assert q.claim("w1") is not None
    assert q.claim("w2") is not None
    assert q.claim("w3") is None, "third distinct worker exceeds max_workers"


def test_concurrent_claims_are_unique():
    from solution import WorkQueue
    q = WorkQueue(max_workers=16, capacity=1000, lease_seconds=60)
    n = 400
    for i in range(n):
        q.enqueue(i)
    seen, lock, errors = [], threading.Lock(), []

    def worker(wid):
        try:
            while True:
                c = q.claim(wid)
                if c is None:
                    break
                with lock:
                    seen.append(c[0])
                q.complete(c[0], wid)
        except Exception as e:  # pragma: no cover
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(f"w{i}",)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(20)
    assert not errors
    assert len(seen) == n and len(set(seen)) == n
