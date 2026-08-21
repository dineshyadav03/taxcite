"""Per-session rate limiting - a real access-control gap named directly
in this project's own honest "what would production need" review, not
something skipped by accident.

Deliberately scoped to what this can actually guarantee: keyed by
session_id, which is client-supplied (from the browser's localStorage)
and trivially spoofed by clearing storage or opening a private window -
this is NOT a security boundary, it's a fairness mechanism so one
browser tab can't monopolize the shared free-tier throttle every other
visitor depends on too. A real production deployment would need
IP-based or authenticated rate limiting layered on top of this, not
instead of it - that's a named, open gap, not something this module
claims to solve.
"""
import threading
import time
from collections import defaultdict, deque

_WINDOW_SECONDS = 300
_MAX_PER_WINDOW = 20

_lock = threading.Lock()
_requests = defaultdict(deque)  # session_id -> deque of request timestamps


def check(session_id):
    """True if this session may proceed, False if it's over the limit
    for the current window. Only a call that's actually allowed to
    proceed gets recorded - a rejected attempt doesn't consume the
    session's quota, so the window rolls forward naturally as old
    entries age out rather than needing an explicit reset."""
    now = time.monotonic()
    with _lock:
        q = _requests[session_id]
        while q and now - q[0] > _WINDOW_SECONDS:
            q.popleft()
        if len(q) >= _MAX_PER_WINDOW:
            return False
        q.append(now)
        return True
