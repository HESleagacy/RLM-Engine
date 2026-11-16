"""Keep a bounded head/tail while accounting for all printed characters."""

import io


class BoundedStdout(io.TextIOBase):
    def __init__(self, limit=65_536):
        super().__init__()
        if limit < 0:
            raise ValueError("stdout limit must be non-negative")
        self.limit = limit
        self.total = 0
        self.head = ""
        self.tail = ""

    def write(self, value):
        if not isinstance(value, str):
            raise TypeError("stdout accepts text")
        self.total += len(value)
        head_limit = (self.limit + 1) // 2
        take = min(len(value), head_limit - len(self.head))
        if take:
            self.head += value[:take]
        tail_limit = self.limit - head_limit
        if tail_limit:
            remaining = len(value) - take
            self.tail = value[-tail_limit:] if remaining >= tail_limit else (self.tail + value[take:])[-tail_limit:]
        return len(value)

    @property
    def truncated(self):
        return self.total > self.limit

    def getvalue(self):
        return self.head + self.tail
