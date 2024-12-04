import os
import tensorflow as tf
from collections import defaultdict, Counter
from datetime import datetime

class ValueStats:
    def __init__(self, name=None):
        self.name = name
        self.reset()

    def feed(self, v):
        self.summation += v
        if v > self.max_value:
            self.max_value = v
            self.max_idx = self.counter
        if v < self.min_value:
            self.min_value = v
            self.min_idx = self.counter

        self.counter += 1

    def mean(self):
        if self.counter == 0:
            print(f"Counter {self.name} is 0")
            assert False
        return self.summation / self.counter

    def summary(self, info=None):
        info = "" if info is None else info
        name = "" if self.name is None else self.name
        if self.counter > 0:
            return f"{info}{name}[{self.counter:4d}]: avg: {self.mean():8.4f}, min: {self.min_value:8.4f}[{self.min_idx:4d}], max: {self.max_value:8.4f}[{self.max_idx:4d}]"
        else:
            return f"{info}{name}[0]"

    def reset(self):
        self.counter = 0
        self.summation = 0.0
        self.max_value = -1e38
        self.min_value = 1e38
        self.max_idx = None
        self.min_idx = None

class MultiCounter:
    def __init__(self, root, verbose=False):
        self.last_time = None
        self.verbose = verbose
        self.counts = Counter()
        self.stats = defaultdict(lambda: ValueStats())
        self.total_count = 0
        self.max_key_len = 0
        self.summary_writer = None
        if root is not None:
            self.summary_writer = tf.summary.create_file_writer(os.path.join(root, "stat_tb"))

    def __getitem__(self, key):
        if len(key) > self.max_key_len:
            self.max_key_len = len(key)

        if key in self.counts:
            return self.counts[key]

        return self.stats[key]

    def inc(self, key):
        if self.verbose:
            print(f"[MultiCounter]: {key}")
        self.counts[key] += 1
        self.total_count += 1
        if self.last_time is None:
            self.last_time = datetime.now()

    def reset(self):
        for k in self.stats.keys():
            self.stats[k].reset()

        self.counts = Counter()
        self.total_count = 0
        self.last_time = datetime.now()

    def time_elapsed(self):
        return (datetime.now() - self.last_time).total_seconds()

    def summary(self, global_counter):
        assert self.last_time is not None
        time_elapsed = (datetime.now() - self.last_time).total_seconds()
        print(f"[{global_counter}] Time spent = {time_elapsed:.2f} s")

        for key, count in self.counts.items():
            print(f"{key}: {count}/{self.total_count}")

        for k in sorted(self.stats.keys()):
            v = self.stats[k]
            info = f"{global_counter}:{k}"
            print(v.summary(info=info.ljust(self.max_key_len + 4)))

            if self.summary_writer is not None:
                with self.summary_writer.as_default():
                    tf.summary.scalar(k, v.mean(), step=global_counter)
