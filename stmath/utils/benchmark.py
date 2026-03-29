import time
import tracemalloc
from ..core.math_kernels import sqrt   # 🔥 custom kernel


class Benchmark:

    @staticmethod
    def timeit(fn, *args, runs=100, warmup=10, **kwargs):


        for _ in range(warmup):
            fn(*args, **kwargs)

        times = []
        for _ in range(runs):
            t1 = time.perf_counter()
            fn(*args, **kwargs)
            t2 = time.perf_counter()
            times.append(t2 - t1)

        avg = sum(times) / len(times)
        variance = sum((t - avg) ** 2 for t in times) / len(times)

        std_dev = sqrt(variance)

        return {
            "avg_time": avg,
            "std_dev": std_dev,
            "runs": runs
        }

    @staticmethod
    def memory(fn, *args, **kwargs):

        tracemalloc.start()
        fn(*args, **kwargs)
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        return {
            "current_kb": current / 1024,
            "peak_kb": peak / 1024
        }

    @staticmethod
    def compare(fn1, fn2, *args, runs=100, **kwargs):

        t1 = Benchmark.timeit(fn1, *args, runs=runs, **kwargs)["avg_time"]
        t2 = Benchmark.timeit(fn2, *args, runs=runs, **kwargs)["avg_time"]

        diff = ((t2 - t1) / t2 * 100) if t2 != 0 else 0.0

        return {
            "your_impl_time": t1,
            "baseline_time": t2,
            "difference_percent": diff,
            "faster": diff < 0
        }

    @staticmethod
    def quick_compare(fn1, fn2, *args):

        t1_start = time.perf_counter()
        fn1(*args)
        t1_end = time.perf_counter()

        t2_start = time.perf_counter()
        fn2(*args)
        t2_end = time.perf_counter()

        return {
            "your_time": t1_end - t1_start,
            "baseline_time": t2_end - t2_start
        }