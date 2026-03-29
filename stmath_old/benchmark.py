# import time
# import tracemalloc
# import math

# def timeit(fn, *args, runs=100, warmup=10, **kwargs):
#     # Warmup runs to stabilize CPU cache/interpreter
#     for _ in range(warmup):
#         fn(*args, **kwargs)
        
#     times = []
#     for _ in range(runs):
#         t0 = time.perf_counter()
#         fn(*args, **kwargs)
#         t1 = time.perf_counter()
#         times.append(t1 - t0)
    
#     avg = sum(times) / runs
#     variance = sum((t - avg) ** 2 for t in times) / runs
#     std_dev = math.sqrt(variance)
    
#     return {
#         "average_time": avg,
#         "std_dev": std_dev,
#         "total_runs": runs
#     }

# def mem_profile(fn, *args, **kwargs):
#     tracemalloc.start()
#     fn(*args, **kwargs)
#     current, peak = tracemalloc.get_traced_memory()
#     tracemalloc.stop()
#     return {
#         "current_kb": current / 1024.0,
#         "peak_kb": peak / 1024.0
#     }

# def compare_performance(fn1, fn2, *args, **kwargs):
#     # Compares two implementations (e.g., STMATH vs Math library)
#     res1 = timeit(fn1, *args, **kwargs)
#     res2 = timeit(fn2, *args, **kwargs)
#     diff = (res2["average_time"] - res1["average_time"]) / res2["average_time"] * 100
#     return f"Implementation 1 is {abs(diff):.2f}% {'faster' if diff < 0 else 'slower'} than Implementation 2"