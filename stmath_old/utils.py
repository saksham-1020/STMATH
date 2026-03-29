# import time
# import sys
# from typing import Iterable, List, Any, Callable

# class Utils:
#     @staticmethod
#     def ensure_list(x: Any) -> List[Any]:
#         """Converts any iterable to a list; if single element, wraps it."""
#         if isinstance(x, (list, tuple, set)):
#             return list(x)
#         return [x]

#     @staticmethod
#     def flatten(nested_list: Iterable) -> List[Any]:
#         """Deep flattens a nested list (O(n)). Essential for matrix operations."""
#         flat_list = []
#         for item in nested_list:
#             if isinstance(item, Iterable) and not isinstance(item, (str, bytes)):
#                 flat_list.extend(Utils.flatten(item))
#             else:
#                 flat_list.append(item)
#         return flat_list

#     @staticmethod
#     def get_memory_usage(obj: Any) -> int:
#         """Returns memory size of an object in bytes. Used for MNC-grade optimization."""
#         return sys.getsizeof(obj)

#     @staticmethod
#     def chunk_data(data: List[Any], size: int) -> Iterable[List[Any]]:
#         """Yield successive n-sized chunks. Useful for Batch Processing in DL."""
#         for i in range(0, len(data), size):
#             yield data[i:i + size]