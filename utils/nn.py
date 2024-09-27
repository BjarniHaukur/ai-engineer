import gc
import inspect
import torch

def should_reduce_batch_size(exception: Exception) -> bool:
    error_msg = str(exception).lower()
    cuda_oom_indicators = [
        "cuda out of memory",
        "cudaoutofmemoryerror",
        "cuda runtime error",
        "memory allocation failed"
    ]
    return any(indicator in error_msg for indicator in cuda_oom_indicators)


def find_executable_batch_size(function: callable = None, starting_batch_size: int = 128):
    batch_size = starting_batch_size

    def decorator(*args, **kwargs):
        nonlocal batch_size
        gc.collect()
        torch.cuda.empty_cache()
        params = list(inspect.signature(function).parameters.keys())
        while True:
            if batch_size == 0:
                raise RuntimeError("No executable batch size found. Reached zero.")
            try:
                return function(batch_size, *args, **kwargs)
            except Exception as e:
                if should_reduce_batch_size(e):
                    gc.collect()
                    torch.cuda.empty_cache()
                    batch_size //= 2
                    print(f"Decreasing batch size to: {batch_size}")
                else:
                    raise

    return decorator
