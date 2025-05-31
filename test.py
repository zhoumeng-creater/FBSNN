import tensorflow as tf
import sys
import numpy

print(f"TensorFlow Version: {tf.__version__}")
print(f"Python Version: {sys.version}") # 使用 sys 模块获取 Python 版本
print(f"NumPy Version (runtime): {numpy.__version__}") # 获取当前环境中安装的 NumPy 版本

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"Detected {len(gpus)} Physical GPUs:")
    for i, gpu in enumerate(gpus):
        print(f"  GPU {i}: {gpu}")
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
            print(f"    Memory growth set for GPU {i}")
        except RuntimeError as e:
            print(f"    Error setting memory growth for GPU {i}: {e}")
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(f"Detected {len(logical_gpus)} Logical GPUs.")
else:
    print("No GPU detected by TensorFlow.") # 这是我们最关心的输出之一

print("\nChecking CUDA availability with tf.test.is_built_with_cuda():")
is_built_with_cuda = tf.test.is_built_with_cuda()
print(f"Is TensorFlow built with CUDA: {is_built_with_cuda}")

# 如果是使用CUDA构建的，进一步检查GPU设备
if is_built_with_cuda:
    print("\nAttempting to list GPU devices again (if built with CUDA):")
    gpu_devices_check = tf.config.experimental.list_physical_devices('GPU')
    if gpu_devices_check:
        print(f"  Successfully listed {len(gpu_devices_check)} GPU(s).")
    else:
        print("  Could not list any GPUs, even though TensorFlow is built with CUDA. This indicates a driver/runtime issue.")
else:
    print("  TensorFlow is NOT built with CUDA, so it cannot use NVIDIA GPUs.")


print("\nDevice placement logging (set to True for more details during model execution):")
print(f"tf.debugging.get_log_device_placement() -> {tf.debugging.get_log_device_placement()}")