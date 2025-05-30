import tensorflow as tf

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

if tf.config.list_physical_devices('GPU'):
    print("TensorFlow is using the GPU")
    # 打印GPU设备信息
    for gpu in tf.config.list_physical_devices('GPU'):
        print(f"Name: {gpu.name}, Type: {gpu.device_type}")
else:
    print("TensorFlow is NOT using the GPU")