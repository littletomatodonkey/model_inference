import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
from PIL import Image

# 用于加载 TRT 模型的函数
def load_engine(trt_runtime, engine_path):
    with open(engine_path, 'rb') as f:
        engine_data = f.read()
    engine = trt_runtime.deserialize_cuda_engine(engine_data)
    return engine

# 用于预处理图像的函数
def preprocess_image(image_path, input_shape):
    image = Image.open(image_path).convert('RGB')
    image = image.resize(input_shape)
    image = np.array(image).astype(np.float32)
    image /= 255.0  # 归一化
    image = np.transpose(image, [2, 0, 1])  # CHW
    return image

# 用于执行推理的函数
def infer(engine, input_img: np.ndarray):
    context = engine.create_execution_context()

    batch_size = 1

    input_size = trt.volume(engine.get_binding_shape(0)) * batch_size * np.dtype(np.float32).itemsize
    output_size = trt.volume(engine.get_binding_shape(1)) * batch_size * np.dtype(np.float32).itemsize
    
    input_img = np.ascontiguousarray(input_img.astype(np.float32))
    # Create host buffer to receive data
    output = np.empty(output_size // np.dtype(np.float32).itemsize, dtype = np.float32)
    # Allocate device memory
    d_input = cuda.mem_alloc(input_size)
    d_output = cuda.mem_alloc(output_size)

    print(f"d_input: {input_size}, d_output: {output_size}, output shape: {output.shape}")

    bindings = [int(d_input), int(d_output)]
    stream = cuda.Stream()
    # Transfer input data to device
    cuda.memcpy_htod_async(d_input, input_img, stream)
    # Execute model
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back
    cuda.memcpy_dtoh_async(output, d_output, stream)
    # Synchronize threads
    stream.synchronize()
    return output

if __name__ == "__main__":
    # 主要步骤
    TRT_ENGINE_PATH = "resnet18.plan"
    IMAGE_PATH = "./demo.jpg"
    # 加载模型
    trt_runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    engine = load_engine(trt_runtime, TRT_ENGINE_PATH)

    # 预处理图像
    input_shape = (224, 224)  # 假设输入尺寸为 224x224
    image = preprocess_image(IMAGE_PATH, input_shape)

    # 执行推理
    output = infer(engine, image)
    print(f"max label: {output.argmax()}")
