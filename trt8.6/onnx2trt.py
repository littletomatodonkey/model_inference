import trt_engine as eng
from onnx import ModelProto
import tensorrt as trt

engine_name = "resnet18.plan"
onnx_path = "./resnet18.onnx"
batch_size = 1 

model = ModelProto()
with open(onnx_path, "rb") as f:
    model.ParseFromString(f.read())

shape = [model.graph.input[0].type.tensor_type.shape.dim[idx].dim_value for idx in range(1, len(model.graph.input[0].type.tensor_type.shape.dim))]
shape = [batch_size] + shape
engine = eng.build_engine(onnx_path, shape=shape)
eng.save_engine(engine, engine_name) 
