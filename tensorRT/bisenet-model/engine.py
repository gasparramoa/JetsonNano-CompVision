import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
trt_runtime = trt.Runtime(TRT_LOGGER)
def build_engine(onnx_path, shape = [1,3,640,480]):

    """
    This is the function to create the TensorRT engine
    Args:
        onnx_path : Path to onnx_file. 
        shape : Shape of the input of the ONNX file. 
    """

    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(1) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
        builder.max_workspace_size = (256 << 3)
        with open(onnx_path, 'rb') as model:
            parser.parse(model.read())
        network.get_input(0).shape = shape

        # [Ramoa] - Layer tem de ter output se não queixa-se... Os tutoriais deviam ter isto implementado :/
        last_layer = network.get_layer(network.num_layers - 1)
        network.mark_output(last_layer.get_output(0))

        engine = builder.build_cuda_engine(network)
        return engine

def save_engine(engine, file_name):
    buf = engine.serialize()
    with open(file_name, 'wb') as f:
        f.write(buf)


def load_engine(trt_runtime, engine_path):
    with open(engine_path, 'rb') as f:
        engine_data = f.read()
    engine = trt_runtime.deserialize_cuda_engine(engine_data)
    return engine