# encoding: utf-8

import tensorrt as trt
import pycuda.driver as cuda
import numpy as np
import pycuda.autoinit 
import torch

def allocate_buffers(engine, batch_size, data_type):

    """
    This is the function to allocate buffers for input and output in the device
    Args:
    engine : The path to the TensorRT engine. 
    batch_size : The batch size for execution time.
    data_type: The type of the data for input and output, for example trt.float32. 

    Output:
    h_input_1: Input in the host.
    d_input_1: Input in the device. 
    h_output_1: Output in the host. 
    d_output_1: Output in the device. 
    stream: CUDA stream.

    """
    #print("\n\n\n:")
    #print(engine.get_binding_shape(0))
    #print("\n\n\n:")
    #print(batch_size * trt.volume(engine.get_binding_shape(0)))

    # Determine dimensions and create page-locked memory buffers (which won't be swapped to disk) to hold host inputs/outputs.
    #h_input_1 = cuda.pagelocked_empty(batch_size * trt.volume(engine.get_binding_shape(0)), dtype=trt.nptype(data_type))
    h_input_1 = cuda.pagelocked_empty(3*640*480, dtype=trt.nptype(data_type))
    #h_output = cuda.pagelocked_empty(batch_size * trt.volume(engine.get_binding_shape(1)), dtype=trt.nptype(data_type))
    h_output = cuda.pagelocked_empty(3*80*60, dtype=trt.nptype(data_type))
    # Allocate device memory for inputs and outputs.
    d_input_1 = cuda.mem_alloc(h_input_1.nbytes)

    d_output = cuda.mem_alloc(h_output.nbytes)
    # Create a stream in which to copy inputs/outputs and run inference.
    stream = cuda.Stream()
    return h_input_1, d_input_1, h_output, d_output, stream 


# 640* 480 * 3  
def load_images_to_buffer(pics, pagelocked_buffer):
    preprocessed = np.asarray(pics).ravel()
    np.copyto(pagelocked_buffer, preprocessed) 


def do_inference(engine, pics_1, h_input_1, d_input_1, h_output, d_output, stream, batch_size, height, width):
    """
    This is the function to run the inference
    Args:
        engine : Path to the TensorRT engine 
        pics_1 : Input images to the model.  
        h_input_1: Input in the host         
        d_input_1: Input in the device 
        h_output_1: Output in the host 
        d_output_1: Output in the device 
        stream: CUDA stream
        batch_size : Batch size for execution time
        height: Height of the output image
        width: Width of the output image

    Output:
        The list of output images

    """


    #print(h_input_1.shape)

    #print(pics_1[0])

    load_images_to_buffer(pics_1, h_input_1)

    #print("Será que deu load?")
    #print(h_input_1)


    with engine.create_execution_context() as context:
        # Transfer input data to the GPU.
        cuda.memcpy_htod_async(d_input_1, h_input_1, stream)

        # Run inference.

        context.profiler = trt.Profiler()
        context.execute(batch_size=1, bindings=[int(d_input_1), int(d_output)])
        #context.execute(bindings=[int(d_input_1), int(d_output)], stream_handle=stream.handle)
        
        # Transfer predictions back from the GPU.
        cuda.memcpy_dtoh_async(h_output, d_output, stream)
        # Synchronize the stream
        stream.synchronize()
        # Return the host output.
        out = h_output.reshape((batch_size,-1, 80, 60))
        #out = h_output.reshape((batch_size,-1, 80, 60))

        #print(out[0])

        torch_out = torch.from_numpy(out) 
        

        torch_out = torch_out[0]

        #print("Torch original: " + str(torch_out))

        torch_out = torch.exp(torch_out)
        #print("torch_score: " + str(torch_out))

        #print("\n\n")
        #print(out[0])

        return torch_out