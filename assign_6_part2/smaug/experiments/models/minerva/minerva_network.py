#!/usr/bin/env python

"""Create the Minerva network."""

import numpy as np
import smaug as sg

def generate_random_data(shape):
  r = np.random.RandomState(1234)
  return (r.rand(*shape) * 0.005).astype(np.float16)

def create_minerva_model():
  with sg.Graph(name="minerva_smv", backend="SMV") as graph:
    # Tensors and kernels are initialized as NCHW layout.
    input_tensor = sg.Tensor(data_layout=sg.NC, tensor_data=generate_random_data((256,256)))
    fc0_tensor = sg.Tensor(data_layout=sg.CN, tensor_data=generate_random_data((256,256)))
    act = sg.input_data(input_tensor)
    act = sg.nn.mat_mul(act, fc0_tensor)
    return graph

if __name__ != "main":
  graph = create_minerva_model()
  graph.print_summary()
  graph.write_graph()
