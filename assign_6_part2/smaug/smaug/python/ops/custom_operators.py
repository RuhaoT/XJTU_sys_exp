from smaug.core import node_pb2, types_pb2
from smaug.python.ops import common

def my_custom_operator(tensor_a, tensor_b, name="my_custom_operator"):
  if tensor_a.shape.dims != tensor_b.shape.dims:
    raise ValueError(
        "The input tensors to MyCustomOperator must be of the same shape")
  return common.add_node(
    name=name,
    op=types_pb2.CustomOp,
    input_tensors=[tensor_a, tensor_b],
    output_tensors_dims=[tensor_a.shape.dims],
    output_tensor_layout=tensor_a.shape.layout)[0]