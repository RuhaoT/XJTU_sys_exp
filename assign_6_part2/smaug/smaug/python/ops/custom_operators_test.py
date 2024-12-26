import unittest
import numpy as np

from smaug.python.smaug_test import SmaugTest
from smaug.python.ops import custom_operators
from smaug.python.ops import data_op
from smaug.python.graph import Graph
from smaug.python.tensor import Tensor
from smaug.core import types_pb2

class CustomOperatorsTest(SmaugTest):
  def test_my_custom_operator_ref(self):
    self.backend = "Reference"
    with Graph(name=self.graph_name, backend="Reference") as graph:
      tensor_a = Tensor(data_layout=types_pb2.NC, tensor_data=np.array([1,2,3,4,5,6,7,8,9,10], dtype=np.float32)) # the tensor data type should match the dtype accepted my the custom operator
      tensor_b = Tensor(data_layout=types_pb2.NC, tensor_data=np.array([2,3,4,5,6,7,8,9,10,11], dtype=np.float32))
      act1 = data_op.input_data(tensor_a, "tensor_a")
      act2 = data_op.input_data(tensor_b, "tensor_b")
      act = custom_operators.my_custom_operator(act1, act2, "my_custom_operator")
      expected_output = Tensor(data_layout=types_pb2.N, tensor_data=np.array([3,5,7,9,11,13,15,17,19,21], dtype=np.float32))
    self.runAndValidate(graph=graph, expected_output=expected_output.tensor_data)

  def test_my_custom_operator_smv(self):
    self.backend = "SMV"
    with Graph(name=self.graph_name, backend=self.backend) as graph:
      tensor_a = Tensor(data_layout=types_pb2.NC, tensor_data=np.array([1,2,3,4,5,6,7,8,9,10], dtype=np.float16)) # the tensor data type should match the dtype accepted my the custom operator
      tensor_b = Tensor(data_layout=types_pb2.NC, tensor_data=np.array([2,3,4,5,6,7,8,9,10,11], dtype=np.float16))
      act1 = data_op.input_data(tensor_a, "tensor_a")
      act2 = data_op.input_data(tensor_b, "tensor_b")
      act = custom_operators.my_custom_operator(act1, act2, "my_custom_operator")
      expected_output = Tensor(data_layout=types_pb2.N, tensor_data=np.array([3,5,7,9,11,13,15,17,19,21], dtype=np.float16))
    self.runAndValidate(graph=graph, expected_output=expected_output.tensor_data)

if __name__ == "__main__":
  unittest.main()