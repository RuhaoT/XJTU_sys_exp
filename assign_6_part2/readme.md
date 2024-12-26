# 第二部分：SMAUG模拟框架的自定义算子实现

## 复现步骤

0. 按照实验报告搭建环境
1. 复制更改的文件到对应的原有路径
2. 重新编译SMAUG模拟框架
3. 运行`smaug/experiments/sims/smv/tests/minerva/run.sh`，查看输出
4. 参考实验部分一编译负载并运行对照测试

## 实现/更改的文件路径

请结合实验报告查看

### SMAUG模拟框架

搭建测试/整理数据：

`smaug/experiments/models/minerva/minerva_network.py`
`smaug/experiments/sims/smv/tests/minerva/`

注册算子并修改后端：

`smaug/smaug/core/backend.cpp`
`smaug/smaug/core/backend.h`
`smaug/smaug/core/network_builder.cpp`
`smaug/smaug/core/node.proto`
`smaug/smaug/core/types.proto`
`smaug/smaug/core/smaug_test.h`
`smaug/smaug/__init__.py`
`smaug/make/Makefile.common`
`smaug/make/kernel_functions.txt`

禁用内积算子输出维度过大检查：

`smaug/smaug/operators/smv/smv_inner_product_op.cpp`

自定义算子实现后端：

`smaug/smaug/operators/custom_operators_test.cpp`
`smaug/smaug/operators/custom_operators.h`
`smaug/smaug/operators/custom_operators.cpp`

自定义算子Python接口：

`smaug/smaug/python/ops/custom_operators_test.py`
`smaug/smaug/python/ops/custom_operators.py`
`smaug/smaug/python/smaug_test.py`

修复了源码中的一个bug：

`smaug/smaug/python/tensor_utils.py`


### Gem5-Aladdin

更改默认ASIC频率：

`gem5-aladdin/configs/aladdin/aladdin_se.py`

### 对照测试

与实验部分一基本相同。

更改测试负载/参数：

`XJTU_sys_exp/assign_6_part1/workload/matmul/mm.cpp`
`XJTU_sys_exp/assign_6_part1/step1_hybrid_cpu.py`