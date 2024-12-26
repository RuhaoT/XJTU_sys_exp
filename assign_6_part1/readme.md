# 第一部分：高级多核异构处理器及缓存结构优化

## 复现步骤

0. 按照实验报告搭建环境，编译Gem5等。
1. 在`workload/compile_dependencies.sh`中更新绝对路径，并运行以编译m5库。
2. 在`workload/matmul/Makefile`中更改RISCV编译器路径，编译负载。
3. 在`step1_hybrid_cpu.py`中更新绝对路径，运行测试。
4. 在`results/`中查看结果。

如果需要实现Typings补全，更新`gem5-stubgen.py`中的绝对路径并运行。

如果需要复现实验报告中的图表，使用`results/step1_debug/analysis.ipynb`，请安装其中的额外Python库。