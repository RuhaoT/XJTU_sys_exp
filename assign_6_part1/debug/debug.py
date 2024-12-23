import m5 
from m5.objects import *
from m5.util import addToPath

from gem5.components.boards.simple_board import SimpleBoard
from gem5.components.boards.riscv_board import RiscvBoard
from gem5.components.memory import SingleChannelDDR3_1600
from gem5.resources.resource import obtain_resource
from gem5.components.cachehierarchies.classic.no_cache import NoCache
from gem5.components.processors.simple_processor import SimpleProcessor
from gem5.components.processors.cpu_types import CPUTypes
from gem5.isas import ISA
from gem5.resources.resource import BinaryResource
from pathlib import Path


from gem5.simulate.simulator import Simulator

import sys
# append current file dir
curr_dir = Path(__file__).resolve().parent.parent.as_posix()
sys.path.append(curr_dir)
from components import cache_hierarchies, processors

test_config_single_cache_level = cache_hierarchies.SingleCacheLevelConfig(
    size="64kB",
    assoc=8,
    replacement_policy=ReplacementPolicies.LRURP(),
    prefetcher=Prefetcher.TaggedPrefetcher(degree=4),
)

test_config_o3_cpu_cache_hierarchy = cache_hierarchies.O3CPUCacheHierarchyCacheConfig(
    l1d=test_config_single_cache_level,
    l1i=test_config_single_cache_level,
    l2=test_config_single_cache_level,
)

test_config_o3_hybrid_cpu_cache_hierarchy = cache_hierarchies.O3HybridCPUCacheHierarchyConfig(
    big_core_cache_config=test_config_o3_cpu_cache_hierarchy,
    big_core_type_id=0,
    little_core_cache_config=test_config_o3_cpu_cache_hierarchy,
    little_core_type_id=1,
    l3=test_config_single_cache_level,
)

test_cache_hierarchy = cache_hierarchies.O3HybridCPUCacheHierarchy(
    test_config_o3_hybrid_cpu_cache_hierarchy,
)

test_config_o3_core = processors.O3CoreConfig(
    width=8,
    rob_size=192,
    num_int_regs=256,
    num_fp_regs=256,
    core_type_id=1,
)

test_config_o3_hybrid_processor = processors.O3HybridProcessorConfig(
    big_core=test_config_o3_core,
    big_core_num=1,
    little_core=test_config_o3_core,
    little_core_num=1,
)

test_processor = processors.O3CPU(test_config_o3_hybrid_processor)

processor = SimpleProcessor(
    cpu_type=CPUTypes.TIMING, isa=ISA.RISCV, num_cores=1
)


board = SimpleBoard(
    processor=test_processor,
    cache_hierarchy=test_cache_hierarchy,
    memory=SingleChannelDDR3_1600("32MiB"),
    clk_freq="3GHz",
)

binary_path = Path("workload/matmul/mm-ijk-gem5")
addToPath(binary_path.as_posix())

board.set_se_binary_workload(
    binary = BinaryResource(
        local_path=binary_path.as_posix(),
    ),
    arguments = [10],
)
sim = Simulator(board)
sim.run()