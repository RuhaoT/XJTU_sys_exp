"""This is the gem5-based experiment executor for the hybrid CPU experiment."""

import dataclasses
import csv
import argparse
from pathlib import Path

from m5.objects import ReplacementPolicies, Prefetcher, Process
from m5.util import addToPath
from gem5.components.memory import SingleChannelDDR3_1600
from gem5.components.boards.simple_board import SimpleBoard
from gem5.resources.resource import BinaryResource
from gem5.simulate.simulator import Simulator

from components import cache_hierarchies, processors
import utils.step1_dataclass as step1_dataclass

@dataclasses.dataclass
class ExperimentParams:
    """Parameters for a single experiment."""
    
    cache_config: cache_hierarchies.O3HybridCPUCacheHierarchyConfig
    processor_config: processors.O3HybridProcessorConfig
    
    mat_size: int = 0
    
    experiment_index: int = -1

def get_replacement_policy(replacement_policy: str) -> ReplacementPolicies:
    """Get the replacement policy object based on the string."""
    
    if replacement_policy == "LRURP":
        return ReplacementPolicies.LRURP()
    elif replacement_policy == "LFURP":
        return ReplacementPolicies.LFURP()
    elif replacement_policy == "SecondChanceRP":
        return ReplacementPolicies.SecondChanceRP()
    else:
        raise ValueError("Invalid replacement policy.")

def get_prefetcher(prefetcher_type: str, degree: int) -> Prefetcher:
    """Get the prefetcher object based on the string."""
    
    if prefetcher_type == "Tagged":
        return Prefetcher.TaggedPrefetcher(degree=degree)
    elif prefetcher_type == "Stride":
        return Prefetcher.StridePrefetcher(degree=degree)
    elif prefetcher_type == "Signature":
        return Prefetcher.SignaturePathPrefetcher()
    elif prefetcher_type == "ISB":
        return Prefetcher.IrregularStreamBufferPrefetcher(degree=degree)
    else:
        raise ValueError("Invalid prefetcher type.")

def parameterization(meta_params: step1_dataclass.ExperimentMetaParameter) -> ExperimentParams:
    """Generate all possible cache configurations based on the meta parameters."""
    
    # generate experiment cache configuration based on the meta parameter
    replacement_policy = get_replacement_policy(meta_params.replacement_policy)
    
    new_cache_config = cache_hierarchies.O3HybridCPUCacheHierarchyConfig(
        big_core_cache_config=cache_hierarchies.O3CPUCacheHierarchyCacheConfig(
            l1d=cache_hierarchies.SingleCacheLevelConfig(
                size=str(meta_params.l1_cache_sample_seed * 2) + "kB",
                assoc=8,
                replacement_policy=replacement_policy,
                prefetcher=get_prefetcher(meta_params.prefetcher_type, degree=meta_params.l1_cache_sample_seed * 2),
            ),
            l1i=cache_hierarchies.SingleCacheLevelConfig(
                size=str(meta_params.l1_cache_sample_seed * 2) + "kB",
                assoc=8,
                replacement_policy=replacement_policy,
                prefetcher=get_prefetcher(meta_params.prefetcher_type, degree=meta_params.l1_cache_sample_seed * 2),
            ),
            l2=cache_hierarchies.SingleCacheLevelConfig(
                size=str(meta_params.l2_cache_sample_seed * 16) + "kB",
                assoc=16,
                replacement_policy=replacement_policy,
                prefetcher=get_prefetcher(meta_params.prefetcher_type, degree=meta_params.l2_cache_sample_seed * 4),
            ),
        ),
        big_core_type_id=0,
        little_core_cache_config=cache_hierarchies.O3CPUCacheHierarchyCacheConfig(
            l1d=cache_hierarchies.SingleCacheLevelConfig(
                size=str(meta_params.l1_cache_sample_seed * 1) + "kB",
                assoc=4,
                replacement_policy=replacement_policy,
                prefetcher=get_prefetcher(meta_params.prefetcher_type, degree=meta_params.l1_cache_sample_seed),
            ),
            l1i=cache_hierarchies.SingleCacheLevelConfig(
                size=str(meta_params.l1_cache_sample_seed * 1) + "kB",
                assoc=4,
                replacement_policy=replacement_policy,
                prefetcher=get_prefetcher(meta_params.prefetcher_type, degree=meta_params.l1_cache_sample_seed),
            ),
            l2=cache_hierarchies.SingleCacheLevelConfig(
                size=str(meta_params.l2_cache_sample_seed * 8) + "kB",
                assoc=8,
                replacement_policy=replacement_policy,
                prefetcher=get_prefetcher(meta_params.prefetcher_type, degree=meta_params.l2_cache_sample_seed * 2),
            ),
        ),
        little_core_type_id=1,
        l3=cache_hierarchies.SingleCacheLevelConfig(
            size=str(meta_params.l3_cache_sample_seed * 64) + "kB",
            assoc=32,
            replacement_policy=replacement_policy,
            prefetcher=get_prefetcher(meta_params.prefetcher_type, degree=meta_params.l3_cache_sample_seed * 4),
        ),   
    )
    
    new_processor_config = processors.O3HybridProcessorConfig(
        big_core=processors.O3CoreConfig(
            width=meta_params.big_core_width,
            rob_size=meta_params.big_core_rob_size,
            num_int_regs=meta_params.big_core_num_int_regs,
            num_fp_regs=meta_params.big_core_num_fp_regs,
            core_type_id=0,
        ),
        big_core_num=meta_params.big_core_num,
        little_core=processors.O3CoreConfig(
            width=meta_params.small_core_width,
            rob_size=meta_params.small_core_rob_size,
            num_int_regs=meta_params.small_core_num_int_regs,
            num_fp_regs=meta_params.small_core_num_fp_regs,
            core_type_id=1,
        ),
        little_core_num=meta_params.small_core_num,
    )
    
    new_experiment = ExperimentParams(
        cache_config=new_cache_config,
        processor_config=new_processor_config,
        mat_size=meta_params.matsize,
    )
    
    return new_experiment

if __name__ == "__m5_main__":
    parser = argparse.ArgumentParser(description="Hybrid CPU Experiment Executor")
    parser.add_argument("--param_file", type=str, help="The path to the experiment parameter file.")
    parser.add_argument("--target_index", type=int, help="The index of the target experiment.")
    parser.add_argument("--workload", type=str, help="The workload to run.")
    
    args = parser.parse_args()
    
    # step 1: load the experiment parameters
    # add target file to path
    addToPath(args.param_file)
    with open(args.param_file, "r") as f:
        # find the target experiment
        reader = csv.DictReader(f)
        target_experiment = None
        for row in reader:
            if int(row["experiment_index"]) == args.target_index:
                target_experiment = row
                break
        # convert the target experiment to dataclass
        target_experiment = step1_dataclass.ExperimentMetaParameter(
            replacement_policy=target_experiment["replacement_policy"],
            prefetcher_type=target_experiment["prefetcher_type"],
            l1_cache_sample_seed=int(target_experiment["l1_cache_sample_seed"]),
            l2_cache_sample_seed=int(target_experiment["l2_cache_sample_seed"]),
            l3_cache_sample_seed=int(target_experiment["l3_cache_sample_seed"]),
            big_core_width=int(target_experiment["big_core_width"]),
            big_core_rob_size=int(target_experiment["big_core_rob_size"]),
            big_core_num_int_regs=int(target_experiment["big_core_num_int_regs"]),
            big_core_num_fp_regs=int(target_experiment["big_core_num_fp_regs"]),
            small_core_width=int(target_experiment["small_core_width"]),
            small_core_rob_size=int(target_experiment["small_core_rob_size"]),
            small_core_num_int_regs=int(target_experiment["small_core_num_int_regs"]),
            small_core_num_fp_regs=int(target_experiment["small_core_num_fp_regs"]),
            big_core_num=int(target_experiment["big_core_num"]),
            small_core_num=int(target_experiment["small_core_num"]),
            matsize=int(target_experiment["matsize"]),
        )
        # convert the target experiment to ExperimentParams
        target_experiment = parameterization(target_experiment)
    
    # step 2: build experiment architecture
    processor = processors.O3CPU(target_experiment.processor_config)
    cache_hierarchy = cache_hierarchies.O3HybridCPUCacheHierarchy(target_experiment.cache_config)
    
    board = SimpleBoard(
        processor=processor,
        cache_hierarchy=cache_hierarchy,
        memory=SingleChannelDDR3_1600("32MiB"),
        clk_freq="3GHz",
    )

    # step 3: set the workload
    binary_path = Path(args.workload)
    addToPath(binary_path.as_posix())
    
    # set the workload
    board.set_se_binary_workload(
        binary = BinaryResource(
            local_path=binary_path.as_posix(),
        ),
    )

    process = []
    # set the process to each core
    cores = board.get_processor().get_cores()
    for index in range(target_experiment.processor_config.big_core_num + target_experiment.processor_config.little_core_num):
        process.append(Process())
        process[-1].pid = 1000 + index # avoid pid conflict
        process[-1].cmd = [binary_path.as_posix(), str(target_experiment.mat_size)]
        cores[index].core.workload = process[-1]
        
    
    Simulator(board).run()
    