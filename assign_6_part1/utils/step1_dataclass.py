import dataclasses

@dataclasses.dataclass
class CacheMetaParameter:
    """Meta parameters for cache configurations."""
    
    replacement_policy: str | list[str]
    prefetcher_type:str | list[str]
    
    l1_cache_sample_seed: int | list[int]
    l2_cache_sample_seed: int | list[int]
    l3_cache_sample_seed: int | list[int]
    
    experiment_index: int = -1
    
@dataclasses.dataclass
class ExperimentMetaParameter:
    """Meta parameters for experiment configurations."""
    
    # cache params
    replacement_policy: str | list[str]
    prefetcher_type:str | list[str]
    l1_cache_sample_seed: int | list[int]
    l2_cache_sample_seed: int | list[int]
    l3_cache_sample_seed: int | list[int]
    
    # processor params
    big_core_width: int | list[int]
    big_core_rob_size: int | list[int]
    big_core_num_int_regs: int | list[int]
    big_core_num_fp_regs: int | list[int]
    small_core_width: int | list[int]
    small_core_rob_size: int | list[int]
    small_core_num_int_regs: int | list[int]
    small_core_num_fp_regs: int | list[int]
    big_core_num: int | list[int]
    small_core_num: int | list[int]
    
    # workload params
    matsize: int | list[int]
    
    # experiment index
    experiment_index: int = -1