import dataclasses

from gem5.isas import ISA
from gem5.components.processors.base_cpu_core import BaseCPUCore
from gem5.components.processors.base_cpu_processor import BaseCPUProcessor

from m5.objects import RiscvO3CPU
from m5.objects.FuncUnitConfig import *
from m5.objects.BranchPredictor import (
    TournamentBP,
    MultiperspectivePerceptronTAGE64KB,
)

@dataclasses.dataclass
class O3CoreConfig:
    width: int
    rob_size: int
    num_int_regs: int
    num_fp_regs: int
    core_type_id: int
    
@dataclasses.dataclass
class O3HybridProcessorConfig:
    big_core: O3CoreConfig
    big_core_num: int
    
    little_core: O3CoreConfig
    little_core_num: int


# O3CPUCore extends RiscvO3CPU. RiscvO3CPU is one of gem5's internal models
# the implements an out of order pipeline. Please refer to
#   https://www.gem5.org/documentation/general_docs/cpu_models/O3CPU
# to learn more about O3CPU.


class O3CPUCore(RiscvO3CPU):
    def __init__(self, width, rob_size, num_int_regs, num_fp_regs):
        """
        :param width: sets the width of fetch, decode, rename, issue, wb, and
        commit stages.
        :param rob_size: determine the number of entries in the reorder buffer.
        :param num_int_regs: determines the size of the integer register file.
        :param num_int_regs: determines the size of the vector/floating point
        register file.
        """
        super().__init__()
        self.fetchWidth = width
        self.decodeWidth = width
        self.renameWidth = width
        self.issueWidth = width
        self.wbWidth = width
        self.commitWidth = width

        self.numROBEntries = rob_size

        self.numPhysIntRegs = num_int_regs
        self.numPhysFloatRegs = num_fp_regs

        self.branchPred = TournamentBP()

        self.LQEntries = 128
        self.SQEntries = 128


# Along with BaseCPUCore, CPUStdCore wraps CPUCore to a core compatible
# with gem5's standard library. Please refer to
#   gem5/src/python/gem5/components/processors/base_cpu_core.py
# to learn more about BaseCPUCore.


class O3CPUStdCore(BaseCPUCore):
    def __init__(self, core_config: O3CoreConfig):
        """
        :param width: sets the width of fetch, decode, raname, issue, wb, and
        commit stages.
        :param rob_size: determine the number of entries in the reorder buffer.
        :param num_int_regs: determines the size of the integer register file.
        :param num_int_regs: determines the size of the vector/floating point
        register file.
        """
        
        width = core_config.width
        rob_size = core_config.rob_size
        num_int_regs = core_config.num_int_regs
        num_fp_regs = core_config.num_fp_regs
        
        self._core_type_id = core_config.core_type_id
        
        core = O3CPUCore(width, rob_size, num_int_regs, num_fp_regs)
        super().__init__(core, ISA.RISCV)
    
    def get_core_type_id(self):
        """Returns the core type ID. Used for customizing caches."""
        return self._core_type_id


# O3CPU along with BaseCPUProcessor wraps CPUCore to a processor
# compatible with gem5's standard library. Please refer to
#   gem5/src/python/gem5/components/processors/base_cpu_processor.py
# to learn more about BaseCPUProcessor.


class O3CPU(BaseCPUProcessor):
    def __init__(self, processor_configs: O3HybridProcessorConfig):
        """
        :param width: sets the width of fetch, decode, raname, issue, wb, and
        commit stages.
        :param rob_size: determine the number of entries in the reorder buffer.
        :param num_int_regs: determines the size of the integer register file.
        :param num_int_regs: determines the size of the vector/floating point
        register file.
        """
        big_cores = [
            O3CPUStdCore(processor_configs.big_core)
            for _ in range(processor_configs.big_core_num)
        ]
        
        little_cores = [
            O3CPUStdCore(processor_configs.little_core)
            for _ in range(processor_configs.little_core_num)
        ]
        
        cores = big_cores + little_cores
        
        super().__init__(cores)
        # self._width = width
        # self._rob_size = rob_size
        # self._num_int_regs = num_int_regs
        # self._num_fp_regs = num_fp_regs

    # def get_area_score(self):
    #     """
    #     :returns the area score of a pipeline using its parameters width,
    #     rob_size, num_int_regs, and num_fp_regs.
    #     """
    #     score = (
    #         self._width
    #         * (2 * self._rob_size + self._num_int_regs + self._num_fp_regs)
    #         + 4 * self._width
    #         + 2 * self._rob_size
    #         + self._num_int_regs
    #         + self._num_fp_regs
    #     )
    #     return score