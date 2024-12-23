"""The custom cache hierarchy for experiment part 1."""

import dataclasses

from gem5.components.boards.abstract_board import AbstractBoard
from gem5.components.cachehierarchies.classic.abstract_classic_cache_hierarchy import (
    AbstractClassicCacheHierarchy,
)

from gem5.components.cachehierarchies.classic.caches.l1dcache import L1DCache
from gem5.components.cachehierarchies.classic.caches.l1icache import L1ICache
from gem5.components.cachehierarchies.classic.caches.l2cache import L2Cache
from gem5.components.cachehierarchies.classic.caches.mmu_cache import MMUCache

from gem5.isas import ISA

from m5.objects import (
    BadAddr,
    Cache,
    L2XBar,
    SystemXBar,
    SubSystem,
    ReplacementPolicies,
    Prefetcher,
)


@dataclasses.dataclass
class SingleCacheLevelConfig:
    """Configuration for a single cache level."""

    size: str
    assoc: int
    replacement_policy: ReplacementPolicies.BaseReplacementPolicy
    prefetcher: Prefetcher.StridePrefetcher(degree=8, latency=1, prefetch_on_access=True)


@dataclasses.dataclass
class O3CPUCacheHierarchyCacheConfig:
    """Configuration for the caches in the cache hierarchy for a specific core type."""

    l1d: SingleCacheLevelConfig
    l1i: SingleCacheLevelConfig
    l2: SingleCacheLevelConfig
    # no l3 because it is shared


@dataclasses.dataclass
class O3HybridCPUCacheHierarchyConfig:
    """Configuration for the cache hierarchy for a hybrid O3 CPU."""

    big_core_cache_config: O3CPUCacheHierarchyCacheConfig
    big_core_type_id: int
    little_core_cache_config: O3CPUCacheHierarchyCacheConfig
    little_core_type_id: int
    # l3 cache is here
    l3: SingleCacheLevelConfig


class O3HybridCPUCacheHierarchy(AbstractClassicCacheHierarchy):

    def __init__(
        self,
        cache_hierarchy_config: O3HybridCPUCacheHierarchyConfig,
    ):
        AbstractClassicCacheHierarchy.__init__(self)

        # Save the sizes to use later. We have to use leading underscores
        # because the SimObject (SubSystem) does not have these attributes as
        # parameters.
        self._cache_hierarchy_config = cache_hierarchy_config

        # Use a high-bandwidth system crossbar.
        self.membus = SystemXBar(width=64)
        # For FS mode
        self.membus.badaddr_responder = BadAddr()
        self.membus.default = self.membus.badaddr_responder.pio

        # We can't create the caches yet, because we don't know how many cores there are.

    # To connect the memory system to the caches
    def get_mem_side_port(self):
        return self.membus.mem_side_ports

    # For FS mode. This is a coherent port.
    def get_cpu_side_port(self):
        return self.membus.cpu_side_ports

    # This is where the bulk of the work happens.
    # The board calls this function after it has created the processor and
    # memory system. The cache hierarchy is responsible for connecting things.
    def incorporate_cache(self, board):
        # Connect the system port to the memory system.
        board.connect_system_port(self.membus.cpu_side_ports)

        # Connect the memory system to the memory port on the board.
        for _, port in board.get_memory().get_mem_ports():
            self.membus.mem_side_ports = port

        # Create an L3 crossbar
        self.l3_bus = L2XBar()

        self.clusters = [
            self._create_core_cluster(
                core, self.l3_bus, board.get_processor().get_isa()
            )
            for core in board.get_processor().get_cores()
        ]

        # Create the shared L3 cache
        l3_size = self._cache_hierarchy_config.l3.size
        l3_assoc = self._cache_hierarchy_config.l3.assoc
        self.l3_cache = L3Cache(
            size=l3_size,
            assoc=l3_assoc,
        )
        self.l3_cache.replacement_policy = self._cache_hierarchy_config.l3.replacement_policy
        self.l3_cache.prefetcher = self._cache_hierarchy_config.l3.prefetcher

        # Connect the L3 cache to the system crossbar and L3 crossbar
        self.l3_cache.mem_side = self.membus.cpu_side_ports
        self.l3_cache.cpu_side = self.l3_bus.mem_side_ports

        if board.has_coherent_io():
            self._setup_io_cache(board)

    def _create_core_cluster(self, core, l3_bus, isa):
        """
        Create a core cluster with the given core.
        In this experiment setting each cluster has only one core.
        So big cores and little cores are treated the same.
        """

        # get core type ID
        core_type_id = core.get_core_type_id()
        # choose the cache config based on the core type ID
        if core_type_id == self._cache_hierarchy_config.big_core_type_id:
            cache_config = self._cache_hierarchy_config.big_core_cache_config
        elif core_type_id == self._cache_hierarchy_config.little_core_type_id:
            cache_config = self._cache_hierarchy_config.little_core_cache_config
        else:
            raise ValueError(f"Unknown core type ID: {core_type_id}")

        l1d_size = cache_config.l1d.size
        l1d_assoc = cache_config.l1d.assoc
        l1d_replacement_policy = cache_config.l1d.replacement_policy
        l1d_prefetcher = cache_config.l1d.prefetcher

        l1i_size = cache_config.l1i.size
        l1i_assoc = cache_config.l1i.assoc
        l1i_replacement_policy = cache_config.l1i.replacement_policy
        l1i_prefetcher = cache_config.l1i.prefetcher

        l2_size = cache_config.l2.size
        l2_assoc = cache_config.l2.assoc
        l2_replacement_policy = cache_config.l2.replacement_policy
        l2_prefetcher = cache_config.l2.prefetcher

        cluster = SubSystem()
        cluster.l1dcache = L1DCache(
            size=l1d_size,
            assoc=l1d_assoc,
            PrefetcherCls=l1d_prefetcher,
        )
        cluster.l1dcache.replacement_policy = l1d_replacement_policy
        cluster.l1icache = L1ICache(
            size=l1i_size,
            assoc=l1i_assoc,
            PrefetcherCls=l1i_prefetcher,
        )
        cluster.l1icache.replacement_policy = l1i_replacement_policy
        cluster.l2cache = L2Cache(
            size=l2_size,
            assoc=l2_assoc,
            PrefetcherCls=l2_prefetcher,
        )
        cluster.l2cache.replacement_policy = l2_replacement_policy

        cluster.iptw_cache = MMUCache(size="8KiB", writeback_clean=False)
        cluster.dptw_cache = MMUCache(size="8KiB", writeback_clean=False)

        cluster.l2_bus = L2XBar()

        # Connect the core to the caches
        core.connect_icache(cluster.l1icache.cpu_side)
        core.connect_dcache(cluster.l1dcache.cpu_side)
        core.connect_walker_ports(
            cluster.iptw_cache.cpu_side, cluster.dptw_cache.cpu_side
        )

        # Connect the caches to the L2 bus
        cluster.l1dcache.mem_side = cluster.l2_bus.cpu_side_ports
        cluster.l1icache.mem_side = cluster.l2_bus.cpu_side_ports
        cluster.iptw_cache.mem_side = cluster.l2_bus.cpu_side_ports
        cluster.dptw_cache.mem_side = cluster.l2_bus.cpu_side_ports

        cluster.l2cache.cpu_side = cluster.l2_bus.mem_side_ports

        cluster.l2cache.mem_side = l3_bus.cpu_side_ports

        if isa == ISA.X86:
            int_req_port = self.membus.mem_side_ports
            int_resp_port = self.membus.cpu_side_ports
            core.connect_interrupt(int_req_port, int_resp_port)
        else:
            core.connect_interrupt()

        return cluster

    def _setup_io_cache(self, board: AbstractBoard) -> None:
        """Create a cache for coherent I/O connections"""
        self.iocache = Cache(
            assoc=8,
            tag_latency=50,
            data_latency=50,
            response_latency=50,
            mshrs=20,
            size="1kB",
            tgts_per_mshr=12,
            addr_ranges=board.mem_ranges,
        )
        self.iocache.mem_side = self.membus.cpu_side_ports
        self.iocache.cpu_side = board.get_mem_side_coherent_io_port()


class L3Cache(Cache):
    def __init__(self, size, assoc):
        super().__init__()
        self.size = size
        self.assoc = assoc
        self.tag_latency = 20
        self.data_latency = 20
        self.response_latency = 1
        self.mshrs = 20
        self.tgts_per_mshr = 12
        self.writeback_clean = False
        self.clusivity = "mostly_incl"
