from __future__ import annotations

import psutil
import os
import json
import platform
import warnings

from typing import Iterator, Dict, Any, Optional, List

from subprocess import Popen, PIPE
from distutils import spawn
from dataclasses import dataclass

from ..constants import *
from .string_util import human_size
from .misc_util import chunk_iterable, timed_lru_cache

from dbgpu import GPUDatabase, GPUSpecification

__all__ = ["GPU", "CPU", "RAM", "MachineCapability"]

@dataclass
class GPU:
    """
    This class holds details about the GPU as returned by the appropriate subprocess.
    """

    id: str
    uuid: str
    load: float
    perf: float
    memory_total: float
    memory_used: float
    temp: float
    driver: str
    name: str

    @property
    def memory_util(self) -> float:
        """
        Calculate utilization in the range [0, 1]
        """
        return float(self.memory_used) / float(self.memory_total)

    @property
    def memory_free(self) -> float:
        """
        Calculate free bytes
        """
        return self.memory_total - self.memory_used

    @property
    def specification(self) -> GPUSpecification:
        """
        Get the GPU specification from the database.
        """
        if not hasattr(self, "_specification"):
            database = GPUDatabase.default()
            try:
                self._specification = database[self.name]
            except KeyError:
                self._specification = database.search(self.name) # let exception propagate from this
        return self._specification

    @classmethod
    def get_process_kwargs(cls) -> Dict[str, Any]:
        """
        Gets keyword arguments to pass into the Popen call.
        """
        process_kwargs: Dict[str, Any] = {"stdout": PIPE, "stderr": PIPE}
        if platform.system() == "Windows":
            from subprocess import CREATE_NO_WINDOW  # type: ignore
            process_kwargs["creationflags"] = CREATE_NO_WINDOW
        return process_kwargs

    @classmethod
    def get_nvidia_gpus(cls, executable: str) -> Iterator[GPU]:
        """
        Executes `nvidia-smi` and parses output.
        """
        p = Popen(
            [
                executable,
                "--query-gpu=index,uuid,pstate,utilization.gpu,memory.total,memory.used,driver_version,name,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            **cls.get_process_kwargs(),
        )
        stdout, stderr = p.communicate()
        output = stdout.decode("UTF-8")
        lines = [
            line for line in [line.strip() for line in output.split(os.linesep)] if line
        ]
        for line in lines:
            (id, uuid, pstate, load, memory_total, memory_used, driver, name, temp) = (
                line.split(",")
            )

            yield GPU(
                id=id,
                uuid=uuid.strip(),
                memory_total=int(memory_total) * 1e6,
                memory_used=int(memory_used) * 1e6,
                load=float(load) / 100.0,
                temp=float(temp),
                perf=1.0-(float(pstate.strip()[1:]) / 12.0),  # 0-12, 0 is max performance (invert for perf)
                driver=driver.strip(),
                name=name.strip(),
            )

    @staticmethod
    def get_amd_gpus(executable: str) -> Iterator[GPU]:
        """
        Executes `rocm-smi` and parses output.
        TODO: This has not been tested sufficiently.
        """
        p = Popen(
            [
                executable,
                "--alldevices",
                "-tu",
                "--showproductname",
                "--showdriverversion",
                "--showuniqueid",
                "--showmeminfo",
                "vram",
                "--json",
            ],
            **GPU.get_process_kwargs(),
        )
        stdout, stderr = p.communicate()
        output = stdout.decode("UTF-8")
        result = json.loads(output)
        for key in result:
            if key == "system":
                continue
            gpu_dict = result[key]
            yield GPU(
                id=key,
                uuid=gpu_dict["Unique ID"],
                memory_total=float(gpu_dict["VRAM Total Memory (B)"]) / 1000000.0,
                memory_used=float(gpu_dict["VRAM Total Used Memory (B)"]) / 1000000.0,
                temp=float(gpu_dict["Temperature (Sensor junction) (C)"]),
                load=float(gpu_dict["GPU use (%)"]) / 100.0,
                perf=1.0, # rocm-smi does not provide performance data
                driver=result["system"]["Driver version"],
                name=gpu_dict["Card series"],
            )

    @classmethod
    def get_gpus(cls, fail_on_error: bool = True) -> Iterator[GPU]:
        """
        Gets the appropriate binary and executes it
        """
        if platform.system() == "Darwin":
            # TODO: Implement for MacOS
            return

        nvidia_smi = spawn.find_executable("nvidia-smi")
        rocm_smi = spawn.find_executable("rocm-smi")

        try:
            if rocm_smi is not None:
                gpus = cls.get_amd_gpus(rocm_smi)
            elif nvidia_smi is not None:
                gpus = cls.get_nvidia_gpus(nvidia_smi)
            elif fail_on_error:
                raise ValueError("No GPU monitoring tool found.")
            else:
                return
            for gpu in gpus:
                yield gpu
        except Exception as e:
            if fail_on_error:
                raise e
            return

    @timed_lru_cache(ttl=1.0)
    def score_ratio(
        self,
        gpu_precision: str = "half",
        required_cuda_version: Optional[str] = None,
        required_gpu_memory_gb: Optional[float] = None,
        max_gpu_memory_bandwidth: Optional[float] = None,
        max_gpu_performance: Optional[float] = None,
        use_relative_performance: bool = True,
    ) -> float:
        """
        Scores the GPUs current ability to perform.
        Potentially uses the required CUDA version to determine compatibility,
        and the gpu precision to determine which performance metric to use.
        """
        # Check if the GPU is compatible with the required CUDA version
        if required_cuda_version is not None:
            # If the GPU does not have a CUDA version, assume it is not compatible
            if not self.specification.cuda_major_version:
                return 0.0

            # Split required version into major and minor parts
            version_parts = required_cuda_version.split(".")
            if len(version_parts) >= 2:
                mj, mn = version_parts[:2]
            else:
                mj, mn = version_parts[0], "0"

            major, minor = int(mj), int(mn)

            # Gather the GPU's CUDA version
            spec_major = self.specification.cuda_major_version
            spec_minor = self.specification.cuda_minor_version
            if not spec_minor:
                spec_minor = 0

            # Compare the versions
            if (
                (major < spec_major) or 
                (major == spec_major and minor < spec_minor)
            ):
                return 0.0

        # Check if the GPU has enough memory
        if required_gpu_memory_gb is not None and self.memory_free < required_gpu_memory_gb * 1e9:
            return 0.0

        # Get performance data
        performance: Optional[float] = None
        if gpu_precision in ["half", "float16", "fp16"]:
            default_max_gpu_performance = 653700.0 # 653.7 TFLOPS, current highest in GPUDB
            performance = self.specification.half_float_performance_gflop_s
            if not performance:
                performance = self.specification.single_float_performance_gflop_s
        elif gpu_precision in ["double", "float64", "fp64"]:
            default_max_gpu_performance = 81720.0 # 81.72 TFLOPS, current highest in GPUDB
            performance = self.specification.double_float_performance_gflop_s
        else:
            default_max_gpu_performance = 93240.0 # 93.24 TFLOPS, current highest in GPUDB
            if gpu_precision is not None and gpu_precision not in ["single", "float", "float32", "fp32"]:
                warnings.warn(f"Unknown gpu precision {gpu_precision}, defaulting to single/float/fp32")
            performance = self.specification.single_float_performance_gflop_s
        if not max_gpu_performance:
            max_gpu_performance = default_max_gpu_performance
        if not performance:
            performance = max_gpu_performance / 100.0 # Assume the GPU is 1% as fast as the best GPU if no data is available

        # Get accurate memory bandwidth data
        if max_gpu_memory_bandwidth is None:
            max_gpu_memory_bandwidth = 10300.0 # 10.3 TB/s, current highest in GPUDB

        # Calculate ratios
        performance_ratio = 1.0 if not use_relative_performance else performance / max_gpu_performance
        if use_relative_performance and max_gpu_memory_bandwidth:
            if self.specification.memory_bandwidth_gb_s:
                memory_bandwidth_ratio = self.specification.memory_bandwidth_gb_s / max_gpu_memory_bandwidth
            else:
                memory_bandwidth_ratio = max_gpu_memory_bandwidth / 100.0 # Assume the GPU is 1% as fast as the best GPU if no data is available
        else:
            memory_bandwidth_ratio = 1.0
        memory_utilization_ratio = 1.0 - self.memory_util
        load_ratio = 1.0 - self.load

        # Calculate weighted score ratio
        return float(
            (performance_ratio * GPU_PERFORMANCE_WEIGHT) +
            (memory_bandwidth_ratio * GPU_MEMORY_BANDWIDTH_WEIGHT) +
            (memory_utilization_ratio * GPU_MEMORY_UTILIZATION_WEIGHT) +
            (load_ratio * GPU_LOAD_WEIGHT)
        ) / GPU_TOTAL_WEIGHT

    def tabulate(self, tablefmt: str = "plain") -> str:
        """
        Returns a tabular representation of the GPU.
        """
        import tabulate

        try:
            spec_fields = self.specification.labeled_fields()
        except KeyError:
            spec_fields = []
            pass

        return tabulate.tabulate(
            [
                ("ID", self.id),
                ("Name", self.name),
                ("Driver", self.driver),
                ("Load", f"{self.load:.2%}"),
                ("Performance", f"{self.perf:.2%}"),
                ("Memory Total", human_size(self.memory_total)),
                ("Memory Available", human_size(self.memory_free)),
                ("Memory Used", human_size(self.memory_used)),
                ("Memory Utilization", f"{self.memory_util:.2%}"),
                ("Temperature", f"{self.temp:.2f}°C"),
            ] + [
                (name, value)
                for name, value in spec_fields
                if name not in ["Name", "Memory Size"]
            ],
            tablefmt=tablefmt,
        )

    def score(
        self,
        gpu_precision: str = "half",
        required_cuda_version: Optional[str] = None,
        required_gpu_memory_gb: Optional[float] = None,
        max_gpu_memory_bandwidth: Optional[float] = None,
        max_gpu_performance: Optional[float] = None,
        use_relative_performance: bool = True,
    ) -> int:
        """
        Scores the GPU's current ability to perform out of the maximum total score.
        """
        return int(
            self.score_ratio(
                gpu_precision=gpu_precision,
                required_cuda_version=required_cuda_version,
                required_gpu_memory_gb=required_gpu_memory_gb,
                max_gpu_memory_bandwidth=max_gpu_memory_bandwidth,
                max_gpu_performance=max_gpu_performance,
                use_relative_performance=use_relative_performance,
            ) * AVAILABILITY_SCORE_MAX
        )

    def __str__(self) -> str:
        """
        String representation of the GPU.
        """
        try:
            import tabulate

            return str(self.tabulate())
        except ImportError:
            return f"GPU(id={self.id}, uuid={self.uuid}, load={self.load}, perf={self.perf}, memory_total={self.memory_total}, memory_used={self.memory_used}, temp={self.temp}, driver={self.driver}, name={self.name})"

    def __hash__(self) -> int:
        """
        Hash the GPU object.
        """
        return hash(
            (
                self.id, self.uuid, self.load,
                self.perf, self.memory_total, self.memory_used,
                self.temp, self.driver, self.name
            )
        )

@dataclass
class CPU:
    """
    This class holds details about the CPU as returned by the appropriate subprocess.
    """

    load: float  # 0-1

    @classmethod
    def get_cpus(cls, interval: float = 0.1) -> Iterator[CPU]:
        """
        Gets all CPUs and their current load.
        """
        for cpu in psutil.cpu_percent(interval=interval, percpu=True):
            yield CPU(load=cpu / 100.0)

    def __hash__(self) -> int:
        """
        Hash the CPU object.
        """
        return hash(self.load)


@dataclass
class RAM:
    """
    This class holds details about the RAM as returned by the appropriate subprocess.
    """

    total: int
    available: int

    @property
    def used(self) -> int:
        """
        Calculate free bytes
        """
        return self.total - self.available

    @property
    def load(self) -> float:
        """
        Calculate utilization in the range [0, 1]
        """
        return 1.0 - (float(self.available) / float(self.total))

    @classmethod
    def get_ram(cls) -> RAM:
        """
        Gets the current RAM usage.
        """
        mem = psutil.virtual_memory()
        return RAM(total=mem.total, available=mem.available)

    def tabulate(self, tablefmt: str = "simple") -> str:
        """
        Returns a tabular representation of the RAM.
        """
        import tabulate

        return tabulate.tabulate(
            [
                ["Total", "Available", "Used", "Load"],
                [
                    human_size(self.total),
                    human_size(self.available),
                    human_size(self.used),
                    f"{self.load:.2%}",
                ],
            ],
            tablefmt=tablefmt,
            headers="firstrow",
        )

    def __str__(self) -> str:
        """
        String representation of the RAM.
        """
        try:
            import tabulate

            return str(self.tabulate())
        except ImportError:
            return f"RAM(total={self.total}, available={self.available})"

    def __hash__(self) -> int:
        """
        Hash the RAM object.
        """
        return hash((self.total, self.available))


@dataclass
class MachineCapability:
    """
    Holds all details about the machine's capabilities.
    """

    gpus: List[GPU]
    cpus: List[CPU]
    ram: RAM

    @classmethod
    def get_capability(cls, fail_on_gpu_error: bool = True) -> MachineCapability:
        """
        Gets all capabilities.
        """
        return MachineCapability(
            gpus=list(GPU.get_gpus(fail_on_error=fail_on_gpu_error)),
            cpus=list(CPU.get_cpus()),
            ram=RAM.get_ram(),
        )

    @timed_lru_cache(ttl=1.0)
    def get_optimal_gpu_id(
        self,
        use_gpu: bool = True,
        gpu_precision: str = "half",
        required_cuda_version: Optional[str] = None,
        required_memory_gb: Optional[float] = None,
        required_gpu_memory_gb: Optional[float] = None,
        max_gpu_memory_bandwidth: Optional[float] = None,
        max_gpu_performance: Optional[float] = None,
        use_relative_performance: bool = True,
    ) -> int:
        """
        Gets the optimal GPU ID based on the current machine's capabilities.
        """
        max_gpu_id: int = -1
        max_gpu_score: float = 0.0
        if use_gpu:
            for gpu in self.gpus:
                score_ratio = gpu.score_ratio(
                    gpu_precision=gpu_precision,
                    required_cuda_version=required_cuda_version,
                    required_gpu_memory_gb=required_gpu_memory_gb,
                    max_gpu_memory_bandwidth=max_gpu_memory_bandwidth,
                    max_gpu_performance=max_gpu_performance,
                    use_relative_performance=use_relative_performance,
                )
                if score_ratio > max_gpu_score:
                    max_gpu_id = int(gpu.id)
                    max_gpu_score = score_ratio
        if max_gpu_id == -1 and use_gpu:
            raise ValueError("No GPU found that meets the requirements.")
        return max_gpu_id

    @timed_lru_cache(ttl=1.0)
    def score_ratio(
        self,
        use_gpu: bool = True,
        gpu_precision: str = "half",
        required_cuda_version: Optional[str] = None,
        required_memory_gb: Optional[float] = None,
        required_gpu_memory_gb: Optional[float] = None,
        max_gpu_memory_bandwidth: Optional[float] = None,
        max_gpu_performance: Optional[float] = None,
        use_relative_performance: bool = True,
    ) -> float:
        """
        Scores the machine's current ability to perform.
        """
        # Check if this machine has GPU capabilities if required
        if use_gpu and not self.gpus:
            return 0.0

        # Check if this machine has enough memory if required
        if required_memory_gb is not None and self.ram.available < required_memory_gb * 1e9:
            return 0.0

        # Calculate the average CPU load
        cpu_load = sum(cpu.load for cpu in self.cpus) / len(self.cpus)

        # Calculate the total CPU score ratio
        cpu_score_ratio = (
            (1.0 - cpu_load) * CPU_LOAD_WEIGHT +
            (1.0 - self.ram.load) * CPU_MEMORY_UTILIZATION_WEIGHT
        ) / CPU_TOTAL_WEIGHT

        # Calculate the highest GPU score if required
        if use_gpu:
            gpu_score_ratio = max([
                gpu.score_ratio(
                    gpu_precision=gpu_precision,
                    required_cuda_version=required_cuda_version,
                    required_gpu_memory_gb=required_gpu_memory_gb,
                    max_gpu_memory_bandwidth=max_gpu_memory_bandwidth,
                    max_gpu_performance=max_gpu_performance,
                    use_relative_performance=use_relative_performance,
                )
                for gpu in self.gpus
            ])
            # Check if no GPU reported a score
            if gpu_score_ratio == 0.0:
                return 0.0

            # Calculate the combined score ratio
            return float(
                (cpu_score_ratio * COMBINED_CPU_WEIGHT) +
                (gpu_score_ratio * COMBINED_GPU_WEIGHT)
            ) / COMBINED_TOTAL_WEIGHT

        # Otherwise, return the CPU score
        return cpu_score_ratio

    def score(
        self,
        use_gpu: bool = True,
        gpu_precision: str = "half",
        required_cuda_version: Optional[str] = None,
        required_memory_gb: Optional[float] = None,
        required_gpu_memory_gb: Optional[float] = None,
        max_gpu_memory_bandwidth: Optional[float] = None,
        max_gpu_performance: Optional[float] = None,
        use_relative_performance: bool = True,
    ) -> int:
        """
        Scores the machine's current ability to perform out of the maximum total score.
        """
        return int(
            self.score_ratio(
                use_gpu=use_gpu,
                gpu_precision=gpu_precision,
                required_memory_gb=required_memory_gb,
                required_cuda_version=required_cuda_version,
                required_gpu_memory_gb=required_gpu_memory_gb,
                max_gpu_memory_bandwidth=max_gpu_memory_bandwidth,
                max_gpu_performance=max_gpu_performance,
                use_relative_performance=use_relative_performance,
            ) * AVAILABILITY_SCORE_MAX
        )

    def tabulate_cpus(self, tablefmt: str = "simple", cols: int = 4) -> str:
        """
        Returns a tabular representation of the CPUs.
        """
        import tabulate

        rows: List[List[str]] = []
        total_cpus: int = 0
        for chunk in chunk_iterable(self.cpus, cols):
            rows.append([f"CPU {total_cpus+i}" for i, _ in enumerate(chunk)])
            rows.append([f"{cpu.load:.2%}" for cpu in chunk])
            rows.append(tabulate.SEPARATING_LINE) # type: ignore
            total_cpus += len(chunk)
        return "\n".join(
            tabulate.tabulate(rows[:-1], tablefmt=tablefmt).splitlines()[1:-1]
        )

    def tabulate_scores(self, tablefmt: str = "simple") -> str:
        """
        Returns a tabular representation of the MachineCapability scores at the moment,
        without consideration for any other machines in a cluster.
        """
        import tabulate

        gpu_scores = dict([
            (gpu.id, gpu.score(use_relative_performance=False))
            for gpu in self.gpus
        ])
        combined_gpu_score = self.score(use_gpu=True, use_relative_performance=False)
        combined_cpu_score = self.score(use_gpu=False, use_relative_performance=False)

        return tabulate.tabulate(
            [
                ["HP Total", "Total"] + [f"GPU {gpu.id}" for gpu in self.gpus],
                [combined_gpu_score, combined_cpu_score] + [gpu_scores[gpu.id] for gpu in self.gpus],
            ],
            tablefmt=tablefmt,
            headers="firstrow",
        )

    def tabulate(self, tablefmt: str = "fancy_grid") -> str:
        """
        Returns a tabular representation of the MachineCapability.
        """
        import tabulate

        return tabulate.tabulate(
            [
                ["Machine Availability Score (MAS)"],
                [self.tabulate_scores()],
                ["Memory"],
                [self.ram.tabulate()],
                ["CPUs"],
                [self.tabulate_cpus()],
                ["GPUs"],
                [gpu.tabulate() for gpu in self.gpus],
            ],
            tablefmt=tablefmt,
        )

    def __hash__(self) -> int:
        """
        Hash the MachineCapability object.
        """
        return hash((tuple(self.gpus), tuple(self.cpus), self.ram))

    def __str__(self) -> str:
        """
        String representation of the MachineCapability.
        """
        try:
            import tabulate
            return str(self.tabulate())
        except ImportError:
            return (
                f"MachineCapability(gpus={self.gpus}, cpus={self.cpus}, ram={self.ram})"
            )
