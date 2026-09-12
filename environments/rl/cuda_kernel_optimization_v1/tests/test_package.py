import verifiers.v1 as vf

from cuda_kernel_optimization_v1 import CudaKernelOptimizationTaskset


def test_prime_v1_taskset_contract():
    taskset = CudaKernelOptimizationTaskset(config=vf.TasksetConfig())

    assert isinstance(taskset, vf.Taskset)

    tasks = taskset.load()

    assert len(tasks) == 4
    assert all(isinstance(task, vf.Task) for task in tasks)
