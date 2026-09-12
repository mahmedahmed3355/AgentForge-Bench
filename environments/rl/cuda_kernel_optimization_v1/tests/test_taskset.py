import verifiers.v1 as vf

from cuda_kernel_optimization_v1.taskset import CudaKernelOptimizationTaskset


def test_taskset_loads():
    taskset = CudaKernelOptimizationTaskset(config=vf.TasksetConfig())

    tasks = taskset.load()

    assert len(tasks) == 4
    assert tasks[0].data.logical_stages == 20
    assert tasks[0].data.scenario_id == "cuda-v1-train-001"
    assert tasks[-1].data.scenario_id == "cuda-v1-hidden-001"
