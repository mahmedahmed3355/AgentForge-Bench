from cuda_kernel_optimization_v1.state import KernelOptimizationState


def test_initial_state():
    state = KernelOptimizationState()

    assert state.logical_stage == 0
    assert state.step_count == 0
    assert state.recovery_count == 0
    assert not state.finalized


def test_state_advances():
    state = KernelOptimizationState()

    state.advance()

    assert state.logical_stage == 1
    assert state.step_count == 1


def test_recovery_is_recorded():
    state = KernelOptimizationState()

    state.fail("compile_error")
    state.recover()

    assert state.recovery_count == 1
    assert "compile_error" in state.failure_history
