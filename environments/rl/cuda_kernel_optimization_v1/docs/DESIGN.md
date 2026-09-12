# CUDA Kernel Optimization V1 Design

## State

The environment state tracks:

- logical stage
- action step count
- candidate modification
- compilation status
- correctness status
- benchmark status
- performance status
- analysis status
- recovery count
- failure history
- finalization

## Failure Model

Failures are recoverable.

Examples include:

- compilation failure
- correctness failure
- performance regression
- invalid candidate
- failed final verification

Recovery must not erase the episode history.

## Oracle

The oracle represents a known successful terminal state.

Oracle validation is required before difficulty calibration.

## Verifier

The verifier independently checks:

- correctness
- performance
- completion of the logical horizon
- finalization

Agent self-report is never accepted as proof of success.

## Anti-Cheat Boundary

The production version must isolate:

- hidden performance targets
- hidden correctness cases
- reference implementation
- verifier internals
- hidden scenarios

from agent-visible task data.

## Production CUDA Layer

The next implementation layer will add:

- isolated CUDA workspace
- candidate source files
- compiler invocation
- randomized correctness tests
- independent reference implementation
- benchmark runner
- performance aggregation
- timeout handling
- resource limits
- hidden test execution
