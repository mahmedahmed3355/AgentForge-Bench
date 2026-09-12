# AgentForge-Bench Task Registry

The benchmark registry provides a lightweight catalog of benchmark tasks.

Each task entry should define:

- task id
- environment path
- task family
- domain
- development status
- Prime Verifiers V1 compatibility

Additional metadata such as difficulty, train/evaluation split,
scenario counts, and release version can be added as the benchmark grows.

The registry is metadata only. It must not contain hidden evaluation
truth, oracle answers, reference solutions, or secret scenario data.
