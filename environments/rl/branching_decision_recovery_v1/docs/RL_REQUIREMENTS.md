# RL Requirements

## Primary RL Problems

Task 003 targets:

1. partial observability
2. information-seeking behavior
3. exploration
4. long-horizon planning
5. temporal credit assignment
6. delayed consequences
7. costly decisions
8. recovery
9. replanning
10. sparse/progress-based reward

## Reward Principles

Reward should represent useful progress.

Repeated inspection, redundant actions, premature validation, and no-op recovery should not produce meaningful cumulative reward.

Final verified success must dominate the reward signal.

## Reward-Hacking Threats

The implementation must test:

- repeated inspection
- repeated validation
- repeated safe actions
- repeated recovery
- repeated replanning
- invalid-action loops
- partial completion
- premature final verification

## Training/Evaluation Separation

Training scenarios and held-out evaluation scenarios must be isolated.

Hidden scenarios must not be shipped as agent-visible task truth.

## Metrics

The trajectory system should support:

- success rate
- return
- episode length
- planning depth
- information queries
- decision count
- decision efficiency
- wrong decisions
- delayed failures
- recoveries
- replans
- redundant actions
- invalid actions
- total cost
- reward versus actual correctness
