# AgentForge-Bench — Canonical Task Authoring & Prime Intellect Verifiers V1 Guide

## 0. Purpose

This document is the implementation contract for an AI coding agent that must create a new AgentForge-Bench task from scratch.

The agent MUST follow this document as the task-authoring specification.

The target is:

- Long-horizon RL / agent environment
- Stateful, multi-step task
- Programmatic verification
- Deterministic and reproducible evaluation
- Hidden evaluation truth
- Anti-cheating / anti-shortcut design
- Difficulty calibrated against capable agents
- Prime Intellect `verifiers.v1` compatible
- Compatible with `prime-rl` training/evaluation
- Clear Oracle / Verifier separation

IMPORTANT:
Prime Intellect's current architecture is `verifiers.v1`. Do not build against the deprecated legacy `import verifiers as vf` API. New environments should target `import verifiers.v1 as vf`.

---

# 1. Prime Intellect Compatibility Contract

## 1.1 Core architecture

The task MUST fit this conceptual architecture:

    Taskset
       |
       +--> TaskData
       |
       +--> Task
       |      |
       |      +--> setup / state / tools / scoring / stop conditions
       |
       +--> Tools
       |
       +--> Reward / Scoring
       |
       v
    Harness
       |
       v
    Runtime
       |
       v
    Trace
       |
       v
    prime-rl / Evaluation

Prime's separation is:

- Taskset = what work exists
- Task = behavior, setup, scoring, stopping and task-specific logic
- Harness = how the agent solves the task
- Runtime = where the agent/harness executes
- Trace = canonical rollout artifact

A task MUST NOT hard-code a specific agent into the task unless a custom harness is genuinely required.

Prefer built-in Prime harnesses whenever possible.

---

# 2. Required Task Package

Recommended package structure:

    environments/
    └── <family>/
        └── <task_name>_v1/
            ├── README.md
            ├── pyproject.toml
            ├── __init__.py
            ├── taskset.py
            ├── config.py
            │
            ├── data/
            │   ├── train.jsonl
            │   ├── eval.jsonl
            │   └── metadata.json
            │
            ├── environment/
            │   ├── __init__.py
            │   ├── state.py
            │   ├── actions.py
            │   ├── environment.py
            │   └── tools.py
            │
            ├── verifier/
            │   ├── __init__.py
            │   ├── verifier.py
            │   ├── correctness.py
            │   ├── hidden_tests.py
            │   └── metrics.py
            │
            ├── rewards/
            │   ├── __init__.py
            │   ├── reward.py
            │   └── shaping.py
            │
            ├── oracle/
            │   ├── __init__.py
            │   └── solve.py
            │
            ├── scenarios/
            │   ├── generator.py
            │   ├── train.py
            │   ├── eval.py
            │   └── hidden.py
            │
            └── tests/
                ├── test_taskset.py
                ├── test_environment.py
                ├── test_state.py
                ├── test_actions.py
                ├── test_reward.py
                ├── test_verifier.py
                ├── test_oracle.py
                ├── test_determinism.py
                ├── test_reset_isolation.py
                ├── test_hidden_evaluation.py
                ├── test_anti_cheat.py
                ├── test_reward_hacking.py
                └── test_failure_recovery.py

Not every task requires every file.

Do not create files merely for appearance. Each file must have a real responsibility.

---

# 3. Minimum Required Components

Every production task MUST have:

1. Task specification
2. Taskset
3. Typed TaskData
4. Task implementation
5. Environment/state implementation
6. Action/tool interface
7. Reward/scoring
8. Oracle/reference solution
9. Verifier
10. Hidden evaluation
11. Scenario/data generation
12. Tests
13. Determinism/reproducibility
14. Documentation
15. Prime-compatible package metadata
16. Validation script/config

---

# 4. `taskset.py`

This is the Prime V1 entry point for the taskset.

Use:

    import verifiers.v1 as vf

The taskset should define:

- TaskData subclass
- Task subclass
- Taskset subclass
- deterministic/lazy task loading where appropriate
- task-specific configuration
- tools if required

Conceptual shape:

    class MyTaskData(vf.TaskData):
        ...

    class MyTask(vf.Task[MyTaskData]):
        ...

    class MyTaskset(vf.Taskset[MyTask, MyTasksetConfig]):
        def load(self):
            ...

Do not put hidden answers or verifier secrets into agent-visible prompt text.

---

# 5. `TaskData`

`TaskData` is the immutable, serializable data for one task instance.

It may contain:

- task id
- prompt
- scenario id
- initial state parameters
- public metadata
- resource limits
- seed
- timeout
- task-specific inputs

It MUST NOT contain secrets that are directly serialized into the agent-visible task.

Reference answers, hidden expected states, hidden test cases and evaluator secrets must remain in the evaluation side.

Use typed fields.

Example conceptual shape:

    class MyTaskData(vf.TaskData):
        scenario_id: str
        prompt: str
        seed: int
        difficulty: str

---

# 6. Task Configuration

Create a typed configuration class when run-level knobs are required.

Example:

    @dataclass
    class MyTasksetConfig(vf.TasksetConfig):
        split: str = "train"
        seed: int = 0

Task configuration should contain configurable behavior.

Do NOT put per-example data in config.

Per-example data belongs in `TaskData`.

Do NOT use configuration as a secret store.

---

# 7. Prime V1 Environment Loader

For reusable Taskset × Harness environments, use the V1 loader architecture:

    def load_taskset(config: MyTasksetConfig) -> MyTaskset:
        return MyTaskset(config=config)

    def load_harness(config: MyHarnessConfig) -> MyHarness:
        return MyHarness(config=config)

    def load_environment(config: vf.EnvConfig) -> vf.Env:
        taskset = vf.load_taskset(config.taskset)
        harness = vf.load_harness(config.harness)
        return vf.Env(
            taskset=taskset,
            harness=harness,
        )

The loader should be tiny.

Do not instantiate expensive resources in the loader.

Do not silently accept `None` and manufacture configuration defaults.

Do not pass already-instantiated resource objects through the environment loader when the framework expects loader specifications.

---

# 8. Harness

Do NOT create a custom harness unless necessary.

Prefer compatible built-in harnesses such as:

- Codex
- Kimi Code
- Terminus 2
- Mini-SWE-Agent
- other Prime-supported harnesses

Create a custom harness only if the task genuinely requires custom agent execution semantics.

The same task should ideally be solvable by multiple compatible harnesses.

This is a major design goal of Prime V1.

---

# 9. Runtime

The task must be runtime-independent where practical.

Possible runtimes include:

- subprocess
- Docker
- Prime sandbox
- Modal / supported remote sandbox

For dangerous code execution, compilation, CUDA, filesystem mutation or system-level operations:

- use isolation
- set timeouts
- restrict resources
- clean up after rollout
- ensure one rollout cannot modify another rollout

Never depend on the developer's host filesystem.

---

# 10. State Design

A long-horizon task MUST have explicit state.

Example:

    State:
        step
        phase
        completed_stages
        current_branch
        checkpoints
        failure_history
        recovery_count
        candidate_state
        validation_status
        performance_history
        terminal_status

State transitions MUST be explicit.

Avoid hidden mutable state that cannot be reproduced.

The state should support:

- reset
- deterministic replay
- checkpointing where required
- failure recovery
- terminal verification
- trajectory analysis

---

# 11. Logical Stages vs Agent Actions

CRITICAL RULE:

    20 logical stages != 20 agent actions

A task may have:

    20 logical stages
    16 successful actions

or:

    20 logical stages
    28 actions because recovery was required

The horizon is a budget, not a requirement that the agent perform exactly N actions.

Define:

- logical stage count
- maximum turns/actions
- timeout
- terminal condition

separately.

---

# 12. Action Design

Actions must be explicit and validated.

Example:

    inspect
    modify
    validate
    execute
    analyze
    checkpoint
    restore
    benchmark
    recover
    finalize

Every action needs:

- valid arguments
- invalid-argument behavior
- preconditions
- postconditions
- state transition
- reward effect
- failure behavior

Invalid actions must not accidentally advance the task.

Repeated actions should be handled intentionally.

---

# 13. Tool Design

If tools are needed:

- make the interface narrow
- validate all arguments
- return deterministic structured results when possible
- do not expose hidden evaluation state
- separate public observations from evaluator truth
- apply resource/time limits
- record tool calls in the rollout trace

A tool should reveal only information that a real agent could legitimately obtain.

Do not create a tool such as:

    get_hidden_solution()

or:

    reveal_expected_state()

or:

    run_hidden_tests()

unless the task explicitly models such a public operation.

---

# 14. Reward Design

Reward MUST reflect the actual objective.

Recommended structure:

    total_reward =
        progress
        + milestone
        + recovery
        + correctness
        + performance
        + efficiency
        + final_success
        - invalid_action_penalty
        - harmful_action_penalty

But:

    final_success

must dominate the shaping signal.

Avoid reward structures where the agent can collect high reward without actually completing the task.

A task should have:

- intermediate feedback
- milestone rewards
- recovery rewards where appropriate
- penalties for harmful behavior
- strong terminal reward

For sparse tasks, shaping should help learning without becoming a shortcut.

---

# 15. Reward-Hacking Protection

The task MUST be tested against reward hacking.

Potential attacks:

- repeatedly executing cheap positive actions
- farming milestone rewards
- triggering partial-credit states repeatedly
- resetting to farm reward
- exploiting rounding
- exploiting reward-before-validation
- exploiting stale state
- exploiting duplicate submissions
- exploiting timeout behavior
- manipulating benchmark measurements
- modifying verifier inputs
- modifying hidden files
- causing evaluator errors that are interpreted as success

Tests MUST demonstrate that these strategies do not outperform genuine completion.

---

# 16. Oracle

Every task MUST have a reference solution.

Oracle responsibilities:

- demonstrate solvability
- establish expected state transitions
- establish expected final state
- generate reference trajectories where useful
- validate verifier correctness

Oracle MUST NOT be exposed to the agent.

The Oracle is not the evaluator itself.

Preferred separation:

    oracle/
        solve.py

    verifier/
        verifier.py
        correctness.py
        hidden_tests.py

The Oracle should solve using the same public interface available to the agent whenever possible.

This prevents an oracle from using privileged operations that the agent could never perform.

---

# 17. Verifier

The verifier is the final authority.

It should check the actual outcome, not the agent's claim.

Bad:

    agent says "done" -> reward 1

Good:

    agent claims done
        |
        v
    inspect actual environment
        |
        v
    run correctness checks
        |
        v
    verify required terminal state
        |
        v
    calculate reward

Verifier should check:

- final state
- required artifacts
- correctness
- constraints
- hidden conditions
- invariants
- side effects
- absence of corruption
- performance where relevant

---

# 18. Hidden Tests

Public tests are not sufficient.

Use three layers:

## Layer A — Public / structural tests

Agent-authoring tests:

- imports work
- package loads
- taskset loads
- task data validates
- basic environment behavior
- obvious correctness

## Layer B — Reference tests

Used by maintainers:

- Oracle succeeds
- all required stages work
- recovery works
- verifier accepts valid solutions
- invalid solutions fail

## Layer C — Hidden evaluation

Never expose:

- hidden scenario parameters
- hidden expected outputs
- hidden seeds when they function as secrets
- hidden test cases
- reference solution details
- exact scoring internals
- evaluator-only files

Hidden evaluation should test general correctness rather than memorization.

---

# 19. Truth Isolation

This is mandatory for hard tasks.

Separate:

    PUBLIC AGENT STATE

from:

    EVALUATION TRUTH

Example:

    public/
        scenario.json
        observations.json

    hidden/
        expected_state.json
        hidden_cases.json
        evaluator_seed.json

The agent can see only the public side.

The verifier can access both.

Never mount hidden files into the agent runtime if they can be read by the agent.

If hidden data must exist inside a container, use:

- separate mount
- permissions
- separate evaluator process
- protected location
- evaluator-side access

Do not rely on "the agent probably won't look there."

---

# 20. Anti-Cheating Requirements

Every task must be reviewed for:

## Filesystem leakage

Check that:

- hidden files are not readable
- oracle code is not readable
- verifier secrets are not readable
- test answers are not readable
- previous rollout artifacts are not accessible

## Environment-variable leakage

Do not store secrets in agent-visible environment variables.

## Process leakage

The agent must not be able to inspect evaluator processes or command lines to obtain secrets.

## Git leakage

Do not commit:

- hidden answer files
- gold patches in agent-visible paths
- hidden evaluation data
- private evaluator configuration

## Package/source leakage

Do not install evaluator-only modules into the agent environment if their source reveals answers.

## Error leakage

Error messages must not reveal:

- hidden expected values
- reference paths
- secret scenario parameters
- verifier internals

Return useful but non-secret errors.

---

# 21. Anti-Memorization

A hard task should not be solvable by memorizing a fixed scenario.

Use:

- parameterized scenarios
- seeded generation
- held-out evaluation scenarios
- multiple valid trajectories
- adversarial edge cases
- unseen combinations
- randomized inputs where appropriate
- lexical/surface variation when relevant

Train and evaluation distributions must be related but not identical.

Recommended:

    TRAIN
      |
      +--> known structural family
      +--> variable parameters
      +--> multiple scenarios

    EVAL
      |
      +--> same underlying capability
      +--> unseen parameter combinations
      +--> unseen scenario instances
      +--> adversarial edge cases

Do not make evaluation fundamentally different from the learned skill.

---

# 22. Difficulty / Hard Task Design

A task becomes HARD through reasoning and dependency, not arbitrary obscurity.

Recommended difficulty mechanisms:

### 22.1 Long-horizon dependency

Later actions depend on earlier decisions.

Example:

    inspect
      ->
    choose branch
      ->
    execute
      ->
    observe consequence
      ->
    diagnose
      ->
    recover
      ->
    re-plan
      ->
    final verification

### 22.2 Branching

Create multiple plausible actions.

At least one path should be viable, but the correct choice should not be obvious from superficial cues.

### 22.3 Delayed consequences

Do not immediately reveal that an earlier action was wrong.

### 22.4 Recovery

A wrong but recoverable decision should require diagnosis and corrective action.

### 22.5 Partial state

The agent should sometimes need to reconstruct what has happened from observations/history.

### 22.6 Hidden evaluation

The final evaluator checks more than the obvious surface condition.

### 22.7 Unseen scenarios

Evaluation must include scenarios not present in training.

### 22.8 Distractors

Include plausible but irrelevant information/actions.

### 22.9 Repeated actions

Test idempotency and repeated calls intentionally.

### 22.10 Sparse terminal reward

The agent should not receive a large reward merely for making progress; genuine completion must matter most.

---

# 23. Recommended AgentForge Hardness Strategies

Use these deliberately when appropriate:

- HRIG — Hidden Rule Induction + Generalization
- HRR — Hierarchical Rule Resolution
- LEM — Lexical Exception Memorization resistance
- PRE — Productive Rule Elimination
- BET — Blocking Environment Trap
- AP — Ambiguity Pruning
- RDT — Rule Destruction Trap
- MSP — Minimal Surface Pair Trap
- FFT — False Friend Trap
- LMR — Longest Match Resolution
- RKB — Rare Knowledge Bottleneck
- SKF — Single-Knowledge Filtering
- DDT — Diacritic Drop Trap
- CCR — Cultural Convention Resolution
- CCI — Convention Conflict Injection
- MGK — Meaning-Gated Knowledge
- MBT — Meaning Blindness Trap
- SKS — Specialized Knowledge System
- EKR — Encoding Knowledge Retrieval
- KOS — Knowledge-Oriented Sorting
- HKS — Hidden Key Sorting
- FCS — Format-Constrained Symbolism
- DLR — Document Layout Reconstruction
- SNT — Selective Noise Targeting

Do not stack every strategy into one task.

Select mechanisms that test the intended capability.

---

# 24. Scenario Generator

A production task should have a scenario generator when generalization matters.

The generator should support:

    generate(seed)
    generate_train(seed)
    generate_eval(seed)
    generate_hidden(seed)

Requirements:

- deterministic for a fixed seed
- bounded
- validated
- no accidental overlap between train/eval/hidden
- reproducible
- capable of producing edge cases

Store scenario IDs and generation parameters.

Do not expose hidden generation parameters to the agent.

---

# 25. Dataset Splits

At minimum:

    train
    eval
    hidden

Prefer:

    train/
    validation/
    public_eval/
    hidden_eval/

The hidden split must not be copied into the public dataset.

For generated tasks, keep a deterministic generation scheme so maintainers can reproduce failures without exposing the hidden set to agents.

---

# 26. Test Suite — Mandatory Categories

Every new task MUST have tests covering:

## A. Import / package

- imports
- dependency installation
- taskset discovery

## B. TaskData

- schema
- serialization
- deterministic fields

## C. Environment

- initialization
- transitions
- terminal state
- reset

## D. Actions

- valid actions
- invalid actions
- malformed arguments
- repeated actions
- boundary cases

## E. Oracle

- Oracle completes task
- Oracle reaches valid final state
- Oracle reward is high/maximum

## F. Verifier

- valid solution accepted
- invalid solution rejected
- partial solution scored correctly
- malformed result rejected

## G. Reward

- progress
- milestones
- recovery
- final success
- penalties
- no reward farming

## H. Determinism

Same seed + same actions must produce equivalent outcomes.

## I. Reset isolation

A previous episode must not contaminate the next episode.

## J. Hidden evaluation

Hidden checks cannot be bypassed through public information.

## K. Anti-cheat

Agent cannot read:

- oracle
- hidden tests
- hidden answers
- evaluator secrets

## L. Reward hacking

Attempt common farming strategies.

## M. Failure recovery

Verify intended recoverable failures.

## N. Timeout/resource safety

Verify runaway actions/processes are stopped safely.

---

# 27. Oracle PASS Gate

Before difficulty calibration:

    Oracle PASS = mandatory

The Oracle should reliably solve the task.

If Oracle fails:

    STOP

Do not make the task harder.

Fix correctness first.

---

# 28. Baseline / No-Op Gate

A useful benchmark should include weak baselines.

At minimum test:

- no-op agent
- random/invalid-action agent where applicable
- greedy baseline
- simple heuristic baseline

Expected behavior:

    Oracle              -> PASS
    competent agent     -> meaningful success
    weak baseline       -> low success
    no-op               -> FAIL

This prevents a broken verifier from giving reward to everyone.

---

# 29. Difficulty Calibration

Difficulty must be measured, not guessed.

Run multiple episodes against target agents.

Collect:

- pass rate
- reward distribution
- average reward
- episode length
- action count
- stage reached
- recovery count
- failure step
- verifier failure category
- timeout rate
- tool error rate
- reward-hacking attempts
- generalization score

Example report:

    Task: task-XYZ
    Episodes: 20

    Oracle:             20/20 PASS
    Weak baseline:       1/20 PASS
    Agent A:             6/20 PASS
    Agent B:             11/20 PASS

    Mean reward:         0.42
    Mean actions:        23.7
    Mean recovery:       1.4
    Main failure:        incorrect branch selection

Do not calibrate difficulty by making the verifier flaky.

---

# 30. Determinism

A task MUST be reproducible.

For fixed:

- task version
- scenario seed
- initial state
- action sequence

the resulting state should be deterministic unless controlled nondeterminism is explicitly part of the task.

If stochasticity is required:

- seed it
- record it
- isolate it
- make evaluation reproducible

---

# 31. Idempotency and State Safety

Every mutating operation should define whether it is:

- idempotent
- repeatable
- rejected on repetition
- ignored
- state-changing

Test:

    action()
    action()

and:

    action()
    reset()
    action()

and:

    action_A
    action_B
    action_A

where relevant.

---

# 32. Failure and Recovery Model

Define failure classes explicitly:

    INVALID_ACTION
    INVALID_ARGUMENT
    PRECONDITION_FAILURE
    EXECUTION_FAILURE
    CORRECTNESS_FAILURE
    PERFORMANCE_REGRESSION
    STATE_CORRUPTION
    TIMEOUT
    RESOURCE_EXHAUSTION

For each failure define:

- whether recoverable
- how agent observes it
- required recovery action
- reward effect
- terminal/non-terminal behavior

Do not make every failure terminal.

---

# 33. Security / Sandbox Requirements

For code execution tasks:

- Docker or Prime sandbox
- non-root where possible
- CPU limit
- memory limit
- disk limit
- process limit where available
- timeout
- network restriction where appropriate
- temporary workspace
- cleanup
- no access to host secrets

The verifier must execute separately from untrusted agent code whenever practical.

---

# 34. CUDA-Specific Requirements

For CUDA tasks:

- CUDA availability must be validated
- compiler/toolchain must be pinned or detected
- reference implementation must be protected
- correctness tests must use multiple inputs
- randomized/held-out inputs are preferred
- benchmark must be independently measured
- performance must not be accepted without correctness
- timing noise must be controlled
- warmup runs should be separated from measured runs
- agent must not modify the reference implementation
- agent must not modify the benchmark/evaluator

Correctness gate:

    candidate correct
        AND
    performance requirement satisfied

not:

    candidate fast
        OR
    candidate correct

---

# 35. File Naming Rules

Use predictable names.

Recommended:

    README.md
    pyproject.toml
    __init__.py
    taskset.py
    config.py

    environment.py
    state.py
    actions.py
    tools.py

    verifier.py
    correctness.py
    hidden_tests.py
    metrics.py

    reward.py
    shaping.py

    solve.py
    generator.py

Avoid:

    final.py
    new.py
    test2.py
    backup.py
    old.py
    fixed.py
    working.py
    temp.py

Never leave accidental files such as:

    system.py.before-fix
    system.py.after-fix
    action:
    assert

inside the production task package.

Use Git history for old versions.

---

# 36. `README.md` Requirements

README MUST explain:

1. Task name
2. Capability being tested
3. Task objective
4. Logical stages
5. Action/tool interface
6. State model
7. Reward
8. Failure/recovery
9. Oracle
10. Verifier
11. Hidden evaluation
12. Anti-cheating model
13. Scenario generation
14. Difficulty
15. Runtime requirements
16. Installation
17. Local execution
18. Tests
19. Prime V1 execution
20. Known limitations

Do not publish hidden answers or hidden test logic.

---

# 37. `pyproject.toml`

Pin the framework range that the task was validated against.

Example:

    [project]
    name = "agentforge-<task-name>"
    version = "1.0.0"

    dependencies = [
        "verifiers>=0.3.1,<0.4",
        "pydantic>=2,<3",
    ]

Add only dependencies actually used.

Optional dependencies:

    pytest
    numpy
    datasets
    gymnasium
    torch

CUDA-specific packages should be optional or environment-specific when possible.

Do not make the Prime task depend on the entire AgentForge repository unless there is a deliberate package dependency.

---

# 38. Recommended Development Tooling

Core:

- Python 3.13 recommended for current Prime/Verifiers development
- uv
- Git
- pytest
- ruff
- mypy or pyright if the project uses static typing
- Docker for isolated execution where needed

Prime:

- `verifiers.v1`
- Prime CLI
- `prime-rl` for RL training integration

Data:

- JSONL for lightweight generated task data
- Pydantic for typed validation
- `datasets` only where useful

RL / environment:

- Gymnasium only if the native AgentForge environment actually exposes a Gymnasium API
- do not add Gymnasium solely to claim compatibility

CUDA:

- PyTorch where needed
- CUDA toolkit / nvcc
- NumPy where needed

Testing:

- pytest
- deterministic seeds
- subprocess/container tests
- hidden evaluation tests maintained outside the public task surface

---

# 39. Prime V1 Validation

A task is NOT Prime-compatible merely because:

    import verifiers.v1

works.

Required validation:

### Stage 1 — Package

- install succeeds
- taskset imports
- config validates

### Stage 2 — Taskset

- Taskset loads
- TaskData validates
- deterministic task generation
- head/sample works

### Stage 3 — Local evaluation

Run a small evaluation with a supported harness.

### Stage 4 — Oracle

Oracle completes successfully.

### Stage 5 — Verifier

Valid and invalid trajectories are distinguished.

### Stage 6 — Runtime

Run inside intended runtime:

- subprocess
- Docker
- Prime sandbox
- or other supported runtime

### Stage 7 — Trace

Confirm the rollout produces a valid V1 trace.

### Stage 8 — Training smoke test

Run a minimal `prime-rl` smoke test if training compatibility is claimed.

### Stage 9 — Adversarial validation

Run anti-cheat and reward-hacking tests.

### Stage 10 — Reproducibility

Repeat the same seed/trajectory and verify equivalent results.

---

# 40. Definition of Done

A task is DONE only if ALL are true:

[ ] README complete
[ ] pyproject complete
[ ] Taskset implemented
[ ] TaskData implemented
[ ] Task implemented
[ ] State model implemented
[ ] Action/tool interface implemented
[ ] Reward implemented
[ ] Oracle implemented
[ ] Verifier implemented
[ ] Hidden evaluation implemented
[ ] Scenario generator implemented where needed
[ ] Train/eval/hidden separation implemented
[ ] Anti-cheat checks pass
[ ] Reward-hacking checks pass
[ ] Reset-isolation tests pass
[ ] Determinism tests pass
[ ] Failure/recovery tests pass
[ ] No-op baseline fails
[ ] Oracle passes
[ ] Prime V1 local evaluation passes
[ ] Intended runtime passes
[ ] Trace validation passes
[ ] RL smoke test passes if training compatibility is claimed
[ ] No secret leakage
[ ] No stale/backup/temp files
[ ] Git diff reviewed
[ ] Documentation matches actual implementation

---

# 41. Agent Execution Protocol

When an AI coding agent receives a task specification, it MUST work in this order:

## Phase 1 — Design

Write:

- task objective
- capability
- state machine
- logical stages
- actions
- success condition
- failure classes
- recovery paths
- reward design
- hidden evaluation design
- anti-cheat plan

STOP if any part is ambiguous.

## Phase 2 — Architecture

Create the package structure.

## Phase 3 — Environment

Implement:

- state
- actions
- tools
- transitions
- reset
- terminal state

## Phase 4 — Verifier

Implement final-state and correctness checks.

## Phase 5 — Oracle

Implement a reference solution using public interfaces.

## Phase 6 — Tests

Write tests before declaring the task complete.

## Phase 7 — Oracle Gate

Run Oracle.

If Oracle fails:

    FIX TASK

Do not increase difficulty.

## Phase 8 — Anti-Cheat

Attempt:

- file discovery
- environment-variable inspection
- process inspection
- source inspection
- hidden path discovery
- answer extraction
- reward farming
- reset farming
- verifier manipulation

Fix all successful attacks.

## Phase 9 — Difficulty

Only after correctness and anti-cheat are stable:

- add branching
- add delayed consequences
- add recovery
- add hidden evaluation
- add unseen scenarios
- add distractors
- calibrate against target agents

## Phase 10 — Prime

Validate:

- V1 Taskset
- TaskData
- Task
- environment loader
- harness
- runtime
- trace
- evaluation
- prime-rl smoke test

## Phase 11 — Final Audit

Verify that README, code, tests, package metadata and actual behavior agree.

---

# 42. Final Task Quality Model

A high-quality AgentForge task should satisfy:

    Solvable
        +
    Verifiable
        +
    Reproducible
        +
    Generalizable
        +
    Hard
        +
    Anti-cheat
        +
    Reward-safe
        +
    Prime-compatible

The goal is NOT:

    "make the task impossible"

The goal is:

    "make the intended capability necessary for success."

---

# 43. Canonical Architecture Diagram

    ┌──────────────────────────────────────────────┐
    │              AgentForge Task                 │
    ├──────────────────────────────────────────────┤
    │                                              │
    │  TaskData                                    │
    │      │                                       │
    │      ▼                                       │
    │  Task / Taskset                              │
    │      │                                       │
    │      ├── State                               │
    │      ├── Actions / Tools                     │
    │      ├── Reward                              │
    │      └── Stop Conditions                     │
    │                                              │
    │      │                                       │
    │      ▼                                       │
    │  Agent / Harness                             │
    │      │                                       │
    │      ▼                                       │
    │  Isolated Runtime                            │
    │      │                                       │
    │      ▼                                       │
    │  Trajectory / Trace                           │
    │                                              │
    └───────────────┬──────────────────────────────┘
                    │
          ┌─────────┴──────────┐
          ▼                    ▼
      Public State        Hidden Evaluator
                               │
                     ┌─────────┴─────────┐
                     ▼                   ▼
                  Verifier             Oracle
                     │
                     ▼
                  Reward
                     │
                     ▼
                 Evaluation
                     │
                     ▼
                  prime-rl

---

# 44. Prime Intellect Reference

The task author should consult the current official Prime documentation before implementing against the framework because the V1 API is actively evolving.

Primary references:

- Verifiers V1 Tasksets:
  https://github.com/PrimeIntellect-ai/verifiers/blob/main/docs/v1/tasksets.md

- Prime Environment documentation:
  https://docs.primeintellect.ai/verifiers/environments

- Verifiers V1 architecture:
  https://www.primeintellect.ai/blog/verifiers-v1

- Prime Verifiers overview:
  https://github.com/PrimeIntellect-ai/verifiers/blob/main/docs/overview.md

---

# 45. Non-Negotiable Rules for the Coding Agent

1. Do not use legacy `import verifiers as vf` for a new task.
2. Use `verifiers.v1`.
3. Do not expose Oracle logic.
4. Do not expose hidden tests.
5. Do not expose evaluator truth.
6. Do not trust agent self-reported success.
7. Verify final state programmatically.
8. Oracle MUST pass before difficulty tuning.
9. No-op/weak baseline MUST fail.
10. Fixed seeds MUST be reproducible.
11. Reset MUST isolate episodes.
12. Reward farming MUST not beat genuine completion.
13. Hidden evaluation MUST test generalization, not arbitrary surprises.
14. Do not create difficulty by making the environment flaky.
15. Keep Harness and Runtime separate from Taskset unless custom behavior is necessary.
16. Keep production package free of backup/temp/debug artifacts.
17. Add only dependencies that are actually required.
18. Validate the task under the exact supported `verifiers` version range.
19. If claiming `prime-rl` compatibility, run an actual training smoke test.
20. Never declare a task complete until the Definition of Done checklist passes.
