# Task 002 — Anti-Memorization Design

## Goal

The benchmark should reward general recovery reasoning rather than memorization
of a fixed trajectory.

## Public Scenario Policy

Public scenarios describe the task family and incident type but must not encode
the evaluator's hidden expected action sequence.

## Scenario Diversity

Future evaluation scenarios should vary:

- seeds
- component locations
- incident manifestations
- dependency consequences
- branch combinations
- failure timing
- downstream effects
- recovery requirements

## Unseen Evaluation

Hidden evaluation should use scenarios that are not present in the public
training examples.

A successful public trajectory must therefore not be sufficient to solve every
hidden episode.

## No Fixed Action Script

The environment must not require a single exact action sequence.

Multiple valid recovery paths may exist when they satisfy the task's public
execution contract.

## No Action-Count Solution

Logical stages describe progress through the task.

They must not become a shortcut where an agent succeeds merely by performing a
fixed number of actions.

Final success should depend on actual state predicates and global validation.

## No Hidden Truth in Observations

Observations may expose consequences that an agent could legitimately inspect.

They must not expose evaluator-only labels or the answer to future decisions.

## Anti-Memorization Evaluation

The final benchmark should measure:

1. success on public scenarios
2. success on unseen scenarios
3. recovery after unexpected consequences
4. robustness to branch variation
5. ability to replan
6. global correctness after recovery

A memorized trajectory should therefore have limited transfer to unseen
episodes.
