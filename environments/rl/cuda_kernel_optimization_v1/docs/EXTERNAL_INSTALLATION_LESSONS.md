# External Installation Lessons

## Purpose

This document records packaging and external-installation failures discovered
during clean-room validation of AgentForge-Bench environments.

These rules are part of the canonical authoring baseline for future tasks.

## Lesson 001 — Local Editable Dependencies Can Hide Failures

A dependency available through a local editable installation may make a task
appear installable even when an external user cannot resolve that dependency.

Every published environment must be tested outside the development repository.

## Lesson 002 — Published Environments Must Have Resolvable Runtime Dependencies

A Prime environment must not depend on a package that is unavailable to the
external package resolver.

Runtime dependencies must be available through the supported installation
path used by the published environment.

## Lesson 003 — Avoid Transitive Git URL Dependency Resolution

A Git URL dependency nested inside another package dependency can fail during
uv dependency resolution.

Published environments should avoid relying on transitive direct-reference
dependencies.

## Lesson 004 — Standalone Environment Boundary

If a task only needs a small reusable contract at runtime, that contract may
be kept inside the published environment rather than introducing an
unresolvable external runtime dependency.

The environment must remain behaviorally consistent with the canonical
AgentForge Core contract.

## Lesson 005 — Clean-Room Installation Is Mandatory

A published environment is not considered externally runnable merely because
local tests pass.

Validation must include installation from a clean environment using the
public distribution source.

## Lesson 006 — Version Changes Must Produce a New Release

Changing package metadata or dependencies requires a new semantic package
version before publishing.

Never republish changed package contents under an already published version.

## Lesson 007 — Test Before Prime Publication

The required order is:

1. Modify source.
2. Run task tests.
3. Build package.
4. Inspect package metadata.
5. Commit to GitHub.
6. Push GitHub.
7. Publish a new Prime Hub version.
8. Perform clean-room external installation.
9. Run real environment/tool execution.
10. Only then consider the release externally validated.
