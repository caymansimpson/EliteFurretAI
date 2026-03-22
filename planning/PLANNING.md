# Planning & Decision Log

This folder serves as the **persistent memory and history** of development conversations, implementation plans, and architectural decisions for EliteFurretAI.

## Purpose

AI assistants (Copilot, Claude, etc.) lose context between sessions. This folder ensures that:

1. **Decisions are preserved**: Every significant conclusion, trade-off, or design choice is documented with full reasoning
2. **Implementation plans survive sessions**: Multi-step plans can be picked up by any model in any future session
3. **Context is recoverable**: Each document contains enough background that a new session can understand the "before state," the problem, the solution, and why it's the right approach for our goals
4. **Progress is trackable**: Completed vs. planned work is clearly delineated

## Document Format

Each document is prefixed with the date (`YYYY-MM-DD`) and contains:

- **Context**: What we were working on and why
- **Before State**: How things worked before the change
- **Problem**: What needed to be fixed, improved, or added
- **Solution**: What we decided to do and the implementation details
- **Reasoning**: Why this approach is better for the goal of building the best VGC bot
- **Planned Next Steps**: How we plan to implement this change
- **Updates**: How our implementation plans went, and what bugs we encountered.

## Naming Convention

```
YYYY-MM-DD-hh-mm-short-description.md
```

Examples:
- `2026-02-28-observation-embedding-improvements.md`
- `2026-02-16-ps-ppo-improvements.md`

## How to Use -- Triggers to keep in mind:
- **Starting a new session**: Find the the most recent file in `planning/*/*.md` to understand which stage of the Project Flow we're at. Confirm with Cayman the stage of development we're working on.
- **After making decisions**: Create or update a document capturing the decision and its rationale.
- **Before implementing**: Check if there's an existing plan document for the work.
- **After implementing**: Update the document with completion status and any deviations from the plan. Also add this context to the relevant primary markdown files (e.g. `rl/RL.md`) when necessary.
- **When I move onto a new stage**: prompt whether you want to update this document to keep everything up-to-date.