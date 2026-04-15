# Showdown Debugging Loop

## Context

We now have a working Showdown debugging workflow for websocket invalid choices and similar runtime legality failures. The goal is to preserve the loop as a reusable operating procedure for future profiling and debugging passes.

This workflow was built while debugging:

- early invalid choices caused by move-slot ordering drift
- later invalid choices caused by force-switch mask legality drift

Both bugs were found under training-like Showdown concurrency using targeted diagnostic scripts rather than ad hoc replay inspection.

## Process

Use the following Ralph Wiggum loop for Showdown runtime debugging. The loop should continue until the error family is resolved or a genuine blocker requires input.

### Ralph Wiggum Loop

1. **Run battles under the right pressure profile**
   - Reproduce the issue with the same backend, concurrency, batching, and checkpoint characteristics that matter for training.
   - Prefer diagnostic scripts that emit machine-readable artifacts (`jsonl`, summary JSON, readable markdown).
   - Use random teams from gen9vgc2024regg (teamrepo sampled) each battle to get a variety of experiences

2. **Find the errors**
   - Extract invalid choices or failures from the run artifacts.
   - Confirm counts by request type, turn window, and battle tag.

3. **Group the errors before reasoning about fixes**
   - Group by exact error message.
   - Group by attempted `/choose` shape.
   - Group by request type (`turn`, `force_switch`, `teampreview`, etc.).
   - Identify whether repeated records are true distinct failures or retries of the same bad request.

4. **Analyze the grouped family**
   - Pick the biggest grouped family
   - Trace the path from model action or heuristic choice to final Showdown serialization.
   - Identify the shared invariant that is broken.
   - Prefer source-cause explanations over transport-layer or retry-layer explanations.
   - We only care about Showdown implementations and not Rust implementations

5. **Suggest changes and stop for Cayman’s input**
   - Provide concrete examples of what's broken and why
   - Present the smallest source-level fix that matches the grouped evidence. Be specific and detailed about what you'd change.
   - Explain tradeoffs, expected performance impact, and why it is the right abstraction boundary.
   - Stop here for explicit input unless the user already asked directly for implementation.

6. **Work on the solution after approval**
   - Implement the minimal source fix.
   - Avoid downstream patches unless the grouped evidence proves the source cannot be corrected cleanly.

7. **Create focused tests**
   - Add the smallest regression tests that pin the exact legality rule or invariant.
   - Prefer tests at the layer where the bug lives.
   - Run focused tests first before broader reruns.

8. **Rerun the diagnostic**
   - Use the same reproduction profile or a close variant.
   - Confirm that the grouped family disappears.
   - Summarize the fixes and tests
   - If new residual families remain, return to step 2 and repeat.
   - If no errors are found, increase tests 10x until you can run 3500 without any errors

9. **Document the bug and the fix**
   - Add or update a planning note with context, before state, problem, solution, reasoning, and next steps.
   - Record any compact repository memory notes that will help future sessions start faster.
