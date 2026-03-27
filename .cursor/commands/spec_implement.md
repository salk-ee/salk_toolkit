**Workflow:**

- Mark the spec status as "In progress"

- Read the WHOLE spec, as it contains important context
  - Read the relevant domain rules .mdc files listed in spec
  - Note the other files it links to, read on as-needed basis

- Go down the **Implementation Plan** list ONE TASK at a time
  - A TASK is a line with a checkmark in front, everything else is clarification
  - Find the first task that is not yet checked off.
  - Clearly state which task you are starting
  - Build it to the best of your ability
    - Document decisions in **Implementation Notes** as you make them
    - ASK for clarifications if unsure about how to proceed
  - Once done, check your work
    - run tests and linter/typechecker as appropriate
  - Check off the **Implementation Plan** task in spec
  - Move on to the next task without prompting if no issues/questions

- If **Implementation plan** section contains multiple features:
  - Work only on the feature you are asked to work on.
  - If not specified, start from the top.

## Implementation Standards

**Code Style**:

- Python: Follow linter and static typecheck rules (`pyproject.toml`)
- Comment prefixes: `CORNER:`, `PERF:`, `WORKAROUND:`

## Implementation Decision Guidelines

**Follow Patterns**: Use established patterns unless there's a compelling reason to deviate
**Document Deviations**: Always explain why you're departing from standard patterns
**Maintain Backwards Compatibility**: Ensure existing features continue working
**Performance Considerations**: Always keep in mind parquets can be millions of rows