## Workflow

**Pre-work**
- Review all relevant .mdc files for the feature
- Search for relevant specs: `grep -r "relevant-keyword" specs/`
  - If found: Read referenced specs: `cat specs/DEV-XXX-*.md`
  - If none: ask clarifying questions, then create new spec
- Review relevant code files
  - Note down the main architectural patterns that need to be followed
  - Note down any potential complexities or blockers to implementation

**Clarify**
Ask questions to clarify everything
- Reason for the feature, dependencies, etc...
- Which among multiple solution paths to take
- Potential complexities/blockers
- Any inner structure to be aware of
Stop when you believe you could implement the feature with what you know

**Write spec**
Create the spec based on the collected information.
- Spec should have the format described in [specs.mdc](mdc:.cursor/rules/specs.mdc)
- Spec should be self-contained and contain all relevant context
  - If linking to a file, be clear about what information is needed from it.
- Spec should contain all relevant user input up to this point
  - It should be integrated into the spec naturally with other info
- Domain rules (as laid out in domain .mdc files) do not need to be repeated
  - Just make sure you link the domain rules .mdc files

**Iteration**
- User reviews, edits and improves the spec
- May ask you to change/improve/clarify something - do so
- May ask you to [Review the spec](mdc:.cursor/rules/spec_verbs/spec_review.mdc)
- This back-and-forth can keep going for many rounds

## Important guidelines

- **Overall**
  - Be short and concise. This is written for smart and capable developers.
  - Use one-sentence bulletpoints, with sub-points and sub-sub-points as needed
  - Structure the text with headers for different topics so it would be easy to follow

- **Requirements** needs to contain all relevant context for this task
  - If a rules file or spec is relevant, link to it and write down why
  - Same with code files - if something needs to be kept in mind, link the file and the reason
  - Developer implementing the feature is expected to only read what is written or linked here.
    - This has to be enough context

- **Implementation plan** should be properly itemized into tasks
  - Tasks should be clear and concise enough for a mid-level developer to follow
    - Given all the preceeding context of the spec
  - Tasks should be in order they need to be done
  - Each task should have a checkmark in front
    - Clarifying comments can be sub-points under it
 
- **Definition of done** should be explicit
  - Especially in what needs manual testing afterwards  

- **Other sections**
  - Spec number should be XXX initially (as in DEV-XXX)
    - It will be replaced with github PR number once first commit is created
  - Spec status should be "Planning"
  - Create **Implementation Notes** and **Q&A** headings in the spec
    - Leave them empty - they are to be filled in during review and implementation