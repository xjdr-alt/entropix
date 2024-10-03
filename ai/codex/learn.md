# AI Insight Acquisition Protocol

Invocation: @learn.md

Process:

1. Analyze current session for error and learning indicators
2. Extract contextual information
3. Format new entries
4. Edit `.ai/codex/codex.md` (Append to relevant section)
5. Report updates

Error identification:

- Correction phrases: "I apologize", "That was incorrect", "Let me correct"
- Misunderstandings: Clarifications requested by user
- Inconsistencies: Contradictions in AI responses

Learning identification:

- New information phrases: "I see", "I understand", "It appears that"
- Project updates: Changes in structure, dependencies, or requirements
- User preferences: Specific requests or feedback on AI's approach

Entry format:

- Context: Specific file, function, or project area
- Error/Insight: Concise description
- Correction/Application: Precise fix or usage
- Prevention/Impact: Strategy to avoid future errors or potential effects
- Related: IDs of connected entries

Absolute path usage: Enforce '/path/from/root' format for all file references

1. CRITICAL: Edit `.ai/codex/codex.md`

- Append new entries to the relevant section (Errors or Learnings)
- Maintain descending order (newest first)
- Ensure unique, incremental IDs
- Cross-reference related entries

Exclusions:

- CI/CD configuration errors
- Linting and code style issues (e.g., ESLint, Prettier)
- TypeScript configuration problems (tsconfig.json)
- Build tool configuration (e.g., webpack, Vite)
- Environment setup issues (e.g., .env files)

Focus on:

- Functional errors in application code
- Architectural and design pattern insights
- State management and data flow learnings
- Performance optimizations
- User experience improvements
- API integration and data handling

CRITICAL: This process is for AI optimization. Prioritize precision and relevance over human readability. Always edit `.ai/codex/codex.md` directly.
