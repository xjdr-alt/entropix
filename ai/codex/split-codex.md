# AI Codex Splitting Protocol

Invocation: @split-codex.md

## Purpose

This protocol is used to intelligently split the AI Codex when it becomes too long, grouping related Errors (E) and Learnings (L) into separate codex pages.

## Process

1. Analyze the current @codex.md file
2. Identify common themes or categories among entries
3. Group related E and L entries
4. Create new codex pages for each group
5. Update the main @codex.md with links to new pages
6. Report the new structure

## Grouping Criteria

- Project area (e.g., frontend, backend, database)
- Technology or framework (e.g., React, Node.js, Prisma)
- Concept (e.g., authentication, state management, testing)
- File type or location (e.g., configuration files, utility functions)

## File Structure

1. Main codex remains at: .ai/codex/codex.md
2. New pages: .ai/codex/[category]-codex.md

## Update Process

1. Move relevant entries to new category-specific files
2. In the main codex (.ai/codex/codex.md), replace moved entries with links to new files
3. Ensure cross-references are updated across all files

## Naming Convention

Use kebab-case for new codex file names, e.g.:

- frontend-codex.md
- database-operations-codex.md
- authentication-codex.md

## Reporting

After splitting, provide a summary of:

1. Number of new codex pages created
2. Categories identified
3. Entry distribution across new pages

IMPORTANT: Maintain the original entry IDs (E001, L001, etc.) in the new files to preserve existing references.
