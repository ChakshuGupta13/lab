# Contributing to `lab`

## Repository purpose

Public code artifacts from research work. The private research tree lives
elsewhere; only curated, reusable, honestly-attributed artifacts land here.

## Two top-level kinds

| Tree | Convention | When to use |
|------|------------|-------------|
| `<domain>/<paper-anchor>/` | Paper implementations. Anchor is `<author><year>` (e.g. `sanadhya2008`). | The artifact reproduces or accompanies a specific published paper. |
| `tools/<domain>/<name>/` | Stand-alone utilities. Name is a short noun describing what the tool does. | The artifact is a reusable utility not tied to a single paper. |

A paper that ships a utility goes under `papers/`; the utility is part of
the paper's repo, not a separate `tools/` entry. Only promote to `tools/`
when the utility is genuinely independent.

## Top-level README is the index — keep it current

The top-level `README.md` is the single index of everything in this repo.
**When you add a new entry under `<domain>/` or `tools/<domain>/`, you
MUST add a corresponding row to the matching section in `README.md` in
the same commit that introduces the entry.** If the domain/section doesn't
yet exist, add it.

Row format (paper):
```markdown
| [<anchor>](<path>/) | <one-sentence description> ([upstream paper or anchor link]) |
```

Row format (tool):
```markdown
| [<name>](<path>/) | <one-sentence description of what the tool does> |
```

Same rule on removal/rename: update the table in the same commit.

## Per-entry README convention

Each entry directory contains its own `README.md` as the user-facing front
page. Expected sections (vary by entry type):

- **What this is** — one paragraph; no novelty claim unless genuinely novel.
- **Install / Quick start** — copy-pasteable commands.
- **Worked example(s)** — at least one runnable example with expected output.
- **Non-goals / disclaimers** — what the artifact does NOT do.
- **Attribution / prior art** — cite the foundational papers the artifact
  builds on. If the underlying idea is well established, say so explicitly.
- **License** — link to `LICENSE` in the entry directory.

## Public-artifact hygiene

Before committing any file into this repo, strip:

- Internal-process comments (`% verified:` audit-trail tags, `% Cited for:`
  rationale notes — these belong in the private repo's bib files only).
- Internal jargon (`Adversary`, `Scenario X`, `Phase Y`, `Thread M.N`,
  audit-log references, paper-internal milestone labels).
- Process commentary ("we initially thought... but discovered...") — public
  artifacts state the result, not the journey.
- Paths to private files (`getting_started/`, `audit-log/`, etc.).

## License

Each entry carries its own `LICENSE` file. MIT is the default; departures
must be documented per-entry.

## Commits

Conventional Commits. Scope is `<topic>/<entry-name>` (e.g.
`feat(tools/quantum/qbudget): ...`, `feat(math/gupta2026architecture): ...`).
Top-level changes (README, AGENTS.md, .gitignore, LICENSE) use scope `infra`.
