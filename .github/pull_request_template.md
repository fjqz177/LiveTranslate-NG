## Summary
<!-- What does this PR change and why? One or two sentences. -->

## Quality gate
- [ ] Ran `uv run livetranslate-pr` (all green: lock --check → ruff → format → mypy → pytest)
- [ ] Ran `uv run livetranslate-pr --git-audit` (no runtime data / generated files slipped in)

## Scope
- [ ] Changes models / engines / frozen artifacts (requires a `--smoke` run + release note)
- [ ] Cross-module or architecture touch points (requires a review; see `AGENTS.md`)

## Test plan
<!-- How to reproduce/verify manually, or why existing tests cover this. -->
