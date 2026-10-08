# CRISP experiment development

- Use this repository as the codebase for our OPSD experiments.
- Commit changes in explicit, focused commits and push directly to `origin/main`.
  Do not create pull requests unless the user explicitly asks for one. Preserve
  existing commits; do not squash or force-push to publish changes.
- Miles is vendored as ordinary tracked source under `miles/`. Make experiment
  changes directly in that tree and commit them in this repository. Do not turn
  it into a submodule or create a nested Git repository.
- `miles/CRISP_UPSTREAM.json` records the imported Miles revision. Preserve the
  upstream license and notices. Keep later experiment changes separate from
  the baseline import so they can be reviewed against that revision.
- Read `miles/AGENTS.md` and the applicable `miles/.claude/rules/` documents before
  editing Miles. Keep framework changes local to the scoring, data, or loss
  component that owns the behavior.
- The original paper implementation remains under `workspace/`. Keep new Miles
  work clearly identified; do not present new results as a paper reproduction
  without matching and validating that protocol.
- Use HTML for new explanatory documents and reports. Conventional README and
  instruction files may remain Markdown.
- The user requested experiment-plan review before a full run. Do not launch
  full experiments until that review is explicitly approved. Approval of a
  bounded pilot does not authorize a larger run.
- Keep credentials, private machine identifiers, model checkpoints, and run
  outputs out of this public repository.
