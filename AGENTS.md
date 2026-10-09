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
- Public changes must not contain private identity or infrastructure information:
  personal names/emails/usernames, user-specific absolute paths, actual host or
  devbox names, hardware codenames, node/cluster/reservation identifiers, private
  addresses, internal storage endpoints, credentials, or raw operational logs.
  Use generic labels such as `worker-a` and configurable paths. Keep real values
  in ignored local configuration and inject them at runtime.
- Check both file contents and Git author/committer metadata before a public
  push. Use neutral project metadata for new commits. Do not embed a private
  identifier in a public denylist, test fixture, screenshot, or example.
- Preserve required upstream license notices and existing public scholarly
  attribution. Do not claim this named public repository is anonymous. Any
  removal of previously published Git metadata needs a separately authorized
  history-cleanup operation; a new commit cannot erase an older commit.
- Keep model checkpoints, generated traces, raw run outputs, and local runtime
  configuration out of this public repository.
- Enable W&B for future OPSD experiments using the restricted OPSD tracking
  profile. Resolve the destination from ignored local configuration and verify
  SDK authentication before reserving GPUs. Save each run URL with its private
  run record. Upload approved numeric metrics and recipe fields only; never raw
  responses, launch arguments, credentials, host metadata or source snapshots.
  Do not retrofit tracking into an already-running experiment without a request.
- Do not save recovery checkpoints for the current OPSD study. Use temporary
  local evaluation snapshots only, remove each after successful evaluation,
  and retain aggregate metrics and private response tapes. Preserve existing
  source checkpoints; restarting a branch uses its fixed warmed source.
- Give future W&B runs descriptive display names: teacher checkpoint policy,
  PI context, target construction, block, seed and attempt when available.
  Keep operational run IDs and groups stable and neutral. Derive names from
  validated recipe fields; never include personal or hardware identifiers.
