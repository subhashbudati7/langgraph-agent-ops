# Deploy runbook — project 2 (langgraph-agent-ops), target 2026-11-27

Release = merge `agent/november-build` into `main` (no squash, so every
commit counts on the contribution graph) + a live demo URL.

## 1. Pre-merge verification

- `python scripts/check_release.py` — one command: ruff + pytest +
  offline demo + trace summary + run report, non-zero exit if any
  step fails.
- Every local commit on `agent/november-build` is mirrored to GitHub —
  run the mirror helper and confirm it reports "nothing to mirror".
  (The helper replays commits through the git database API, so remote
  SHAs differ from local SHAs by design; the helper's own state, not
  `git ls-remote` equality, is the mirror check.)
- Offline demo re-verified with zero trace errors:
  `python scripts/demo.py` then `python scripts/summarize_trace.py`
  shows 0 errors, and `python scripts/run_report.py` writes a report
  with `"ok": true`.
- API smoke (needs the `api` extra): `python scripts/serve_api.py`,
  then `GET /health` returns 200 and `POST /run` with the demo task
  returns 200 with two worker results and no error field.

## 2. Demo verification (needs a Docker host for the containerized demo)

This VM has no Docker, so a containerized demo run happens on a
machine that has it; the API itself is plain FastAPI + uvicorn and
the offline stack needs no key, so the live smoke above already
covers the serving path. If a Dockerfile is added before the 27th,
verify: image builds, container serves the API, `GET /health` is
200, `POST /run` returns the demo task's two results.

## 3. Merge and release

```bash
git checkout main
git merge --no-ff agent/november-build -m "release: multi-agent workflow with ops (November)"
git push origin main
```

Verify the CI workflow is green on `main`, then tag the release
(e.g. `v0.1.0`) so the ops table and README numbers are pinned to a
ref. Note: project 1 had no `main` branch at release time (its
default branch was the build branch) — check whether this repo has
one before merging, and create it at the branch tip via the GitHub
API if it does not exist.

## 4. Live demo URL — DECISION NEEDED

Hosting is not chosen yet. Candidates (prices verified October 2026
for project 1, same container-serving shape applies here):

- **Hugging Face Spaces** (Docker SDK): purpose-built for model
  demos. The free CPU tier is dead for new accounts — creating a new
  Docker Space needs PRO ($9/mo) or added credits. Still the
  recommendation, but only after he sees the real price.
- **Google Cloud Run** (warm always-on): ~$25–30/mo; no account
  decision beyond his existing Google account.
- **Render / Fly.io**: container hosting; verify current free-tier
  availability before choosing — do not assume a free tier exists.
- **Vercel**: not a fit — it doesn't run long-lived API servers.

Pick one, deploy the API, and put the URL at the top of the README.
This is the one deploy step that needs Subhash's call (account +
hosting choice), so it travels through the report channel, not a
silent run.

## 5. Report

One summary after the deploy: merge commit, CI status, live demo URL,
final test count, commit count for the project.
