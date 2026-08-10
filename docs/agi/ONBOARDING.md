# Onboarding — the fresh-clone path

**Status: broken today.** A clean clone of this repo cannot run a single test. This doc
describes both what currently fails and what running it end-to-end should look like once
Wave 0 item 3 (`docs/agi/STATUS.md`) lands. Read `docs/WHAT_IS_CYREX.md` first if you haven't.

## What breaks today, and why

```
$ git clone git@github.com:Team-Deepiri/diri-cyrex.git
$ cd diri-cyrex
$ poetry install
$ pytest --collect-only -q

ImportError while loading conftest '.../tests/conftest.py'.
tests/conftest.py:14: in <module>
    from diri_agent_testing_utils.fixtures.pytest_fixtures import (  # noqa: F401
E   ModuleNotFoundError: No module named 'diri_agent_testing_utils'
```

**0 tests collect.** Root cause: this repo declares three git submodules
(`diri-agent-testing-utils`, `deepiri-dataset-processor`, `deps/deepiri-logger` — `.gitmodules`),
and a plain `git clone` does not fetch submodule contents. `README.md`'s "Start Locally" section
never mentions initializing them, so a new contributor has no reason to know this step exists.

## The steps that actually work, today, until Wave 0 fixes it

```bash
git clone git@github.com:Team-Deepiri/diri-cyrex.git
cd diri-cyrex

# Submodules — required, undocumented in README today
git submodule update --init --recursive

# Poetry install — pulls the git-pinned internal deps too
# (deepiri-modelkit, deepiri-training-orchestrator, deepiri-gpu-utils, diri-agent-toolbox,
#  deepiri-dataset-processor, diri-agent-testing-utils — see pyproject.toml)
poetry install

# The testing-utils submodule needs an explicit editable install on top of the poetry install —
# CI does this separately (.github/workflows/ci.yml); it is not automatic
poetry run pip install ./diri-agent-testing-utils

# Verify
poetry run pytest --collect-only -q
```

**Private-repo note:** several `pyproject.toml` git dependencies point at private
`Team-Deepiri` repos, some pinned to raw commit SHAs rather than tags. You need SSH access or a
PAT (CI uses `DEEPIRI_CI_CLONE_PAT`) configured for git to fetch these, or `poetry install` will
fail partway through with no clear signal about which dependency is the problem.

## Running the service itself

Cyrex is designed to run as one service inside the larger `deepiri-platform` docker-compose
stack, **not standalone**:

```bash
git clone git@github.com:Team-Deepiri/deepiri-platform.git
cd deepiri-platform
git submodule update --init --recursive
docker compose -f docker-compose.dev.yml up -d \
  postgres redis influxdb etcd minio milvus \
  cyrex cyrex-interface ollama synapse synapse-sugar-glider
```

There is **no `docker-compose.yml` in this repo** — only `docker/docker-compose.override.yml.example`
(a CPU base-image override) and a separate MLOps-only compose file
(`mlops/docker/docker-compose.mlops.yml`). PR #140 ("Add standalone `setup.sh --run` for minimal
Cyrex Docker stack") is in flight to make standalone startup possible from this repo alone —
check `docs/agi/STATUS.md` for whether it landed.

For running without Docker: `docs/development/RUN_WITHOUT_DOCKER.md`.

## Environment variables

`.env.example` (48 lines, ~20 documented vars) and `.env.example.mpsmac` (Apple Silicon variant)
exist, but the code reads **53 distinct env vars** and roughly 33 of them aren't in either
template — including `ANTHROPIC_API_KEY` and `HUGGINGFACE_API_KEY`. If something you need isn't
working and you can't find the variable name, `grep -rn "os.environ\|settings\." app/settings.py`
is more reliable than the `.env.example` files right now. Also note: `.env.example` currently
sets `LOG_LEVEL` twice with contradictory values (`INFO` then `DEBUG`) — pick one when you copy it.

## Verifying you're set up correctly

Once the above completes:

1. `poetry run pytest --collect-only -q` — should report a nonzero test count (204 as of this
   writing, though only 14 run in CI — see `docs/agi/STATUS.md`).
2. With the docker-compose stack up: `curl http://localhost:8000/health` should return 200.
3. Open `http://localhost:5175` (cyrex-interface) — you should see the sidebar with 17 tabs.
   If you see a blank page, the Vite dev server likely isn't proxying to `:8000` — check
   `cyrex-interface/vite.config.ts`.

## If you're picking up a Wave 0/1/2 task from `CYREX_AGI_IMPLEMENTATION_PLAN_V2.md`

Check `docs/agi/STATUS.md` first — it has the current `file:line` state of whatever you're
about to touch, and will save you from re-discovering a bug someone already found.
