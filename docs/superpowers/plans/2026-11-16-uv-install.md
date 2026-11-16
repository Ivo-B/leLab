# Install with uv

**Goal:** Make uv the documented way to install and develop Garage, with a committed lockfile, while pip installs keep working.

**Architecture:** `pyproject.toml` stays the single source of dependencies. Dev tools move into a PEP 735 `dev` dependency group, so `uv sync` installs them. `uv.lock` pins every version for development and CI. Demo machines install the app with `uv tool install` from the fork, which builds the UI through `setup.py`. Forge credentials move out of the package directory when LeLab is not a checkout.

**Tech Stack:** uv 0.11, Python 3.12, setuptools with `nodejs-wheel-binaries`, GitHub Actions.

## Decisions

- Demo machines: `uv tool install git+https://github.com/Ivo-B/leLab`. Upgrade with `uv tool upgrade lelab`.
- Credentials: a checkout keeps the repo-root `.env`. Any other install reads and writes `$XDG_CONFIG_HOME/lelab/.env` (default `~/.config/lelab/.env`). In a checkout, values in the repo `.env` win over the user file.
- Packaging: uv first, pip still works. The LeRobot git pin stays a direct reference in `dependencies`, and the `test` and `dev` extras stay for pip users.

## Findings that drive the work

- `lelab/evaluate.py` imports `weave` when an evaluation starts, but `weave` is not a dependency and is not installed.
- `forge_config.py` reads and writes `.env` at `Path(__file__).parents[2]`, which is `site-packages` for a non-editable install. Credentials saved from Setup are lost on upgrade.
- There is no lockfile; CI runs `pip install -e ".[test]"`.
- The owner's `lelab` tool was installed from `huggingface/leLab`, then patched by hand with an editable install and test packages. `uv tool upgrade lelab` would replace the fork with upstream.

## Tasks

- [x] **pyproject:** add `weave`; add `[dependency-groups] dev` with pytest, pytest-asyncio, httpx, ruff, and pre-commit; add a `notebook` extra with `marimo`; add `.python-version` with `3.12`.
- [x] **Credentials path:** resolve the `.env` file to the repo root in a checkout and to the user config dir otherwise. Load the repo file first, then the user file, without overriding the shell. Tests cover both cases and `XDG_CONFIG_HOME`.
- [x] **Lock:** `uv lock`, then `uv sync` the local `.venv`. Commit `uv.lock`.
- [x] **CI:** `tests.yml` uses `astral-sh/setup-uv`, `uv sync --locked`, and `uv run pytest`. `quality.yml` checks that `uv.lock` matches `pyproject.toml`. Both run when `uv.lock` changes.
- [x] **Docs:** README Getting started and Connect Forge, `CLAUDE.md` commands, the `.env.template` header.
- [x] **Verify:** `uv run pytest` passes; `uv run lelab --help` works; `uv run python -c "import weave"` works.
- [x] **Owner machine:** reinstall the `lelab` tool so its receipt matches what it runs.

## Out of scope

- Publishing to PyPI.
- Changing how `setup.py` builds the frontend.
- The Space Dockerfile, which only runs upstream.
