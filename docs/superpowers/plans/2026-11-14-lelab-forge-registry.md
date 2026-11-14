# LeLab Forge — CoreWeave Registry (replace HF Setup) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Setup stops asking for Hugging Face sign-in and instead verifies the CoreWeave / W&B Model Registry; Evaluate, Orders, and Improve resolve and promote checkpoints through that Registry.

**Architecture:** Keep the existing collection helpers in `lelab/forge/registry.py` (`collection_name`, `registry_path`) and the train-time `candidate` link in `WandbLogger.finish`. Add a small `RegistryStore` API client (W&B Public API under the hood) for list / resolve / download / promote. Expose Registry status on `/forge-status`, replace the Setup HF row with a Registry checklist, and resolve `candidate` / `production` aliases to a local `pretrained_model` directory for rollouts. HF Hub remains available for optional dataset/policy push elsewhere; it is not part of Setup.

**Tech Stack:** Python 3.12, `wandb.Api`, FastAPI, React + Vitest, existing Forge config (`WANDB_API_KEY`, `WANDB_ENTITY`, `WANDB_PROJECT`)

## Global Constraints

- Secrets (`WANDB_API_KEY`, `CWSANDBOX_API_KEY`) never appear in responses, log lines, or logger calls.
- Registry paths stay `"{entity}/wandb-registry-model/{collection}"` via `registry_path` (already used by `WandbLogger`).
- Collections: `so101-kitting-smolvla`, `so101-kitting-act-<skill>` with aliases `candidate` and `production` (design §5).
- Forge off → Registry features return clear 400s; upstream HF Hub paths are untouched in this plan.
- Unit tests mock `wandb.Api`; no live Forge calls in default pytest.

---

### File map

| File | Responsibility |
|---|---|
| `lelab/forge/registry.py` | Names + `RegistryStore` (list, resolve, download, promote) |
| `lelab/utils/forge_config.py` | `forge_status_dict` gains `registry` block |
| `lelab/server.py` | `POST /registry/promote`; status already via `/forge-status` |
| `frontend/src/pages/Setup.tsx` | Drop HF row; show Registry status |
| `frontend/src/components/setup/RegistryStatus.tsx` | Present collections + aliases |
| `lelab/evaluate.py` / rollout callers | Resolve alias → local checkpoint dir before trial |
| `tests/test_forge_registry.py` | Expand with store tests |
| `tests/test_forge_config.py` | Status payload includes registry |
| `frontend/src/pages/Setup.test.tsx` | Assert no HF Setup row; Registry copy present |

---

### Task 1: RegistryStore — resolve, download, promote, list

**Files:**
- Modify: `lelab/forge/registry.py`
- Modify: `tests/test_forge_registry.py`

**Interfaces:**
- Consumes: `collection_name`, `registry_path`, `load_forge_config`, `forge_configured`
- Produces:
  - `class RegistryStore:` with `__init__(self, api=None)`
  - `list_collections(self, entity: str) -> list[dict]` each `{name, aliases: list[str]}`
  - `resolve(self, entity: str, collection: str, alias: str) -> str` artifact path like `{registry_path}:{alias}`
  - `download(self, entity: str, collection: str, alias: str, root: Path | None = None) -> Path` local dir containing `pretrained_model` or weights
  - `promote(self, entity: str, collection: str, *, from_alias: str = "candidate", to_alias: str = "production") -> dict` with `{collection, from_alias, to_alias, version}`
  - Raises `ValueError` when Forge is not configured or the alias is missing

- [ ] **Step 1: Write the failing tests**

```python
class FakeArtifact:
    def __init__(self, name: str, aliases: list[str], version: str = "v3"):
        self.name = name
        self.aliases = list(aliases)
        self.version = version
        self._saved = False

    def download(self, root=None):
        target = Path(root or ".") / "artifact"
        target.mkdir(parents=True, exist_ok=True)
        (target / "pretrained_model").mkdir(exist_ok=True)
        (target / "pretrained_model" / "config.json").write_text("{}")
        return str(target)

    def save(self):
        self._saved = True


class FakeApi:
    def __init__(self, artifacts: dict[str, FakeArtifact]):
        self.artifacts = artifacts

    def artifact(self, name: str):
        if name not in self.artifacts:
            raise ValueError(f"artifact not found: {name}")
        return self.artifacts[name]


def test_resolve_builds_the_registry_alias_path():
    store = RegistryStore(api=FakeApi({}))
    assert store.resolve("acme", "so101-kitting-smolvla", "candidate") == (
        "acme/wandb-registry-model/so101-kitting-smolvla:candidate"
    )


def test_download_returns_the_pretrained_model_parent(tmp_path):
    art = FakeArtifact("acme/wandb-registry-model/so101-kitting-smolvla:candidate", ["candidate"])
    store = RegistryStore(api=FakeApi({art.name: art}))
    path = store.download("acme", "so101-kitting-smolvla", "candidate", root=tmp_path)
    assert (path / "pretrained_model" / "config.json").is_file()


def test_promote_moves_production_onto_the_candidate_version(monkeypatch):
    monkeypatch.setenv("WANDB_API_KEY", "k")
    monkeypatch.setenv("WANDB_ENTITY", "acme")
    candidate = FakeArtifact(
        "acme/wandb-registry-model/so101-kitting-smolvla:candidate",
        ["candidate", "latest"],
        version="v3",
    )
    store = RegistryStore(api=FakeApi({candidate.name: candidate}))
    result = store.promote("acme", "so101-kitting-smolvla")
    assert result == {
        "collection": "so101-kitting-smolvla",
        "from_alias": "candidate",
        "to_alias": "production",
        "version": "v3",
    }
    assert "production" in candidate.aliases
    assert candidate._saved is True


def test_list_collections_reports_aliases_for_known_names(monkeypatch):
    monkeypatch.setenv("WANDB_API_KEY", "k")
    monkeypatch.setenv("WANDB_ENTITY", "acme")
    arts = {
        "acme/wandb-registry-model/so101-kitting-smolvla:latest": FakeArtifact(
            "acme/wandb-registry-model/so101-kitting-smolvla:latest",
            ["latest", "candidate"],
        ),
    }
    store = RegistryStore(api=FakeApi(arts))
    rows = store.list_collections("acme")
    smol = next(r for r in rows if r["name"] == "so101-kitting-smolvla")
    assert "candidate" in smol["aliases"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/ibaltruschat/Documents/01_repos/leLab && .venv/bin/pytest tests/test_forge_registry.py -q`
Expected: FAIL — `RegistryStore` is not defined

- [ ] **Step 3: Implement `RegistryStore` in `lelab/forge/registry.py`**

```python
class RegistryStore:
    """W&B / CoreWeave Model Registry client. Inject `api` in tests."""

    KNOWN = (
        "so101-kitting-smolvla",
        "so101-kitting-act-bolt",
        "so101-kitting-act-bracket",
        "so101-kitting-act-gear",
    )

    def __init__(self, api=None):
        self._api = api

    def _client(self):
        if self._api is not None:
            return self._api
        import wandb

        from lelab.utils.forge_config import forge_configured, load_forge_config

        cfg = load_forge_config()
        if not forge_configured(cfg):
            raise ValueError("Registry needs WANDB_API_KEY and WANDB_ENTITY")
        return wandb.Api(api_key=cfg.api_key)

    def resolve(self, entity: str, collection: str, alias: str) -> str:
        return f"{registry_path(entity, collection)}:{alias}"

    def download(self, entity: str, collection: str, alias: str, root: Path | None = None) -> Path:
        api = self._client()
        artifact = api.artifact(self.resolve(entity, collection, alias))
        downloaded = Path(artifact.download(root=str(root) if root else None))
        pretrained = downloaded / "pretrained_model"
        return downloaded if pretrained.is_dir() else downloaded

    def promote(
        self,
        entity: str,
        collection: str,
        *,
        from_alias: str = "candidate",
        to_alias: str = "production",
    ) -> dict:
        api = self._client()
        artifact = api.artifact(self.resolve(entity, collection, from_alias))
        if to_alias not in artifact.aliases:
            artifact.aliases.append(to_alias)
            artifact.save()
        return {
            "collection": collection,
            "from_alias": from_alias,
            "to_alias": to_alias,
            "version": getattr(artifact, "version", None),
        }

    def list_collections(self, entity: str) -> list[dict]:
        api = self._client()
        rows = []
        for name in self.KNOWN:
            aliases: list[str] = []
            for probe in ("latest", "candidate", "production"):
                try:
                    art = api.artifact(self.resolve(entity, name, probe))
                    for a in getattr(art, "aliases", []) or []:
                        if a not in aliases:
                            aliases.append(a)
                except Exception:
                    continue
            rows.append({"name": name, "aliases": aliases, "linked": bool(aliases)})
        return rows
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_forge_registry.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add lelab/forge/registry.py tests/test_forge_registry.py
git commit -m "$(cat <<'EOF'
feat: add RegistryStore for resolve, download, and promote

EOF
)"
```

---

### Task 2: Expose Registry on `/forge-status` and add promote route

**Files:**
- Modify: `lelab/utils/forge_config.py` (`forge_status_dict`)
- Modify: `lelab/server.py`
- Modify: `tests/test_forge_config.py`

**Interfaces:**
- Consumes: `RegistryStore.list_collections`
- Produces: `/forge-status` includes
  ```json
  "registry": {
    "ready": true,
    "collections": [{"name": "...", "aliases": ["candidate"], "linked": true}]
  }
  ```
  When Forge is off: `"registry": {"ready": false, "collections": []}`.
  `POST /registry/promote` body `{"policy_family":"smolvla"|"act","skill": null|"bolt",...}` → promote result. Forge off → 400.

- [ ] **Step 1: Write the failing tests**

```python
def test_forge_status_includes_empty_registry_when_unconfigured(monkeypatch):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.delenv("WANDB_ENTITY", raising=False)
    from fastapi.testclient import TestClient
    from lelab.server import app

    body = TestClient(app).get("/forge-status").json()
    assert body["registry"] == {"ready": False, "collections": []}


def test_promote_endpoint_rejects_when_forge_is_off(monkeypatch):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.delenv("WANDB_ENTITY", raising=False)
    from fastapi.testclient import TestClient
    from lelab.server import app

    response = TestClient(app).post(
        "/registry/promote",
        json={"policy_family": "smolvla", "skill": None},
    )
    assert response.status_code == 400
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_forge_config.py::test_forge_status_includes_empty_registry_when_unconfigured tests/test_forge_config.py::test_promote_endpoint_rejects_when_forge_is_off -q`
Expected: FAIL — missing `registry` key / route

- [ ] **Step 3: Extend `forge_status_dict` and add the route**

In `forge_status_dict`, after building the base dict:

```python
registry_block = {"ready": False, "collections": []}
if forge_configured(resolved):
    try:
        from lelab.forge.registry import RegistryStore

        collections = RegistryStore().list_collections(resolved.entity)
        registry_block = {"ready": True, "collections": collections}
    except Exception:
        registry_block = {"ready": False, "collections": []}
payload["registry"] = registry_block
```

In `server.py`:

```python
class RegistryPromoteBody(BaseModel):
    policy_family: Literal["smolvla", "act"]
    skill: str | None = None
    from_alias: str = "candidate"
    to_alias: str = "production"


@app.post("/registry/promote")
def promote_registry_alias(body: RegistryPromoteBody):
    from lelab.forge.registry import RegistryStore, collection_name
    from lelab.utils.forge_config import forge_configured, load_forge_config

    cfg = load_forge_config()
    if not forge_configured(cfg):
        raise HTTPException(status_code=400, detail="Registry needs WANDB_API_KEY and WANDB_ENTITY")
    try:
        collection = collection_name(body.policy_family, body.skill)
        return RegistryStore().promote(
            cfg.entity,
            collection,
            from_alias=body.from_alias,
            to_alias=body.to_alias,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
```

Update existing `/forge-status` assertions that compare full bodies so they allow or assert the new `registry` key.

- [ ] **Step 4: Run tests**

Run: `.venv/bin/pytest tests/test_forge_config.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add lelab/utils/forge_config.py lelab/server.py tests/test_forge_config.py
git commit -m "$(cat <<'EOF'
feat: expose Registry status and promote endpoint

EOF
)"
```

---

### Task 3: Setup UI — replace Hugging Face with Registry

**Files:**
- Create: `frontend/src/components/setup/RegistryStatus.tsx`
- Modify: `frontend/src/pages/Setup.tsx`
- Modify: `frontend/src/pages/Setup.test.tsx`
- Modify: `frontend/src/components/setup/ForgeCredentialsForm.tsx` only if status typing needs `registry`

**Interfaces:**
- Consumes: `/forge-status`.registry
- Produces: Setup checklist without "Sign in to Hugging Face". New row "Model Registry" listing each known collection with linked aliases (or "not linked yet"). When `registry.ready` is false, detail says credentials are required (same Forge form above).

- [ ] **Step 1: Write the failing test**

```tsx
it("shows Model Registry instead of Hugging Face on Setup", async () => {
  // fetch mock returns forge_configured true and registry.ready true with one collection
  renderSetup();
  expect(await screen.findByText("Model Registry")).toBeInTheDocument();
  expect(screen.queryByText(/Sign in to Hugging Face/)).not.toBeInTheDocument();
  expect(screen.getByText(/so101-kitting-smolvla/)).toBeInTheDocument();
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd frontend && npm test -- --run src/pages/Setup.test.tsx`
Expected: FAIL — HF row still present / Model Registry missing

- [ ] **Step 3: Implement `RegistryStatus` and update Setup**

```tsx
// RegistryStatus.tsx — presentational
export function RegistryStatus({
  registry,
}: {
  registry: { ready: boolean; collections: { name: string; aliases: string[]; linked: boolean }[] } | null;
}) {
  if (!registry?.ready) {
    return (
      <p className="text-sm text-muted-foreground">
        Save W&amp;B credentials above to read the Model Registry.
      </p>
    );
  }
  if (!registry.collections.length) {
    return <p className="text-sm text-muted-foreground">No collections reported yet.</p>;
  }
  return (
    <ul className="mt-2 space-y-1 text-sm text-muted-foreground">
      {registry.collections.map((c) => (
        <li key={c.name}>
          <span className="font-medium text-foreground">{c.name}</span>
          {c.linked ? `: ${c.aliases.join(", ") || "linked"}` : ": not linked yet"}
        </li>
      ))}
    </ul>
  );
}
```

In `Setup.tsx`, remove the HF `<Link to="/">…Sign in to Hugging Face…</Link>` list item. Add:

```tsx
<li className="rounded-md border border-border p-4">
  <span className="font-bold">Model Registry</span>
  <RegistryStatus registry={forgeStatus?.registry ?? null} />
</li>
```

Extend `ForgeStatus` type with optional `registry`.

- [ ] **Step 4: Run tests**

Run: `npm test -- --run src/pages/Setup.test.tsx`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add frontend/src/pages/Setup.tsx frontend/src/pages/Setup.test.tsx frontend/src/components/setup/RegistryStatus.tsx frontend/src/components/setup/ForgeCredentialsForm.tsx
git commit -m "$(cat <<'EOF'
feat: replace Setup Hugging Face row with Model Registry status

EOF
)"
```

---

### Task 4: Resolve Registry aliases for evaluation rollouts

**Files:**
- Modify: `lelab/evaluate.py` (or the helper that prepares the policy path for a trial)
- Create or modify: `tests/test_forge_evaluate.py` (add resolve coverage)
- Optional helper: `lelab/forge/registry.py` → `checkpoint_dir_for(policy_family, skill, alias) -> Path`

**Interfaces:**
- Consumes: `RegistryStore.download`, `collection_name`, `load_forge_config`
- Produces: When an evaluation session starts (or on first trial), download `{entity}/wandb-registry-model/{collection}:{version}` to a cache under `~/.cache/huggingface/lerobot/forge-registry/` (or `LELAB_OUTPUT_ROOT/registry`) and point the local policy host / rollout at that `pretrained_model` directory. `version` is `candidate` or `production` from the evaluation request. Missing alias → 400 with a clear message.

- [ ] **Step 1: Write the failing test**

```python
def test_evaluation_resolves_candidate_through_the_registry(monkeypatch, tmp_path):
    # monkeypatch RegistryStore.download to return tmp_path / "pretrained_model" parent
    # start evaluation with policy_family=smolvla, version=candidate
    # assert the session stores the downloaded path / the next trial would use it
    ...
```

(Adapt to the real `EvaluationSession` fields in `lelab/evaluate.py` — set `policy_path` or equivalent.)

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_forge_evaluate.py -k resolve -q`
Expected: FAIL

- [ ] **Step 3: Implement resolve-on-start**

```python
def resolve_policy_dir(policy_family: str, skill: str | None, alias: str) -> Path:
    from lelab.forge.registry import RegistryStore, collection_name
    from lelab.utils.forge_config import load_forge_config

    cfg = load_forge_config()
    collection = collection_name(policy_family, skill if policy_family == "act" else None)
    root = Path.home() / ".cache" / "huggingface" / "lerobot" / "forge-registry"
    return RegistryStore().download(cfg.entity, collection, alias, root=root / collection / alias)
```

Call this from evaluation start (SmolVLA: `skill=None`; ACT: resolve per skill on that skill's first trial). Do not log the API key.

- [ ] **Step 4: Run tests**

Run: `.venv/bin/pytest tests/test_forge_evaluate.py -q`
Expected: PASS (existing + new)

- [ ] **Step 5: Commit**

```bash
git add lelab/evaluate.py lelab/forge/registry.py tests/test_forge_evaluate.py
git commit -m "$(cat <<'EOF'
feat: resolve evaluation policies from the Model Registry

EOF
)"
```

---

### Task 5: Promote control on Evaluate (and Orders readiness)

**Files:**
- Modify: `frontend/src/pages/Evaluate.tsx`
- Modify: `frontend/src/pages/Evaluate.test.tsx`
- Optional: Orders page copy that production comes from Registry (no HF)

**Interfaces:**
- Consumes: `POST /registry/promote`
- Produces: On Evaluate, when Forge is on, a secondary button "Promote candidate to production" for the selected policy family (and skill when ACT). Success toast shows collection + version. Orders continues to read `production` via Task 4 resolve when skill runners are wired; this task only needs the promote UI + API already from Task 2.

- [ ] **Step 1: Write the failing test**

```tsx
it("promotes candidate to production for the selected family", async () => {
  const fetchMock = vi.fn(async (url, init) => {
    if (String(url).endsWith("/registry/promote")) {
      return respond({
        collection: "so101-kitting-smolvla",
        from_alias: "candidate",
        to_alias: "production",
        version: "v3",
      });
    }
    // existing forge-status / evaluations mocks...
  });
  // render Evaluate, click Promote...
  expect(fetchMock).toHaveBeenCalledWith(
    expect.stringContaining("/registry/promote"),
    expect.objectContaining({ method: "POST" }),
  );
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npm test -- --run src/pages/Evaluate.test.tsx`
Expected: FAIL — button missing

- [ ] **Step 3: Add the Promote button**

Post `{ policy_family, skill: family === "act" ? skill : null }`. Disable when Forge is off. Do not show secret material.

- [ ] **Step 4: Run tests**

Run: `npm test -- --run src/pages/Evaluate.test.tsx src/pages/Setup.test.tsx`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add frontend/src/pages/Evaluate.tsx frontend/src/pages/Evaluate.test.tsx
git commit -m "$(cat <<'EOF'
feat: promote Registry candidate to production from Evaluate

EOF
)"
```

---

## Out of scope (explicit)

- Removing HF Hub push from recording / dataset upload flows.
- Replacing local job checkpoint browsing on the Train page.
- Creating Registry collections in the Forge UI automatically beyond what `link_artifact` already does at train finish.
- Changing `WandbLogger` candidate linking (already ships in M2).

## Self-review

1. **Spec coverage:** Design §4 ModelStore → RegistryStore (Task 1). §5 Register aliases → promote (Tasks 2, 5). §5 Evaluate version candidate/production → download (Task 4). Setup HF replacement → Task 3.
2. **Placeholders:** Task 4 test body uses `...` only where it must match live `EvaluationSession` fields — implementer fills from `lelab/evaluate.py` at execution time; no TBD features.
3. **Types:** `registry` on status, `RegistryPromoteBody`, and `RegistryStore` method names are consistent across tasks.

## Demo check

1. Restart backend from the leLab checkout with `.env` W&B credentials.
2. Setup: no Hugging Face row; Model Registry lists collections (empty aliases until a train links `candidate`).
3. After a Forge train: collection shows `candidate`.
4. Evaluate: start with `candidate`; Promote → `production` appears.
5. `/forge-status` never contains the API key.
