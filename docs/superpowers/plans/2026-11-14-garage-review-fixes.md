# Garage review fixes

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the defects an independent review found in the uncommitted Garage, Forge status, and CoreWeave job work, then commit all of it with a clear history.

**Architecture:** The uncommitted work lands first as one baseline commit so the fixes show as separate diffs. Weave rows record the host a trial really ran on. The Forge chips refetch when the route changes and show status as icon plus label. The compute slot stops claiming Local when nothing runs. The Hugging Face login moves from a dead header chip into the Record dialog. The tracking line is printed to stdout.

**Tech Stack:** Python 3.12, FastAPI, React, Vitest. Pytest binary: `/Users/ibaltruschat/.local/share/uv/tools/lelab/bin/python -m pytest`. Frontend tests: `cd frontend && npm test -- --run <paths>`.

Source of the findings: [whole-diff review](9ee2e22f-5197-43c7-a8f7-bb4952468516). Decisions already made by the owner: log the real host in Weave and keep the requested host as a second field; delete the dead Hugging Face chip and dialog and show the login banner inside the Record dialog.

## Global Constraints

- Work on branch `forge/garage` (branched from `origin/main` at f5c3ee6, upstream unset). Push only to `origin` (`https://github.com/Ivo-B/leLab.git`). Do not push `upstream`. Do not push to `main`. Do not push or merge in this plan. The owner asks for that separately.
- Do not commit `frontend/public/brand/` (gitignored, holds the CoreWeave artwork), `.env`, `graphify-out/`, or `.superpowers/`. Run `git status --short` before each commit and stage only the files the task lists.
- Do not start the arm. Do not call W&B, ARIA, CoreWeave, or the network from tests. Do not start the API with a fake `CWSANDBOX_API_KEY`.
- Do not wire the CoreWeave policy host. Do not touch `lelab/agent/`. Do not change calibration. Do not press Start Calibration.
- Brand: spell `CoreWeave` exactly. No exclamation points in UI copy. Status is icon plus label, never color alone. Garage stays a nickname, not an official product.
- New Python files start with the Apache-2.0 header from `tests/test_forge_config.py` lines 1 to 13. Edits keep the header already on the file.
- Commit messages use the `<type>: <description>` form, passed by HEREDOC. Do not amend. Do not skip hooks.
- Two type errors already exist on main and are not ours: `src/lib/meshLoaders.ts(64,26)` and `src/pages/Evaluate.tsx(230,22)`. Any other `tsc` error is a failure.

## Out of scope

- Starting the CoreWeave policy host for an evaluation. That is a later slice that touches the rollout on the arm. Until then Weave rows say `local`.
- The order agent tags its pick result with the requested host (`lelab/agent/executor.py:73`). A live pick does not run a rollout yet, so this is not changed here.
- A live Sandbox run. Whether the Sandbox exec stream carries stderr (so progress parsing works) is only provable live. Task 5 makes the tracking link independent of that.
- The README keeps the upstream name LeLab.

---

### Task 0: Commit the work built so far

**Files:**

- Commit: everything the review covered, as one baseline commit.

- [ ] **Step 1: Check the tree**

Run: `cd /Users/ibaltruschat/Documents/01_repos/leLab && git status --short`

Expected: the modified and new files under `frontend/`, `lelab/evaluate.py`, `lelab/server.py`, `tests/test_forge_evaluate.py`, the specs `docs/superpowers/specs/2026-11-14-garage-name-design.md` and `docs/superpowers/specs/2026-11-14-forge-auth-status-design.md`, this plan, and untracked `graphify-out/`. Nothing under `frontend/public/brand/` appears, because it is gitignored. Confirm with `git check-ignore -v frontend/public/brand/cw-dark.png frontend/public/brand/cw-light.png`.

- [ ] **Step 2: Run the suites**

Run: `/Users/ibaltruschat/.local/share/uv/tools/lelab/bin/python -m pytest tests/test_forge_evaluate.py -q` and `cd frontend && npm test -- --run`.

Expected: 35 Python tests pass, 156 frontend tests pass.

- [ ] **Step 3: Commit**

```bash
git add frontend/index.html frontend/src docs/superpowers/specs/2026-11-14-garage-name-design.md docs/superpowers/specs/2026-11-14-forge-auth-status-design.md docs/superpowers/plans/2026-11-14-garage-review-fixes.md lelab/evaluate.py lelab/server.py tests/test_forge_evaluate.py
git status --short
git commit -m "$(cat <<'EOF'
feat: name the app Garage and show Forge and CoreWeave status

The sidebar shows Garage with the CoreWeave mark for the page theme. The
status bar shows Weights & Biases and CoreWeave chips, the compute of
the running job, and whether an evaluation is open. A CoreWeave job card
names the Sandbox and links the run. GET /evaluations/current reports
the open session.
EOF
)"
```

Expected: `git status --short` shows only `?? graphify-out/` before the commit and a clean tree except `graphify-out/` after.

---

### Task 1: Log the host a trial really ran on

**Files:**

- Modify: `lelab/forge/weave_eval.py`
- Modify: `lelab/evaluate.py`
- Modify: `tests/test_forge_evaluate.py`
- Modify: `frontend/src/pages/Evaluate.tsx`
- Modify: `frontend/src/pages/Evaluate.test.tsx`
- Modify: `docs/superpowers/specs/2026-11-14-forge-auth-status-design.md`

**Interfaces:**

- Consumes: `TrialRow` from `lelab.forge.weave_eval`.
- Produces: `TrialRow.requested_policy_host: str | None = None` as the last field. Weave `output` gains `requested_policy_host`. `policy_host` in the row and in `inputs` is `evaluate.ACTUAL_POLICY_HOST`, which is `"local"`. The Evaluate note reads `The CoreWeave policy host is not wired yet. Trials run on this machine and are logged as local in Weave.`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_forge_evaluate.py`:

```python
def test_a_coreweave_request_is_logged_as_the_local_host_it_ran_on(fake_logger) -> None:
    client = TestClient(app)
    assert (
        client.post(
            "/evaluations", json={**BODY, "policy_host": "coreweave", "trials_per_skill": 1}
        ).status_code
        == 200
    )
    client.post(
        "/evaluations/current/trial",
        json=_trial_body(top_jpeg_base64="", side_jpeg_base64=""),
    )
    client.post("/evaluations/current/close", json={})

    inputs, output, _ = fake_logger.predictions[0]

    assert inputs["policy_host"] == "local"
    assert output["policy_host"] == "local"
    assert output["requested_policy_host"] == "coreweave"
```

In `frontend/src/pages/Evaluate.test.tsx`, change the `HOST_NOTE` constant to:

```ts
const HOST_NOTE =
  "The CoreWeave policy host is not wired yet. Trials run on this machine and are logged as local in Weave.";
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `/Users/ibaltruschat/.local/share/uv/tools/lelab/bin/python -m pytest tests/test_forge_evaluate.py::test_a_coreweave_request_is_logged_as_the_local_host_it_ran_on -q` and `cd frontend && npm test -- --run src/pages/Evaluate.test.tsx`.

Expected: the Python test fails with a `KeyError` or an `assert "coreweave" == "local"`. The Evaluate note test fails on the old wording.

- [ ] **Step 3: Write the minimal implementation**

In `lelab/forge/weave_eval.py`, add the last field to `TrialRow`:

```python
    side_video: str | None
    requested_policy_host: str | None = None
```

In `WeaveTrialLog.log_trial`, add to `output` after `"policy_host": row.policy_host,`:

```python
                "requested_policy_host": row.requested_policy_host,
```

In `lelab/evaluate.py`, after `MAX_TRIALS = 50` add:

```python
# Nothing starts the CoreWeave policy host yet, so every rollout runs on this machine.
ACTUAL_POLICY_HOST = "local"
```

In `EvaluationSession._log_pending`, replace `policy_host=self.request.policy_host,` with:

```python
                    policy_host=ACTUAL_POLICY_HOST,
                    requested_policy_host=self.request.policy_host,
```

In `frontend/src/pages/Evaluate.tsx`, change `COREWEAVE_HOST_NOTE` to the string in the Interfaces list.

In the spec `2026-11-14-forge-auth-status-design.md`, change `are tagged CoreWeave in Weave` to `are logged as local in Weave`, and add one sentence: `Weave rows record the host a trial ran on and keep the requested host as requested_policy_host.`

- [ ] **Step 4: Run the tests to verify they pass**

Run: `/Users/ibaltruschat/.local/share/uv/tools/lelab/bin/python -m pytest tests/test_forge_evaluate.py tests/test_forge_weave_eval.py -q` and `cd frontend && npm test -- --run src/pages/Evaluate.test.tsx`.

Expected: all pass. If an existing test fails because it compares a whole `output` dict, add `"requested_policy_host": None` (or the requested value) to that expectation. Do not loosen the new test.

- [ ] **Step 5: Commit**

```bash
git add lelab/forge/weave_eval.py lelab/evaluate.py tests/test_forge_evaluate.py frontend/src/pages/Evaluate.tsx frontend/src/pages/Evaluate.test.tsx docs/superpowers/specs/2026-11-14-forge-auth-status-design.md
git commit -m "$(cat <<'EOF'
fix: log the host a trial really ran on in Weave

A CoreWeave request still runs the trial on this machine, so the row now
says local and keeps the request as requested_policy_host.
EOF
)"
```

---

### Task 2: Keep the Forge chips fresh and readable without color

**Files:**

- Modify: `frontend/src/components/shell/ForgeStatusChips.tsx`
- Modify: `frontend/src/components/shell/ForgeStatusChips.test.tsx`

**Interfaces:**

- Consumes: `GET /forge-status` with `forge_configured`, `sandbox_configured`, `entity`.
- Produces: the chips refetch `/forge-status` once per route change. A ready chip is `role="status"` with a `CircleCheck` icon and its label. A chip that is not set up is a button with a `CircleAlert` icon and the label `Weights & Biases · not set up` or `CoreWeave · not set up`. Aria labels stay as they are (`Weights & Biases connected as <entity>`, `CoreWeave ready`, `Weights & Biases not configured`, `CoreWeave not configured`, and `CoreWeave · N run(s)`). The not-set-up chips use muted styling, not amber.

- [ ] **Step 1: Write the failing tests**

In `frontend/src/components/shell/ForgeStatusChips.test.tsx`:

Add `Link` to the router import: `import { Link, MemoryRouter } from "react-router-dom";`.

Replace the final test, `treats a failed status request as not configured and asks only once`, with these two:

```tsx
  it("treats a failed status request as not configured", async () => {
    vi.stubGlobal("fetch", vi.fn((_url: string) => respond({}, false)));
    renderChips();
    expect(await screen.findByRole("button", { name: "Weights & Biases not configured" })).toHaveTextContent(
      "Weights & Biases · not set up",
    );
    expect(screen.getByRole("button", { name: "CoreWeave not configured" })).toHaveTextContent(
      "CoreWeave · not set up",
    );
  });

  it("asks again after a route change so saved keys show up", async () => {
    const fetchMock = vi
      .fn((_url: string) => respond({ forge_configured: false, sandbox_configured: false }))
      .mockImplementationOnce((_url: string) => respond({ forge_configured: false, sandbox_configured: false }))
      .mockImplementation((_url: string) =>
        respond({ forge_configured: true, sandbox_configured: true, entity: "team" }),
      );
    vi.stubGlobal("fetch", fetchMock);
    render(
      <MemoryRouter>
        <ApiProvider>
          <ForgeStatusChips />
          <Link to="/next">Next page</Link>
        </ApiProvider>
      </MemoryRouter>,
    );
    expect(await screen.findByRole("button", { name: "Weights & Biases not configured" })).toBeInTheDocument();

    fireEvent.click(screen.getByRole("link", { name: "Next page" }));

    expect(await screen.findByLabelText("Weights & Biases connected as team")).toBeInTheDocument();
    expect(screen.getByLabelText("CoreWeave ready")).toBeInTheDocument();
    expect(fetchMock).toHaveBeenCalledTimes(2);
    expect(String(fetchMock.mock.calls[0][0])).toBe(`${BASE}/forge-status`);
  });

  it("marks a ready chip as a status", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(() => respond({ forge_configured: true, sandbox_configured: true, entity: "team" })),
    );
    renderChips();
    expect(await screen.findByRole("status", { name: "CoreWeave ready" })).toBeInTheDocument();
  });
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd frontend && npm test -- --run src/components/shell/ForgeStatusChips.test.tsx`

Expected: the not-set-up text, the route-change refetch, and the `status` role tests fail.

- [ ] **Step 3: Write the minimal implementation**

Replace `frontend/src/components/shell/ForgeStatusChips.tsx` with:

```tsx
import { useEffect, useState } from "react";
import { CircleAlert, CircleCheck } from "lucide-react";
import { Link, useLocation } from "react-router-dom";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { useApi } from "@/contexts/ApiContext";

const WANDB_HINT = "Add WANDB_API_KEY and WANDB_ENTITY on Setup.";
const COREWEAVE_HINT = "Add CWSANDBOX_API_KEY and CWSANDBOX_TRAIN_IMAGE on Setup.";

type ForgeBody = {
  forge_configured?: unknown;
  sandbox_configured?: unknown;
  entity?: unknown;
};

type Which = "wandb" | "coreweave";

const chipClass = "inline-flex items-center gap-2 rounded-full border px-3 py-1 text-xs";
const readyClass = `${chipClass} border-border bg-card text-foreground`;
const missingClass = `${chipClass} border-border bg-card text-muted-foreground hover:text-foreground`;

function runsLabel(count: number): string {
  if (count <= 0) return "CoreWeave";
  return `CoreWeave · ${count} ${count === 1 ? "run" : "runs"}`;
}

export function ForgeStatusChips({ coreweaveRuns = 0 }: { coreweaveRuns?: number }) {
  const { baseUrl, fetchWithHeaders } = useApi();
  const { pathname } = useLocation();
  const [body, setBody] = useState<ForgeBody | "error" | null>(null);
  const [open, setOpen] = useState<Which | null>(null);

  // Ask again on every route change so keys saved on Setup show up without a reload.
  useEffect(() => {
    let cancelled = false;
    fetchWithHeaders(`${baseUrl}/forge-status`)
      .then(async (response) => {
        if (!response.ok) throw new Error(`forge-status ${response.status}`);
        return (await response.json()) as ForgeBody;
      })
      .then((next) => {
        if (!cancelled) setBody(next);
      })
      .catch(() => {
        if (!cancelled) setBody("error");
      });
    return () => {
      cancelled = true;
    };
  }, [baseUrl, fetchWithHeaders, pathname]);

  if (body === null) {
    return <span className="text-xs text-muted-foreground">Checking Forge…</span>;
  }

  const status = body === "error" ? {} : body;
  const entity = typeof status.entity === "string" ? status.entity : "";
  const wandbReady = status.forge_configured === true && entity.length > 0;
  const coreweaveReady = status.sandbox_configured === true;

  return (
    <>
      {wandbReady ? (
        <span
          role="status"
          className={readyClass}
          aria-label={`Weights & Biases connected as ${entity}`}
        >
          <CircleCheck size={14} aria-hidden />
          {entity}
        </span>
      ) : (
        <button
          type="button"
          onClick={() => setOpen("wandb")}
          className={missingClass}
          aria-label="Weights & Biases not configured"
        >
          <CircleAlert size={14} aria-hidden />
          Weights & Biases · not set up
        </button>
      )}
      {coreweaveReady ? (
        <span
          role="status"
          className={readyClass}
          aria-label={coreweaveRuns > 0 ? runsLabel(coreweaveRuns) : "CoreWeave ready"}
        >
          <CircleCheck size={14} aria-hidden />
          {runsLabel(coreweaveRuns)}
        </span>
      ) : (
        <button
          type="button"
          onClick={() => setOpen("coreweave")}
          className={missingClass}
          aria-label="CoreWeave not configured"
        >
          <CircleAlert size={14} aria-hidden />
          CoreWeave · not set up
        </button>
      )}
      <Dialog open={open !== null} onOpenChange={(next) => { if (!next) setOpen(null); }}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>{open === "coreweave" ? "CoreWeave" : "Weights & Biases"}</DialogTitle>
            <DialogDescription>{open === "coreweave" ? COREWEAVE_HINT : WANDB_HINT}</DialogDescription>
          </DialogHeader>
          <Link to="/setup" className="text-sm font-medium text-primary underline">
            Open Setup
          </Link>
        </DialogContent>
      </Dialog>
    </>
  );
}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd frontend && npm test -- --run src/components/shell` and `npx tsc --noEmit -p tsconfig.app.json`.

Expected: all shell tests pass. `tsc` shows only the two pre-existing errors. If `StatusBar.test.tsx` fails on a label such as `CoreWeave · 2 runs`, the label did not change, so investigate before editing the test.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/shell/ForgeStatusChips.tsx frontend/src/components/shell/ForgeStatusChips.test.tsx
git commit -m "$(cat <<'EOF'
fix: refresh the Forge chips on navigation and show status with an icon

Saved keys now show without a reload. A chip that is not set up says so
in words instead of a dot color.
EOF
)"
```

---

### Task 3: Stop the compute slot claiming Local when nothing runs

**Files:**

- Modify: `frontend/src/components/shell/StatusBar.tsx`
- Modify: `frontend/src/components/shell/StatusBar.test.tsx`
- Modify: `frontend/src/components/shell/AppShell.test.tsx`

**Interfaces:**

- Consumes: `/jobs` rows with `runner`, `/evaluations/current` with `open` and `policy_runs_on`.
- Produces: the compute slot reads `CoreWeave` or `Hugging Face` for a running training job on that runner, `Local` for any other running job, else for an open evaluation `Local` when `policy_runs_on` is `this machine` and `CoreWeave` otherwise, else `Idle`.

- [ ] **Step 1: Write the failing tests**

In `frontend/src/components/shell/StatusBar.test.tsx`:

Replace the test `shows Local and No evaluation when nothing runs` with:

```tsx
  it("shows Idle and No evaluation when nothing runs", async () => {
    vi.stubGlobal("fetch", vi.fn(() => respond({})));
    renderBar();
    expect(await screen.findByText("Idle")).toBeInTheDocument();
    expect(screen.getByText("No evaluation")).toBeInTheDocument();
    expect(screen.queryByText("Local")).not.toBeInTheDocument();
  });

  it("shows Local for a running local job", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn((url: string) =>
        respond(url === `${BASE}/jobs` ? { jobs: [{ name: "so101-act", state: "running", runner: "local" }] } : {}),
      ),
    );
    renderBar();
    expect(await screen.findByText("Local")).toBeInTheDocument();
  });
```

Replace the test `shows Evaluation open while a session is open` with:

```tsx
  it("shows Evaluation open and where the policy runs while a session is open", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn((url: string) =>
        respond(
          url === `${BASE}/evaluations/current` ? { open: true, policy_runs_on: "this machine" } : {},
        ),
      ),
    );
    renderBar();
    expect(await screen.findByText("Evaluation open")).toBeInTheDocument();
    expect(screen.getByText("Local")).toBeInTheDocument();
  });
```

In `frontend/src/components/shell/AppShell.test.tsx`, change `expect(screen.getByText("Local")).toBeInTheDocument();` to `expect(screen.getByText("Idle")).toBeInTheDocument();`.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd frontend && npm test -- --run src/components/shell/StatusBar.test.tsx src/components/shell/AppShell.test.tsx`

Expected: the Idle tests fail because the slot still says Local.

- [ ] **Step 3: Write the minimal implementation**

In `frontend/src/components/shell/StatusBar.tsx`:

Change the type: `type Compute = "Idle" | "Local" | "CoreWeave" | "Hugging Face";`.

In `IDLE`, change `compute: "Local",` to `compute: "Idle",`.

Add `policy_runs_on?: unknown;` to `StatusBody`.

Replace the `compute:` line in the returned object with:

```ts
    compute: running
      ? (running.runner && COMPUTE[running.runner]) || "Local"
      : evaluation.open === true
        ? evaluation.policy_runs_on === "this machine"
          ? "Local"
          : "CoreWeave"
        : "Idle",
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd frontend && npm test -- --run src/components/shell` and `npx tsc --noEmit -p tsconfig.app.json`.

Expected: all pass, `tsc` shows only the two pre-existing errors.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/shell/StatusBar.tsx frontend/src/components/shell/StatusBar.test.tsx frontend/src/components/shell/AppShell.test.tsx
git commit -m "$(cat <<'EOF'
fix: say Idle when no compute is in use

The compute slot follows the running job or the open evaluation and no
longer claims Local when nothing runs.
EOF
)"
```

---

### Task 4: Move the Hugging Face login into the Record dialog

**Files:**

- Delete: `frontend/src/components/landing/HfAuthChip.tsx`
- Delete: `frontend/src/components/landing/HfAuthDialog.tsx`
- Modify: `frontend/src/components/landing/HfAuthBanner.tsx`
- Modify: `frontend/src/components/landing/RecordingModal.tsx`
- Create: `frontend/src/components/landing/HfAuthBanner.test.tsx`

**Interfaces:**

- Consumes: `useHfAuth()` from `@/contexts/HfAuthContext` with `auth.status` of `loading`, `authenticated`, or `unauthenticated`.
- Produces: `HfAuthBanner` takes an optional `title` prop. Default title stays `Hugging Face access required for cloud training`. The explanatory line reads `... with Write access (so datasets and trained policies can upload to your account), then paste it below.` `RecordingModal` shows `<HfAuthBanner title="Log in to Hugging Face to set the repository owner" />` when `auth.status === "unauthenticated"` and a dataset name is set, in place of the one-line amber text.

- [ ] **Step 1: Write the failing test**

Create `frontend/src/components/landing/HfAuthBanner.test.tsx`. Read `frontend/src/contexts/HfAuthContext.tsx` first to see what `/hf-auth-status` returns for an unauthenticated user and how `HfAuthProvider` reads it. Then write tests that render `HfAuthBanner` inside `ApiProvider` and `HfAuthProvider` with `fetch` stubbed:

- With an unauthenticated status the default title `Hugging Face access required for cloud training` shows.
- With the `title` prop set to `Log in to Hugging Face to set the repository owner` that text shows and the default title does not.
- With an authenticated status the banner renders nothing.

Do not call the network. Stub `fetch` for `${BASE}/hf-auth-status` only.

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd frontend && npm test -- --run src/components/landing/HfAuthBanner.test.tsx`

Expected: the `title` prop test fails because the prop does not exist.

- [ ] **Step 3: Write the minimal implementation**

In `HfAuthBanner.tsx`, change the component signature to take `{ title = "Hugging Face access required for cloud training" }: { title?: string }`, render `{title}` in place of the fixed heading text, and change `(so trained policies can upload to your account)` to `(so datasets and trained policies can upload to your account)`.

In `RecordingModal.tsx`, import `HfAuthBanner` from `./HfAuthBanner` and replace the unauthenticated `<p className="text-xs text-amber-400/80">Log in to Hugging Face to set the repository owner.</p>` with `<HfAuthBanner title="Log in to Hugging Face to set the repository owner" />`.

Delete `HfAuthChip.tsx` and `HfAuthDialog.tsx` after confirming nothing imports them: `rg -n "HfAuthChip|HfAuthDialog" frontend/src` must return no matches outside the two files themselves.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd frontend && npm test -- --run` and `npx tsc --noEmit -p tsconfig.app.json`.

Expected: all frontend tests pass, `tsc` shows only the two pre-existing errors.

- [ ] **Step 5: Commit**

```bash
git add -A frontend/src/components/landing
git commit -m "$(cat <<'EOF'
fix: offer the Hugging Face login where a Hub upload needs it

The login banner now shows in the Record dialog. The header chip and its
dialog had no remaining entry point and are removed.
EOF
)"
```

---

### Task 5: Print the tracking line to stdout

**Files:**

- Modify: `lelab/forge/tracking.py`
- Modify: the existing tracking test file that covers `WandbLogger` (find it with `rg -ln "WandbLogger" tests`)

**Interfaces:**

- Consumes: `WandbLogger.__init__` and `lelab.jobs.extract_tracking_url`.
- Produces: after `wandb.init`, the constructor prints `Track this run --> <url>` to stdout with `flush=True`, in addition to the existing `logger.info`. LeRobot's console handler writes to stderr, and the Sandbox runner reads only stdout, so this guarantees the line reaches `extract_tracking_url` either way.

- [ ] **Step 1: Write the failing test**

In the existing test file for `WandbLogger`, reuse its fake `wandb` module and add a test that builds a `WandbLogger`, captures stdout with `capsys`, and asserts both that `"Track this run --> https://wandb.ai/team/so101-kitting/runs/abc"` appears in `capsys.readouterr().out` and that `lelab.jobs.extract_tracking_url(<that line>)` returns the URL. Use the fake run URL the file already uses, or `https://wandb.ai/team/so101-kitting/runs/abc` if it has none. Do not call W&B.

- [ ] **Step 2: Run the test to verify it fails**

Run: `/Users/ibaltruschat/.local/share/uv/tools/lelab/bin/python -m pytest <that test file> -q -k stdout`

Expected: the new test fails because nothing is printed to stdout.

- [ ] **Step 3: Write the minimal implementation**

In `lelab/forge/tracking.py`, after `logger.info("Track this run --> %s", self._run.url)` add:

```python
        # LeRobot's console handler writes to stderr; the Sandbox runner reads stdout only.
        print(f"Track this run --> {self._run.url}", flush=True)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `/Users/ibaltruschat/.local/share/uv/tools/lelab/bin/python -m pytest tests/test_forge_tracking.py tests/test_jobs.py -q` (use the real file names `rg` found).

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add lelab/forge/tracking.py <the test file>
git commit -m "$(cat <<'EOF'
fix: print the W&B run line to stdout

The Sandbox runner reads stdout only, and the logger writes to stderr.
EOF
)"
```

---

### Task 6: Keep generated files out of git

**Files:**

- Modify: `.gitignore`

- [ ] **Step 1: Add the ignore**

Append `graphify-out/` to `.gitignore`, on its own line near the existing `frontend/public/brand/` entry or at the end.

- [ ] **Step 2: Verify**

Run: `git status --short` and `git check-ignore -v graphify-out/GRAPH_REPORT.md`.

Expected: `graphify-out/` no longer appears as untracked, and the check reports the new rule.

- [ ] **Step 3: Commit**

```bash
git add .gitignore
git commit -m "chore: ignore generated graphify output"
```

---

### Task 7: Final verification

**Files:** none.

- [ ] **Step 1: Run everything**

Run: `/Users/ibaltruschat/.local/share/uv/tools/lelab/bin/python -m pytest tests -q` with `required_permissions: ["all"]` (the dataset tests need `~/.cache/huggingface`), `cd frontend && npm test -- --run`, and `cd frontend && npx tsc --noEmit -p tsconfig.app.json`.

Expected: Python all pass (577 plus the new tests), frontend all pass, `tsc` shows only the two pre-existing errors.

- [ ] **Step 2: Check the tree**

Run: `git status --short`, `git ls-files frontend/public/brand`, and `git log --oneline origin/main..HEAD`.

Expected: a clean tree, no tracked file under `frontend/public/brand`, and seven commits or fewer: the baseline plus one per task.

- [ ] **Step 3: Check it in the browser**

Start Vite only (`cd frontend && npm run dev -- --host 127.0.0.1 --port 8080`). Do not start the API. Confirm in the browser: the sidebar shows Garage and the mark for each theme; the status bar reads `Idle` and `No evaluation`; both chips read `Weights & Biases · not set up` and `CoreWeave · not set up` with an icon; clicking a chip opens its hint and `Open Setup` goes to `/setup`; the tour and welcome dialog say Garage. State plainly that the API-backed states (ready chips, run counts, the Record login banner) were covered by tests only, not seen live.

- [ ] **Step 4: Report**

Report the commit list and the verification output. Do not push and do not open a pull request. The owner asks for that separately.
