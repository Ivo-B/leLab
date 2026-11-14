# Order-agent conversation gaps

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A second planner call receives a valid OpenAI tool result, and the Orders page can answer `ask_operator`.

**Architecture:** `OrderLoop` records the assistant `tool_calls` message and the matching `tool` message. `InferencePlanner` sends tool content as a JSON string and still drops frames over 200,000 bytes. The Orders page reads `question` from `GET /orders/current` and posts `POST /orders/current/answer`. The session API already stores the question and the answer.

**Tech Stack:** Python 3.12, FastAPI, React, Vitest. Pytest binary: `/Users/ibaltruschat/.local/share/uv/tools/lelab/bin/python -m pytest`.

## Global Constraints

- Work on branch `forge/order-gaps`. Push only to `origin` (`https://github.com/Ivo-B/leLab.git`). Do not push `upstream`. Do not push this branch to `main`.
- Do not start the arm. Do not call `wandb`, ARIA, or the network from tests.
- Do not wire `capture`, `run_skill`, or `count_tray` in `lelab/agent/session.py`. A live pick still returns `needs_review` and does not move the arm. Camera tiles stay "No frame yet".
- Do not change `export_trace`. It still returns `None`. Do not invent an Agent Lens URL.
- Do not invent a W&B URL.
- New Python edits keep the Apache-2.0 header already on the file.
- Commit only the files the task lists. Do not commit `graphify-out/` or `.env`. Do not skip hooks.

## Out of scope

Opening cameras, running a rollout, and returning an Agent Lens URL are a later slice. Inference is a long-running subprocess, and the tracing SDK does not yield a URL this code can honestly return.

---

### Task 1: Send a valid tool result to the planner

**Files:**

- Modify: `lelab/agent/loop.py`
- Modify: `lelab/agent/planner.py`
- Modify: `tests/test_agent_loop.py`
- Modify: `tests/test_agent_planner.py`
- Modify: `docs/superpowers/plans/2026-11-14-lelab-order-gaps.md` if it is still untracked

**Interfaces:**

- Consumes: `Action` from `lelab.agent.loop`.
- Produces: `Action.tool_call_id: str`, default `""`. After a tool runs, the transcript gains an assistant message whose `tool_calls[0].id` equals the following tool message's `tool_call_id`. Tool content sent to the client is a `str`. Frames longer than 200,000 bytes are omitted. A missing model id becomes `call_0`.

- [x] **Step 1: Write the failing tests**

Add this test to `tests/test_agent_loop.py`:

```python
def test_second_planner_call_receives_a_tool_result_message() -> None:
    seen: list[list[dict]] = []

    class RecordingPlanner:
        def next_action(self, transcript: list[dict]) -> Action:
            seen.append(transcript)
            if len(seen) == 1:
                return Action("tool", "look", {}, tool_call_id="call_look")
            return Action("done")

    OrderLoop(RecordingPlanner(), FakeExecutor()).run("one bolt", "smolvla", "local")
    assistant = seen[1][1]
    tool = seen[1][2]
    assert assistant["role"] == "assistant"
    assert assistant["tool_calls"][0]["id"] == "call_look"
    assert assistant["tool_calls"][0]["type"] == "function"
    assert assistant["tool_calls"][0]["function"]["name"] == "look"
    assert assistant["tool_calls"][0]["function"]["arguments"] == "{}"
    assert tool == {
        "role": "tool",
        "tool_call_id": "call_look",
        "content": {"top": None, "side": None, "wrist": None},
    }
```

In `tests/test_agent_planner.py`, inside `test_planner_sends_the_catalogue_model_and_drops_a_large_frame`, after the existing `"side"` assertion, add:

```python
    assert isinstance(captured["messages"][1]["content"], str)
```

- [x] **Step 2: Run the tests to verify they fail**

Run: `/Users/ibaltruschat/.local/share/uv/tools/lelab/bin/python -m pytest tests/test_agent_loop.py::test_second_planner_call_receives_a_tool_result_message tests/test_agent_planner.py::test_planner_sends_the_catalogue_model_and_drops_a_large_frame -q`

Expected: the loop test fails because the second transcript has no assistant tool call. The planner test fails because tool content is still a dict.

- [x] **Step 3: Write the minimal implementation**

Add `tool_call_id: str = ""` to `Action`, after `detail`.

In `action_from_message`, keep the first call's id:

```python
    call = calls[0]
    call_id = getattr(call, "id", None) or "call_0"
    try:
        arguments = json.loads(call.function.arguments)
    except json.JSONDecodeError:
        return Action("error")
    if not isinstance(arguments, dict):
        return Action("error")
    return Action("tool", call.function.name, arguments, tool_call_id=call_id)
```

In `OrderLoop.run`, replace the single `role: tool` append with an assistant message plus a tool message. Import `json` in `loop.py`. When `action.tool_call_id` is empty, use `f"call_{len(transcript)}"`.

```python
            call_id = action.tool_call_id or f"call_{len(transcript)}"
            transcript.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": action.name,
                                "arguments": json.dumps(action.arguments),
                            },
                        }
                    ],
                }
            )
            transcript.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": self.steps[-1]["result"],
                }
            )
```

In `_message_for_planner`, when the role is `tool` and `content` is a dict, set content to `json.dumps(_encode_tool_content(content))`.

- [x] **Step 4: Run the tests to verify they pass**

Run: `/Users/ibaltruschat/.local/share/uv/tools/lelab/bin/python -m pytest tests/test_agent_loop.py tests/test_agent_planner.py -q`

Expected: all passed.

- [x] **Step 5: Commit**

```bash
git add lelab/agent/loop.py lelab/agent/planner.py tests/test_agent_loop.py tests/test_agent_planner.py docs/superpowers/plans/2026-11-14-lelab-order-gaps.md
git commit -m "$(cat <<'EOF'
feat: send a valid tool result to the planner

EOF
)"
```

---

### Task 2: Answer an operator question from Orders

**Files:**

- Modify: `frontend/src/pages/Orders.tsx`
- Modify: `frontend/src/pages/Orders.test.tsx`

**Interfaces:**

- Consumes: `GET /orders/current` field `question: string | null`. `POST /orders/current/answer` body `{ "message": string }`, already implemented in `lelab/server.py`.
- Produces: while an order is running and `question` is a non-empty string, the page shows that text, an input labeled `Answer`, and a button `Send answer`. Clicking it posts the trimmed answer. A null or empty question shows no answer form.

- [ ] **Step 1: Write the failing test**

Follow the existing `startOrder` helper in `frontend/src/pages/Orders.test.tsx`. Add:

```tsx
  it("shows the operator question and posts the answer", async () => {
    const fetchMock = await startOrder({
      "/orders/current": () =>
        respond({
          status: "running",
          counts: { bolt: 1 },
          trace_url: null,
          question: "Which bin?",
        }),
      "/orders/current/answer": () => respond({ status: "running" }),
    });
    expect(await screen.findByText("Which bin?")).toBeInTheDocument();
    await userEvent.type(screen.getByLabelText("Answer"), "the left bin");
    await userEvent.click(screen.getByRole("button", { name: "Send answer" }));
    await waitFor(() =>
      expect(posted(fetchMock, "/orders/current/answer")).toEqual([{ message: "the left bin" }]),
    );
  });

  it("hides the answer form when the order has no question", async () => {
    await startOrder();
    expect(await screen.findByText("Status: running")).toBeInTheDocument();
    expect(screen.queryByLabelText("Answer")).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Send answer" })).not.toBeInTheDocument();
  });
```

- [ ] **Step 2: Run the test to verify it fails**

Run from `frontend/`: `npm test -- --run src/pages/Orders.test.tsx`

Expected: FAIL because the question text is not rendered.

- [ ] **Step 3: Write the minimal implementation**

Add `question: string | null` to the `Order` state. The poll copies `question` when it is a non-empty string, otherwise `null`. Keep the existing count and trace behavior.

When `order.question` is non-null, render the question text, an input with label `Answer`, and a button `Send answer`. The button posts `{ message: answer.trim() }` to `/orders/current/answer` and is disabled while busy or while the answer is blank. After a successful post, clear the local answer text. Do not clear the question until a later poll returns no question. Show the server error through the existing alert path.

- [ ] **Step 4: Run the test to verify it passes**

Run from `frontend/`: `npm test -- --run src/pages/Orders.test.tsx`

Expected: the file passes.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/pages/Orders.tsx frontend/src/pages/Orders.test.tsx
git commit -m "$(cat <<'EOF'
feat: answer an operator question from Orders

EOF
)"
```
