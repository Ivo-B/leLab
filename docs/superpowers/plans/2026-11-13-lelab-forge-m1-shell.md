# LeLab Forge M1 — theme, app shell, Setup checklist

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A personal fork of leLab that looks like a CoreWeave product, with a loop-shaped navigation (Setup, Record, Train, Evaluate, Orders, Improve) and a Setup checklist, while every existing arm workflow still works.

**Architecture:** The look is a token layer in `frontend/src/index.css`. A new `AppShell` wraps the existing React Router routes. New pages are Setup plus three placeholders. No backend and no Forge API calls in this milestone. Later milestones (M2 tracking, M3 sandboxes, M4 evaluation, M5 agent, M6 improve) each get their own plan.

**Tech Stack:** React 18, Vite, Tailwind, shadcn/ui, react-router-dom 7, vitest, Plus Jakarta Sans via `@fontsource/plus-jakarta-sans`.

**Spec:** `docs/superpowers/specs/2026-11-13-lelab-forge-so101-design.md` sections 7 and 10 (milestone M1).

## Global Constraints

- Accent blue only: primary `#0541E9`, bright `#2374FF`. No teal, no cyan, no full-page blue wash.
- Dark theme is the default. Light theme stays available.
- Plus Jakarta Sans, weights 400, 700 and 800 only. New UI copy is sentence case.
- Status is an icon plus a label, never color alone.
- The CoreWeave logo and wordmark are not committed. The app loads `/brand/mark.svg` and falls back to a letter mark when the file is missing.
- Product names, when they appear: CoreWeave Sandboxes, CoreWeave Registry, Weights & Biases Models, W&B Weave, CoreWeave Agent Lens, CoreWeave ARIA.
- `docs/` is gitignored by upstream. This plan adds a negation so `docs/superpowers/` is tracked. Commit the spec and this plan with the gitignore change.
- Do not fork LeRobot. Do not add Forge SDKs in this milestone.
- Run frontend tests with `npm test` from `frontend/`. Run the frontend dev server with `npm run dev` from `frontend/`.

---

### Task 1: Track the design docs

**Files:**
- Modify: `.gitignore:68`
- Already written: `docs/superpowers/specs/2026-11-13-lelab-forge-so101-design.md`
- Already written: `docs/superpowers/plans/2026-11-13-lelab-forge-m1-shell.md`

**Interfaces:**
- Consumes: nothing
- Produces: `docs/superpowers/` is a tracked path

- [ ] **Step 1: Stop ignoring the design docs**

In `.gitignore`, replace the single `docs/` line with:

```
docs/*
!docs/superpowers/
```

- [ ] **Step 2: Commit the spec, this plan, and the gitignore change**

```bash
git add .gitignore docs/superpowers/specs/2026-11-13-lelab-forge-so101-design.md docs/superpowers/plans/2026-11-13-lelab-forge-m1-shell.md
git commit -m "docs: track the Forge design spec and the M1 plan"
```

Expected: `git check-ignore docs/superpowers/plans/2026-11-13-lelab-forge-m1-shell.md` prints nothing and exits 1.

---

### Task 2: CoreWeave theme tokens and the font

**Files:**
- Modify: `frontend/package.json` (dependency only, via npm)
- Modify: `frontend/src/index.css`
- Modify: `frontend/tailwind.config.ts`
- Modify: `frontend/src/main.tsx`
- Modify: `frontend/src/App.tsx` (default theme)
- Modify: `frontend/src/components/ui/button.tsx:7` (weight 500 is not an allowed brand weight)

**Interfaces:**
- Consumes: existing shadcn tokens (`bg-background`, `text-foreground`, `bg-primary`, `border-border`, `text-muted-foreground`)
- Produces: those utilities resolve to the CoreWeave palette; `font-sans` is Plus Jakarta Sans; a visitor with no saved theme gets dark mode

Hex values and the HSL channels Tailwind expects (`hsl(var(--token))` with space-separated components):

| Token | Hex | HSL channels |
|---|---|---|
| Primary | `#0541E9` | `224 96% 47%` |
| Bright (ring, focus) | `#2374FF` | `218 100% 57%` |
| Deep surface | `#001581` | `230 100% 25%` |
| Grey text | `#464B51` | `210 6% 30%` |
| Muted grey | `#686D73` | `213 5% 43%` |
| Light grey | `#C2C6CC` | `216 7% 78%` |
| Hairline | `#AFB3B8` | `214 6% 70%` |

Destructive stays red and is marked for brand review. The brand palette has no red, amber, or green. Use it only with an icon and a label.

- [ ] **Step 1: Install the font**

From `frontend/`:

```bash
npm install @fontsource/plus-jakarta-sans
```

- [ ] **Step 2: Load weights 400, 700 and 800**

Add these imports at the top of `frontend/src/main.tsx`, above the `index.css` import:

```tsx
import "@fontsource/plus-jakarta-sans/400.css";
import "@fontsource/plus-jakarta-sans/700.css";
import "@fontsource/plus-jakarta-sans/800.css";
```

- [ ] **Step 3: Register the font**

In `frontend/tailwind.config.ts`, inside `theme.extend`, add:

```ts
fontFamily: {
  sans: ["Plus Jakarta Sans", "ui-sans-serif", "system-ui", "sans-serif"],
},
```

- [ ] **Step 4: Replace the token block in `frontend/src/index.css`**

Replace the whole `:root` block and the whole `.dark` block (keep both `@layer base` wrappers and the `body` rule) with:

```css
  :root {
    --background: 0 0% 100%;
    --foreground: 210 6% 30%;
    --card: 0 0% 100%;
    --card-foreground: 210 6% 30%;
    --popover: 0 0% 100%;
    --popover-foreground: 210 6% 30%;
    --primary: 224 96% 47%; /* #0541E9, brand review: accent only */
    --primary-foreground: 0 0% 100%;
    --secondary: 216 7% 94%;
    --secondary-foreground: 210 6% 30%;
    --muted: 216 7% 94%;
    --muted-foreground: 213 5% 43%;
    --accent: 216 7% 94%;
    --accent-foreground: 210 6% 30%;
    /* Brand review: the palette has no red. Icon + label required wherever this is used. */
    --destructive: 0 72% 51%;
    --destructive-foreground: 0 0% 100%;
    --border: 214 6% 70%;
    --input: 214 6% 70%;
    --ring: 218 100% 57%; /* #2374FF */
    --radius: 0.5rem;
    --sidebar-background: 0 0% 100%;
    --sidebar-foreground: 210 6% 30%;
    --sidebar-primary: 224 96% 47%;
    --sidebar-primary-foreground: 0 0% 100%;
    --sidebar-accent: 216 7% 94%;
    --sidebar-accent-foreground: 210 6% 30%;
    --sidebar-border: 214 6% 70%;
    --sidebar-ring: 218 100% 57%;
  }

  .dark {
    --background: 0 0% 0%;
    --foreground: 0 0% 100%;
    --card: 230 100% 25%; /* #001581 */
    --card-foreground: 0 0% 100%;
    --popover: 230 100% 25%;
    --popover-foreground: 0 0% 100%;
    --primary: 224 96% 47%;
    --primary-foreground: 0 0% 100%;
    --secondary: 230 100% 18%;
    --secondary-foreground: 0 0% 100%;
    --muted: 230 100% 18%;
    --muted-foreground: 216 7% 78%;
    --accent: 230 100% 18%;
    --accent-foreground: 0 0% 100%;
    --destructive: 0 72% 51%;
    --destructive-foreground: 0 0% 100%;
    --border: 230 60% 32%;
    --input: 230 60% 32%;
    --ring: 218 100% 57%;
    --sidebar-background: 0 0% 0%;
    --sidebar-foreground: 0 0% 100%;
    --sidebar-primary: 224 96% 47%;
    --sidebar-primary-foreground: 0 0% 100%;
    --sidebar-accent: 230 100% 18%;
    --sidebar-accent-foreground: 0 0% 100%;
    --sidebar-border: 230 60% 32%;
    --sidebar-ring: 218 100% 57%;
  }
```

Add `font-sans` to the body rule:

```css
  body {
    @apply bg-background text-foreground font-sans;
  }
```

- [ ] **Step 5: Default to dark, and drop weight 500 on buttons**

In `frontend/src/App.tsx`, pass `defaultTheme="dark"` to `ThemeProvider`.

In `frontend/src/contexts/ThemeContext.tsx`, change the default `storageKey` from `"vite-ui-theme"` to `"lelab-forge-theme"` so an old leLab preference does not override the new default.

In `frontend/src/components/ui/button.tsx`, change `font-medium` to `font-bold` in the `buttonVariants` base string.

- [ ] **Step 6: Restyle the page roots that hardcode a background**

Replace these class strings. Do not change anything else on the line.

| File | Find | Replace |
|---|---|---|
| `frontend/src/pages/Landing.tsx` | `min-h-screen bg-black text-white pb-16` | `min-h-screen bg-background text-foreground pb-16` |
| `frontend/src/pages/Landing.tsx` | `bg-black/95 backdrop-blur supports-[backdrop-filter]:bg-black/70 border-b border-gray-800` | `bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/70 border-b border-border` |
| `frontend/src/pages/Training.tsx` (6 occurrences) | `min-h-screen bg-slate-900 text-white p-4` | `min-h-screen bg-background text-foreground p-4` |
| `frontend/src/pages/Calibration.tsx` | `min-h-screen bg-slate-900 text-white p-4` | `min-h-screen bg-background text-foreground p-4` |
| `frontend/src/pages/Teleoperation.tsx` | `min-h-screen bg-black flex items-center justify-center p-2 sm:p-4` | `min-h-screen bg-background text-foreground flex items-center justify-center p-2 sm:p-4` |
| `frontend/src/pages/EditDataset.tsx` | `min-h-screen bg-black p-6 pb-24 text-white` | `min-h-screen bg-background text-foreground p-6 pb-24` |
| `frontend/src/pages/Inference.tsx` | `min-h-screen bg-black text-white flex items-center justify-center` | `min-h-screen bg-background text-foreground flex items-center justify-center` |
| `frontend/src/pages/Inference.tsx` | `min-h-screen bg-black text-white flex flex-col p-4 sm:p-6 lg:p-8` | `min-h-screen bg-background text-foreground flex flex-col p-4 sm:p-6 lg:p-8` |
| `frontend/src/pages/Recording.tsx` (2 occurrences) | `min-h-screen bg-black text-white flex items-center justify-center` | `min-h-screen bg-background text-foreground flex items-center justify-center` |
| `frontend/src/pages/Recording.tsx` | `h-screen bg-black text-white p-6 flex flex-col overflow-hidden` | `h-screen bg-background text-foreground p-6 flex flex-col overflow-hidden` |
| `frontend/src/pages/Upload.tsx` | `min-h-screen bg-black text-white flex items-center justify-center` | `min-h-screen bg-background text-foreground flex items-center justify-center` |
| `frontend/src/pages/Upload.tsx` | `min-h-screen bg-black text-white p-8` | `min-h-screen bg-background text-foreground p-8` |
| `frontend/src/pages/NotFound.tsx` | `min-h-screen flex items-center justify-center bg-gray-100` | `min-h-screen flex items-center justify-center bg-background text-foreground` |

Inner cards that still say `bg-gray-800` or `bg-slate-800` stay as they are in this task. Fix a card only when the browser check in Task 6 shows it unreadable on the new background.

- [ ] **Step 7: Verify the build**

From `frontend/`:

```bash
npm test
npm run build
```

Expected: existing vitest tests pass and `vite build` exits 0.

- [ ] **Step 8: Commit**

```bash
git add frontend/package.json frontend/package-lock.json frontend/src/index.css frontend/tailwind.config.ts frontend/src/main.tsx frontend/src/App.tsx frontend/src/contexts/ThemeContext.tsx frontend/src/components/ui/button.tsx frontend/src/pages
git commit -m "feat: apply the CoreWeave theme tokens and Plus Jakarta Sans"
```

---

### Task 3: Brand mark that is safe to publish

**Files:**
- Create: `frontend/src/components/shell/BrandMark.tsx`
- Create: `frontend/src/components/shell/BrandMark.test.tsx`
- Create: `frontend/BRAND.md`
- Modify: `.gitignore` (append)
- Modify: `frontend/src/components/Logo.tsx`
- Modify: `frontend/src/components/landing/LandingTopBar.tsx`

**Interfaces:**
- Consumes: nothing
- Produces: `BrandMark` with prop `iconOnly?: boolean`. It renders an `img` pointing at `/brand/mark.svg` while that file loads, and a letter `L` in a `data-testid="brand-fallback"` element when the image errors. The word "LeLab" is `font-extrabold` unless `iconOnly` is set.

- [ ] **Step 1: Write the failing test**

`frontend/src/components/shell/BrandMark.test.tsx`:

```tsx
import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { BrandMark } from "./BrandMark";

describe("BrandMark", () => {
  it("shows the wordmark and swaps to the letter mark when the logo file is missing", () => {
    render(<BrandMark />);
    expect(screen.getByText("LeLab")).toBeInTheDocument();
    const img = screen.getByRole("img", { name: "LeLab" });
    expect(img).toHaveAttribute("src", "/brand/mark.svg");
    fireEvent.error(img);
    expect(screen.getByTestId("brand-fallback")).toHaveTextContent("L");
    expect(screen.queryByRole("img")).not.toBeInTheDocument();
  });

  it("hides the wordmark when iconOnly is set", () => {
    render(<BrandMark iconOnly />);
    expect(screen.queryByText("LeLab")).not.toBeInTheDocument();
  });
});
```

- [ ] **Step 2: Run the test and confirm it fails**

From `frontend/`:

```bash
npx vitest run src/components/shell/BrandMark.test.tsx
```

Expected: FAIL, cannot find `./BrandMark`.

- [ ] **Step 3: Implement BrandMark**

`frontend/src/components/shell/BrandMark.tsx`:

```tsx
import { useState } from "react";

interface BrandMarkProps {
  iconOnly?: boolean;
}

export function BrandMark({ iconOnly = false }: BrandMarkProps) {
  const [failed, setFailed] = useState(false);

  return (
    <div className="flex items-center gap-2">
      {failed ? (
        <span
          data-testid="brand-fallback"
          aria-hidden
          className="grid h-8 w-8 place-items-center rounded-md bg-primary font-extrabold text-primary-foreground"
        >
          L
        </span>
      ) : (
        <img
          src="/brand/mark.svg"
          alt="LeLab"
          className="h-8 w-8"
          onError={() => setFailed(true)}
        />
      )}
      {!iconOnly && <span className="font-extrabold text-xl">LeLab</span>}
    </div>
  );
}
```

- [ ] **Step 4: Point the old logo at BrandMark and gitignore the asset**

Replace the body of `frontend/src/components/Logo.tsx` so it renders `<BrandMark iconOnly={iconOnly} className={className} />`. Add `className?: string` to `BrandMarkProps` and apply it on the wrapper div (`cn` from `@/lib/utils`) so the existing `Logo` prop still works.

Replace the image and the "LeLab" span in `frontend/src/components/landing/LandingTopBar.tsx` with `<BrandMark />`.

Append to `.gitignore`:

```
# Local brand assets. See frontend/BRAND.md. Never commit the CoreWeave logo.
frontend/public/brand/
```

`frontend/BRAND.md`:

```markdown
# Brand assets

The CoreWeave logo and wordmark are not in git. To use them locally, create
`frontend/public/brand/mark.svg` (the geometric mark, at least 64px wide, on a
black or white field). The app falls back to a letter mark when the file is
absent.

Ask in Slack #help-brand-support before using the logo in anything shown
outside CoreWeave. The theme tokens and Plus Jakarta Sans are fine to publish.
```

- [ ] **Step 5: Run the test**

```bash
npx vitest run src/components/shell/BrandMark.test.tsx
```

Expected: PASS, 2 tests.

- [ ] **Step 6: Commit**

```bash
git add frontend/src/components/shell frontend/src/components/Logo.tsx frontend/src/components/landing/LandingTopBar.tsx frontend/BRAND.md .gitignore
git commit -m "feat: fall back to a letter mark when the brand logo is absent"
```

---

### Task 4: Navigation model

**Files:**
- Create: `frontend/src/components/shell/nav.ts`
- Create: `frontend/src/components/shell/nav.test.ts`

**Interfaces:**
- Consumes: nothing
- Produces:

```ts
export interface NavItem {
  label: string;
  to: string;
  isActive: (pathname: string) => boolean;
}

export const NAV: NavItem[];
```

Labels, in order: Setup, Record, Train, Evaluate, Orders, Improve.

Active rules:

| Label | `to` | Active when |
|---|---|---|
| Setup | `/setup` | `/setup` or path starts with `/calibration` |
| Record | `/` | exactly `/`, or starts with `/recording`, `/teleoperation`, `/upload`, `/edit-dataset` |
| Train | `/training` | starts with `/training` |
| Evaluate | `/evaluate` | `/evaluate` or starts with `/inference` |
| Orders | `/orders` | `/orders` |
| Improve | `/improve` | `/improve` |

`/` must not mark Record active on `/training`.

- [ ] **Step 1: Write the failing test**

`frontend/src/components/shell/nav.test.ts`:

```ts
import { describe, expect, it } from "vitest";
import { NAV } from "./nav";

function active(pathname: string): string[] {
  return NAV.filter((item) => item.isActive(pathname)).map((item) => item.label);
}

describe("NAV", () => {
  it("lists the loop in order", () => {
    expect(NAV.map((item) => item.label)).toEqual([
      "Setup",
      "Record",
      "Train",
      "Evaluate",
      "Orders",
      "Improve",
    ]);
  });

  it("marks exactly one item for each existing route", () => {
    expect(active("/")).toEqual(["Record"]);
    expect(active("/recording")).toEqual(["Record"]);
    expect(active("/edit-dataset")).toEqual(["Record"]);
    expect(active("/calibration")).toEqual(["Setup"]);
    expect(active("/training")).toEqual(["Train"]);
    expect(active("/training/job-1")).toEqual(["Train"]);
    expect(active("/inference")).toEqual(["Evaluate"]);
    expect(active("/evaluate")).toEqual(["Evaluate"]);
    expect(active("/orders")).toEqual(["Orders"]);
    expect(active("/improve")).toEqual(["Improve"]);
    expect(active("/nope")).toEqual([]);
  });
});
```

- [ ] **Step 2: Run it and confirm it fails**

```bash
npx vitest run src/components/shell/nav.test.ts
```

Expected: FAIL, cannot find `./nav`.

- [ ] **Step 3: Implement `nav.ts`**

```ts
export interface NavItem {
  label: string;
  to: string;
  isActive: (pathname: string) => boolean;
}

function startsWith(pathname: string, prefix: string): boolean {
  return pathname === prefix || pathname.startsWith(`${prefix}/`);
}

export const NAV: NavItem[] = [
  {
    label: "Setup",
    to: "/setup",
    isActive: (p) => p === "/setup" || startsWith(p, "/calibration"),
  },
  {
    label: "Record",
    to: "/",
    isActive: (p) =>
      p === "/" ||
      ["/recording", "/teleoperation", "/upload", "/edit-dataset"].some((prefix) =>
        startsWith(p, prefix),
      ),
  },
  { label: "Train", to: "/training", isActive: (p) => startsWith(p, "/training") },
  {
    label: "Evaluate",
    to: "/evaluate",
    isActive: (p) => p === "/evaluate" || startsWith(p, "/inference"),
  },
  { label: "Orders", to: "/orders", isActive: (p) => p === "/orders" },
  { label: "Improve", to: "/improve", isActive: (p) => p === "/improve" },
];
```

- [ ] **Step 4: Run the test**

```bash
npx vitest run src/components/shell/nav.test.ts
```

Expected: PASS, 2 tests.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/shell/nav.ts frontend/src/components/shell/nav.test.ts
git commit -m "feat: define the Forge loop navigation"
```

---

### Task 5: App shell, status bar, and routes

**Files:**
- Create: `frontend/src/components/shell/StatusBar.tsx`
- Create: `frontend/src/components/shell/AppShell.tsx`
- Create: `frontend/src/pages/Setup.tsx`
- Create: `frontend/src/pages/PlaceholderPage.tsx`
- Modify: `frontend/src/App.tsx`
- Modify: `frontend/src/pages/Landing.tsx` (remove `LandingTopBar`; the shell replaces it)

**Interfaces:**
- Consumes: `NAV` and `NavItem` from `./nav`; `BrandMark`; existing endpoints:
  - `GET /teleoperation-status` → `{ teleoperation_active: boolean }`
  - `GET /recording-status` → `{ recording_active: boolean }`
  - `GET /replay-status` → `{ replay_active: boolean }`
  - `GET /inference-status` → `{ inference_active: boolean }`
  - `GET /jobs` → `{ jobs: { id: string; name: string; state: "running" | "done" | "failed" | "interrupted" }[] }`
- Produces: `AppShell` renders the nav, the status bar, and `<Outlet />`. Routes `/setup`, `/evaluate`, `/orders`, `/improve` exist. Every previously existing route still exists.

Status bar content, left to right:

| Slot | M1 value | How |
|---|---|---|
| Arm | "Arm idle" or "Arm in use" | "Arm in use" when any of the four `*_active` flags is true. Lucide `Cable` icon. |
| Job | "No job" or the running job's `name` | First job with `state === "running"`. Lucide `Cpu` icon. |
| Policy host | "Local" | Fixed string. Lucide `Monitor` icon. The CoreWeave host arrives in M4. |
| Forge | "Not connected" | Fixed string. Lucide `Unplug` icon. The connection arrives in M2. |

Poll the five endpoints every 5 seconds. A failed fetch leaves the previous value and does not throw. Each slot is a `<span>` with the icon (`aria-hidden`) and the text, so the state is not carried by color.

Setup page checklist, each row a link:

| Row | Links to |
|---|---|
| Calibrate the leader and follower arms | `/calibration` |
| Check the cameras on the recording page | `/recording` |
| Sign in to Hugging Face | `/` |
| Connect Forge | text only: "Add WANDB_API_KEY and CWSANDBOX_API_KEY to the backend environment. This arrives in the next milestone." |

`PlaceholderPage` takes a `title` and a `body`. Evaluate, Orders, and Improve use it:

- Evaluate: "Real-arm evaluation in W&B Weave arrives in a later milestone. Rollouts still run from the inference page."
- Orders: "The order-picking agent arrives in a later milestone."
- Improve: "CoreWeave ARIA reports arrive in a later milestone."

- [ ] **Step 1: Write StatusBar and AppShell**

`frontend/src/components/shell/StatusBar.tsx`:

```tsx
import { useEffect, useState, type ReactNode } from "react";
import { Cable, Cpu, Monitor, Unplug } from "lucide-react";

interface CellStatus {
  arm: "Arm idle" | "Arm in use";
  job: string;
}

const IDLE: CellStatus = { arm: "Arm idle", job: "No job" };

async function readCellStatus(): Promise<CellStatus> {
  const [teleop, recording, replay, inference, jobs] = await Promise.all(
    ["/teleoperation-status", "/recording-status", "/replay-status", "/inference-status", "/jobs"].map(
      (path) => fetch(path).then((r) => (r.ok ? r.json() : {})),
    ),
  );
  const armActive = Boolean(
    teleop.teleoperation_active ||
      recording.recording_active ||
      replay.replay_active ||
      inference.inference_active,
  );
  const jobsList = Array.isArray(jobs.jobs)
    ? (jobs.jobs as { state?: string; name?: string }[])
    : [];
  const running = jobsList.find((job) => job.state === "running");
  return { arm: armActive ? "Arm in use" : "Arm idle", job: running?.name ?? "No job" };
}

function Slot({ icon, label }: { icon: ReactNode; label: string }) {
  return (
    <span className="flex items-center gap-2 text-sm">
      <span aria-hidden className="text-primary">{icon}</span>
      {label}
    </span>
  );
}

export function StatusBar() {
  const [status, setStatus] = useState<CellStatus>(IDLE);

  useEffect(() => {
    let cancelled = false;
    const tick = () => {
      readCellStatus()
        .then((next) => { if (!cancelled) setStatus(next); })
        .catch(() => undefined);
    };
    tick();
    const id = setInterval(tick, 5000);
    return () => { cancelled = true; clearInterval(id); };
  }, []);

  return (
    <div className="flex h-12 items-center gap-6 border-b border-border px-4">
      <Slot icon={<Cable size={16} />} label={status.arm} />
      <Slot icon={<Cpu size={16} />} label={status.job} />
      <Slot icon={<Monitor size={16} />} label="Local" />
      <Slot icon={<Unplug size={16} />} label="Not connected" />
    </div>
  );
}
```

`frontend/src/components/shell/AppShell.tsx`:

```tsx
import { NavLink, Outlet } from "react-router-dom";
import { BrandMark } from "./BrandMark";
import { NAV } from "./nav";
import { StatusBar } from "./StatusBar";

export function AppShell() {
  return (
    <div className="flex h-screen bg-background text-foreground">
      <aside className="flex w-56 shrink-0 flex-col gap-6 border-r border-border p-4">
        <BrandMark />
        <nav className="flex flex-col gap-1">
          {NAV.map((item) => (
            <NavLink
              key={item.label}
              to={item.to}
              className={({ isActive }) =>
                `rounded-md px-3 py-2 text-sm font-bold ${
                  isActive ? "bg-primary text-primary-foreground" : "hover:bg-accent"
                }`
              }
              end={item.to === "/"}
            >
              {item.label}
            </NavLink>
          ))}
        </nav>
      </aside>
      <div className="flex min-w-0 flex-1 flex-col">
        <StatusBar />
        <main className="min-h-0 flex-1 overflow-auto">
          <Outlet />
        </main>
      </div>
    </div>
  );
}
```

`NavLink`'s `isActive` matches by path prefix except when `end` is set. That agrees with `nav.ts` for these routes. Do not pass a custom `className` function that calls `item.isActive`; use `NavLink`'s own `isActive` so the test below can render it.

- [ ] **Step 2: Write the pages**

`frontend/src/pages/PlaceholderPage.tsx`:

```tsx
export function PlaceholderPage({ title, body }: { title: string; body: string }) {
  return (
    <section className="mx-auto max-w-3xl p-8">
      <h1 className="font-extrabold text-3xl">{title}</h1>
      <p className="mt-4 text-base text-muted-foreground">{body}</p>
    </section>
  );
}
```

`frontend/src/pages/Setup.tsx`:

```tsx
import { Link } from "react-router-dom";

const LINKS = [
  { href: "/calibration", title: "Calibrate the leader and follower arms", detail: "Ports, motor direction, and range." },
  { href: "/recording", title: "Check the cameras", detail: "Wrist, top, and side. The side camera is recorded and kept out of the policy." },
  { href: "/", title: "Sign in to Hugging Face", detail: "Datasets and policies still push to the Hub." },
];

export default function Setup() {
  return (
    <section className="mx-auto max-w-3xl p-8">
      <h1 className="font-extrabold text-3xl">Setup</h1>
      <p className="mt-2 text-muted-foreground">Work through this once per cell.</p>
      <ol className="mt-6 flex flex-col gap-3">
        {LINKS.map((item) => (
          <li key={item.href}>
            <Link to={item.href} className="block rounded-md border border-border p-4 hover:bg-accent">
              <span className="font-bold">{item.title}</span>
              <span className="mt-1 block text-sm text-muted-foreground">{item.detail}</span>
            </Link>
          </li>
        ))}
        <li className="rounded-md border border-border p-4">
          <span className="font-bold">Connect Forge</span>
          <span className="mt-1 block text-sm text-muted-foreground">
            Add WANDB_API_KEY and CWSANDBOX_API_KEY to the backend environment. Connecting
            them arrives in the next milestone.
          </span>
        </li>
      </ol>
    </section>
  );
}
```

- [ ] **Step 3: Mount the shell**

In `frontend/src/App.tsx`, import `AppShell`, `Setup`, and `PlaceholderPage`. Wrap the existing routes in a parent route and add the four new ones:

```tsx
<Route element={<AppShell />}>
  <Route path="/" element={<Landing />} />
  <Route path="/setup" element={<Setup />} />
  <Route path="/teleoperation" element={<Teleoperation />} />
  <Route path="/recording" element={<Recording />} />
  <Route path="/upload" element={<Upload />} />
  <Route path="/training" element={<Training />} />
  <Route path="/training/:jobId" element={<Training />} />
  <Route path="/inference" element={<Inference />} />
  <Route path="/calibration" element={<Calibration />} />
  <Route path="/edit-dataset" element={<EditDataset />} />
  <Route
    path="/evaluate"
    element={
      <PlaceholderPage
        title="Evaluate"
        body="Real-arm evaluation in W&B Weave arrives in a later milestone. Rollouts still run from the inference page."
      />
    }
  />
  <Route
    path="/orders"
    element={<PlaceholderPage title="Orders" body="The order-picking agent arrives in a later milestone." />}
  />
  <Route
    path="/improve"
    element={<PlaceholderPage title="Improve" body="CoreWeave ARIA reports arrive in a later milestone." />}
  />
  <Route path="*" element={<NotFound />} />
</Route>
```

In `frontend/src/pages/Landing.tsx`, remove the `LandingTopBar` import and the `<LandingTopBar />` element. The shell now provides the header, so the sticky bar under it should use `top: 0` instead of `var(--lelab-topbar-h)`. Delete the `--lelab-topbar-h` style.

- [ ] **Step 4: Write the shell test**

`frontend/src/components/shell/AppShell.test.tsx`:

```tsx
import { render, screen } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { describe, expect, it } from "vitest";
import { AppShell } from "./AppShell";

describe("AppShell", () => {
  it("highlights Train and shows the status bar", () => {
    render(
      <MemoryRouter initialEntries={["/training"]}>
        <Routes>
          <Route element={<AppShell />}>
            <Route path="/training" element={<p>Training page</p>} />
          </Route>
        </Routes>
      </MemoryRouter>,
    );
    const train = screen.getByRole("link", { name: "Train" });
    expect(train.className).toContain("bg-primary");
    expect(screen.getByRole("link", { name: "Record" }).className).not.toContain("bg-primary");
    expect(screen.getByText("Local")).toBeInTheDocument();
    expect(screen.getByText("Not connected")).toBeInTheDocument();
    expect(screen.getByText("Training page")).toBeInTheDocument();
  });
});
```

- [ ] **Step 5: Run the tests and the build**

From `frontend/`:

```bash
npx vitest run src/components/shell
npm run build
```

Expected: shell tests pass and the build exits 0.

- [ ] **Step 6: Check it in the browser**

Start the frontend (`npm run dev` in `frontend/`) and the backend (`lelab --dev` from the repo root, or the Vite dev server alone if the API is already running). Then:

1. Open `/`. The left nav shows the six labels, Record is highlighted, the status bar reads "Arm idle", "No job", "Local", "Not connected", and the landing content is still usable.
2. Open `/setup` and follow each of the three links. Each opens the existing page.
3. Open `/training`, `/inference`, `/calibration`, `/evaluate`, `/orders`, `/improve`. Each renders inside the shell, and the highlighted nav item matches the table in this task.
4. Toggle the theme to light from the existing theme control if one is visible; otherwise set `localStorage["lelab-forge-theme"] = "light"` and reload. Text stays readable. Set it back to `"dark"`.
5. Confirm the page background is black in dark mode, not blue, and the only blue is the active nav item and the status icons.

Fix any page whose root is still a hardcoded dark slate before committing.

- [ ] **Step 7: Commit**

```bash
git add frontend/src/components/shell frontend/src/pages/Setup.tsx frontend/src/pages/PlaceholderPage.tsx frontend/src/pages/Landing.tsx frontend/src/App.tsx
git commit -m "feat: add the Forge app shell and the setup checklist"
```

---

## Self-review

Spec section 7 (UI) against this plan:

| Requirement | Task |
|---|---|
| Token layer, Plus Jakarta Sans, weights 400/700/800, dark default, blue as accent | Task 2 |
| Logo out of git, letter-mark fallback | Task 3 |
| Left nav: Setup, Record, Train, Evaluate, Orders, Improve | Tasks 4 and 5 |
| Status bar: arm, job, policy host, Forge | Task 5 (host and Forge are fixed labels until M2 and M4) |
| Setup as one checklist | Task 5 |
| "Open in Forge" links | Not in M1. No Forge objects exist yet. First used in the M2 plan. |
| Page template (title, one primary action, details panel) | Not in M1. Applied when Evaluate, Orders, and Improve become real pages. |
| WCAG AA contrast check with a tool | Not automated here. Task 5 step 6 is the visual check. |
| Inner component grays (`bg-gray-800` and similar) | Left in place. Task 2 step 6 and Task 5 step 6 say to fix a card when the browser shows it unreadable. |

M2 through M6 are separate plans, written when the previous milestone is done.
