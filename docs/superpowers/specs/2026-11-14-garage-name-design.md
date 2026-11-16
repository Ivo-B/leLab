# Garage sidebar name

**Date:** 2026-11-14

**Status:** Implemented locally

## Name

The visible product name in this fork is **Garage**. It is a nickname for the robot practice room, not an official CoreWeave product. The Python package, the GitHub repo, and the `lelab` command stay `lelab`.

Garage is a place, not a task. The arm can practice kitting, drawing, or anything else in the same room.

## Logo

The sidebar loader picks the mark from the page theme. A dark surface uses `frontend/public/brand/cw-dark.png`, the full CoreWeave lockup with the white W. A light surface uses `frontend/public/brand/cw-light.png`, the same lockup with a black W. The black matte is already transparent in that file. The dark file has its black matte knocked out. That directory stays gitignored. The official artwork is not committed. When a file is missing, the tile shows the letter **G**.

The image alt text is `CoreWeave`. The word beside it is `Garage`. Under the name, one muted line reads `A LeLab demo on CoreWeave`.

## Copy that changes

- `frontend/index.html` title and `og:title`
- `frontend/src/components/shell/BrandMark.tsx` and its test
- `frontend/src/components/onboarding/WelcomeDialog.tsx`
- `frontend/src/components/SingleTabGuard.tsx`
- `frontend/src/components/UpdateNotice.tsx`
- `frontend/src/components/landing/UsageInstructionsModal.tsx`
- `frontend/src/lib/onboardingSteps.ts` title `Welcome to LeLab`

Comments and the update-check hook stay as they are. The README keeps the upstream name LeLab.

## Brand constraints

CoreWeave is spelled `CoreWeave`. The nickname is sentence case in headings and a proper noun in the wordmark. No invented logo, no exclamation point, and no claim that Garage is a CoreWeave product.
