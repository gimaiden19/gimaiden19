# Operations-management LLM Agent app for a small fitness & rehabilitation centre

| | |
|---|---|
| **Period** | 2026.09 – present (in progress) |
| **Role** | Sole designer / developer |
| **Repo** | [github.com/gimaiden19/theone](https://github.com/gimaiden19/theone) |
| **Stack** | Next.js 15 (App Router) · SQLite (better-sqlite3) · Drizzle ORM · Claude Messages API · Ollama-swappable LLM adapter · Electron installer |

## Why it exists

A single-owner centre ran accounting in spreadsheets and by hand, managed freelance trainers' schedules manually, and planned social-media content ad hoc. The users (owner, trainers) have no development background, so the requirement was: **one installer file, no Node/terminal, runs on the centre's own PC, data stays local.**

## What was built

| Module | Notes |
|---|---|
| Accounts & roles | `admin` (owner) with full rights; trainer accounts see their own timetable and assigned members; admin can reset passwords |
| Accounting | Per-branch income / expense (branch structure prepared for future locations), monthly reports comparable with the same month last year <!-- TODO: 구현 상태 확인 -->, trainer settlement (per-session revenue split, withholding rate configurable) |
| HR | Trainer records, working days / conditions, active / on-leave / resigned states, retention period for resigned records (default 2 years, configurable) |
| Timetable | University-style weekly grid per owner / trainer, "(member) name time–time" cells; substitutions, cancellations and no-shows edited by the trainer and reflected live |
| Members | Spreadsheet-like table → detail view; historical spreadsheet data imported by period |
| Agents | **Accounting agent** — answers questions over the ledger, proposes analysis questions proactively (e.g., monthly revenue vs. target, attendance-based product suggestions). **Content agent** — topic → plan → outline proposals → user picks → draft → **approval**. **HR agent** — personnel records and status changes |
| LLM layer | Claude Messages API by default; adapter boundary allows switching to local Ollama |
| Distribution | Electron-packaged `.exe`; public release ships without keys, owner build ships with keys bundled |

## Design decisions I would defend in a review

- **Local-first**: data on the centre PC, not a cloud DB — the owner's requirement and a privacy win
- **SQLite over a server DB**: single machine, ≤3 concurrent users, one-file backup
- **Human-in-the-loop by construction**: the content agent cannot publish without an explicit approval step; the accounting agent *proposes* questions rather than silently acting — the same "endorse / abstain" stance as my research on selective prediction
- **Undecided requirements → admin settings**, not hard-coded values

## How it was built (honest note)

Requirements interviews, architecture choices, feature specs, data schema, review of generated code and functional testing were mine; most code was written in collaboration with AI coding agents. Planning documents are versioned (`기획안 v0.3`, `시스템설계서 v0.2`, …) alongside the code.

## Status

Core skeleton, schema, login / settings and shared infrastructure done; module completion and the owner's installer build in progress. <!-- TODO: 현재 진행 상태로 갱신 -->
