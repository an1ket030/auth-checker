---
name: git-workflow
description: Git branching strategy, commit conventions, and workflow for the AuthChecker project.
---

# Git Workflow Skill

## Branch Strategy

Use a simplified Git Flow model:

| Branch | Purpose | Merges Into |
|--------|---------|-------------|
| `main` | Stable, production-ready code | — |
| `dev` | Integration branch for features | `main` |
| `feature/<name>` | New features | `dev` |
| `fix/<name>` | Bug fixes | `dev` or `main` (hotfix) |
| `refactor/<name>` | Code refactoring | `dev` |
| `docs/<name>` | Documentation updates | `dev` |
| `build/<name>` | Build/CI changes | `dev` |

### Creating a Branch
```bash
# Always branch from dev for features
git checkout dev
git pull origin dev
git checkout -b feature/add-nfc-scanning

# For hotfixes, branch from main
git checkout main
git pull origin main
git checkout -b fix/login-crash
```

## Commit Message Convention

Use **Conventional Commits** format:

```
<type>(<scope>): <short description>

[optional body]
[optional footer]
```

### Types
| Type | When to Use |
|------|-------------|
| `feat` | New feature |
| `fix` | Bug fix |
| `docs` | Documentation only |
| `style` | Formatting, no logic change |
| `refactor` | Code restructuring, no behavior change |
| `test` | Adding/updating tests |
| `build` | Build system, dependencies, EAS config |
| `chore` | Maintenance, tooling |

### Scopes (AuthChecker-specific)
| Scope | Component |
|-------|-----------|
| `mobile` | React Native / Expo app |
| `backend` | Python FastAPI backend |
| `ml` | Machine learning model/inference |
| `db` | Database changes |
| `docker` | Docker/container config |
| `auth` | Authentication system |
| `ocr` | OCR/document processing |

### Examples
```bash
git commit -m "feat(mobile): add NFC tag scanning screen"
git commit -m "fix(backend): resolve hash mismatch in document verification"
git commit -m "build(mobile): update EAS config for preview APK"
git commit -m "docs: update PRD with phase 2 requirements"
git commit -m "refactor(ml): extract model inference into separate module"
```

## Daily Workflow

### Starting Work
```bash
# 1. Pull latest changes
git checkout dev
git pull origin dev

# 2. Create feature branch
git checkout -b feature/your-feature-name

# 3. Start working...
```

### During Work — Stage & Commit Often
```bash
# Stage specific files (preferred over git add .)
git add mobile/src/screens/NewScreen.js
git add backend/routes/new_endpoint.py

# Commit with conventional message
git commit -m "feat(mobile): add document scan result screen"
```

### Finishing Work
```bash
# 1. Pull latest dev and rebase
git checkout dev
git pull origin dev
git checkout feature/your-feature-name
git rebase dev

# 2. Push branch
git push origin feature/your-feature-name

# 3. Create PR (or merge locally if solo)
git checkout dev
git merge feature/your-feature-name
git push origin dev

# 4. Clean up
git branch -d feature/your-feature-name
```

## Pre-Push Checklist

Before pushing, always verify:

- [ ] **Backend runs**: `python -m backend.main` starts without errors
- [ ] **App builds**: `cd mobile && npx expo start` launches without errors
- [ ] **No secrets committed**: Check no `.env`, API keys, or credentials are staged
- [ ] **Gitignore is respected**: Run `git status` and verify no unwanted files

## Useful Git Commands

```bash
# View compact log
git log --oneline -15

# See what changed in last commit
git show --stat

# Undo last commit (keep changes staged)
git reset --soft HEAD~1

# Stash work in progress
git stash push -m "WIP: working on scan screen"
git stash pop

# See all branches
git branch -a

# Interactive rebase (squash messy commits)
git rebase -i HEAD~3

# Check what's ignored
git status --ignored
```

## Files to NEVER Commit

These are already in `.gitignore` but double-check:
- `venv/`, `node_modules/`, `.expo/`
- `.env` and `.env.*` files
- `*.sqlite`, `*.sqlite3`
- `backend/uploads/`, `ml/model/`, `uploads/`
- `*.key`, `*.pem`
- `__pycache__/`

## Tagging Releases

When deploying a build:
```bash
# Tag with version
git tag -a v1.0.0 -m "Release v1.0.0 - MVP demo"
git push origin v1.0.0
```
