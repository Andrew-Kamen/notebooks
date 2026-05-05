# Version Control — robust_control_sam dev install setup

This project develops against forked/modified versions of `Piccolo.jl` and `DirectTrajOpt.jl`. The modifications are kept on **private GitHub mirrors** so they can be synced across machines without making the work public.

## Repo layout

| Repo | Public origin | Private mirror | Active branch | Notes |
|---|---|---|---|---|
| `Piccolo.jl` | `harmoniqs/Piccolo.jl` | `Andrew-Kamen/Piccolo-private` | `modulate` | Has `VariationalSplinePulseProblem`, `free_phase`, cubic Hermite NLP dynamics |
| `DirectTrajOpt.jl` | `harmoniqs/DirectTrajOpt.jl` | `Andrew-Kamen/DirectTrajOpt-private` | `main` | Has `CubicHermitePathConstraint`, time-dependent bilinear integrator |
| `NamedTrajectories.jl` | public — registered version is fine | — | — | No local modifications |
| `PiccoloQuantumObjects.jl` | public — registered version is fine | — | — | No local modifications |

## Remote setup per machine

### Laptop (where `origin` = public, `private` = mirror)

```
origin   → https://github.com/harmoniqs/<Repo>.git           (public, read-only for me)
private  → https://github.com/Andrew-Kamen/<Repo>-private.git (write access)
```

A bare `git push` here would target `origin` (public) and either fail or leak — **always specify `private`**.

### SSH machine (where `origin` = the private mirror)

Cloned directly from the private repo, so:

```
origin   → https://github.com/Andrew-Kamen/<Repo>-private.git (write access)
```

A bare `git push` works correctly here — `origin` is already private.

## Day-to-day workflow

### Push from laptop

```bash
# Piccolo.jl
cd Piccolo.jl
git add -A
git commit -m "..."
git push private modulate

# DirectTrajOpt.jl
cd DirectTrajOpt.jl
git add -A
git commit -m "..."
git push private main
```

### Push from SSH machine

```bash
git add -A
git commit -m "..."
git push        # origin = private mirror
```

### Sync (pull other machine's changes)

On the laptop:
```bash
cd Piccolo.jl       && git pull private modulate
cd DirectTrajOpt.jl && git pull private main
```

On the SSH machine:
```bash
git pull
```

### Pull upstream harmoniqs updates onto the modified branch

```bash
cd Piccolo.jl
git fetch origin
git merge origin/main          # or origin/modulate, depending on what upstream advanced
git push private modulate
```

(Same pattern for `DirectTrajOpt.jl` with the `main` branch.)

## Initial clone on a new machine

```bash
git clone https://github.com/Andrew-Kamen/Piccolo-private.git Piccolo.jl
git clone https://github.com/Andrew-Kamen/DirectTrajOpt-private.git DirectTrajOpt.jl
```

Default branches are already correct (`modulate` and `main` respectively) — no checkout needed.

For HTTPS auth, use a fine-grained personal access token (https://github.com/settings/tokens?type=beta) as the password. For SSH, generate a key with `ssh-keygen -t ed25519`, paste `~/.ssh/id_ed25519.pub` into GitHub → Settings → SSH keys, and use `git@github.com:...` URLs.

## Directory layout assumed by the notebooks

The notebooks in `src/modulate_iswap/` use `Pkg.develop(path=...)` with paths like:

```julia
piccolo_path       = joinpath(@__DIR__, "..", "..", "..", "Piccolo.jl")
directtrajopt_path = joinpath(@__DIR__, "..", "..", "..", "DirectTrajOpt.jl")
```

So `Piccolo.jl`, `DirectTrajOpt.jl`, and `robust_control_sam` must be **siblings** in whatever parent directory you choose:

```
<parent>/
├── Piccolo.jl/
├── DirectTrajOpt.jl/
└── robust_control_sam/
    └── src/
        └── modulate_iswap/
            └── *.ipynb
```

If a remote machine has a different layout, edit the `joinpath` paths in the notebook.

## Foot-guns

- **Don't `git push` without specifying `private` on the laptop.** It would push to `harmoniqs/...` (public) and either fail or leak.
- **Both machines must commit before pushing.** `git push` only ships committed history.
- **iCloud Drive is not a sync mechanism for git.** The laptop happens to live in iCloud, but treat git pushes as the source of truth across machines — opening a notebook on a second iCloud-synced Mac while the first is mid-write can corrupt `.git`.
- **The private repo on GitHub may have leftover branches** (dependabot, PR refs) from when it was created via fork/import rather than empty. Harmless, but you can clean them up via the GitHub web UI if desired.
