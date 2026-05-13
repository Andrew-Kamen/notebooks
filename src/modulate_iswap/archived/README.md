# Archived scripts and outputs

Nothing here has been deleted — these are historical / exploratory files moved out of the main `modulate_iswap/` folder to reduce clutter. Active scripts remain in the parent directory.

## Subfolder contents

### `cubic_modulate_iswap_versions/`
The 8 iterative versions (`v1`–`v8`) of the original cubic-spline iSWAP exploration plus their output PNGs. These predate the current `*_rollout.jl` / `*_3level_rollout.jl` workflow.

### `early_andrew_notebooks/`
Earliest exploratory notebooks before the modulated-iSWAP pipeline solidified: `03_andrew*.ipynb`, `convergence.ipynb`, `modulate_iswap.ipynb`, `robust_hadamard_variational_rollout.ipynb`. Useful for understanding the historical progression but superseded.

### `free_phase_explorations/`
The sincos / free-phase parameterization branch — `robust_iswap_free_phase*.ipynb`, `robust_iswap_sincos_freephase_2.ipynb`, the corresponding output dirs, and `sincos_free_phase_plan.md`. Tried as an alternative to fixed-phase optimization.

### `older_iswap_variants/`
Older variants of `robust_iswap_detuned*` — different time scales (100 ns, 240 ns), buffer widths (0 ns, 10 ns, 15 ns edge), coupling strengths (10 MHz), and the original 150 ns + `.bak` files. Also their output directories. Superseded by the current `_5nsbuf_*` and `_1MHz_300ns_*` scripts.

### `old_ipopt_logs/`
Ipopt log files from runs whose output directories have moved here. The current logs (for the active runs) stay in the parent folder.

### `old_profile_outputs/`
`profile.pb.gz`, `profile_rollout_gradient.pb.gz`, `profile_rollout_3level.out.log` — outputs from earlier profiling sessions. The active profiling script (`profile_rollout_3level.jl`) remains in the parent folder; these are just its old outputs.

## Where to find current work

See the parent `modulate_iswap/` folder and `SESSION_HANDOFF.md` there for the active scripts, run directories, and methodology summary.
