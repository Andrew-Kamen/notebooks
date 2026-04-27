# Robust Modulated iSWAP Gate — Working Plan

## Goal
Design a robust iSWAP gate using:
- **Gaussian square coupling pulse** g_eff(t) with rising/falling edges and flat-top at 2π × 10 MHz
- **Microwave drives** (X₁, Y₁, X₂, Y₂) active **only during the flat-top**
- **No microwaves during rising/falling edges** (buffer regions)
- Robustness to ZI, IZ, ZZ errors via adjoint variational objective

## Experimental Setup

### Qubit frequencies and tuning
- δ₁₂ = ω₁_idle - ω₂_idle (idle frequency detuning, known)
- **Before the gate**: qubit 2 is flux-tuned ON RESONANCE with qubit 1 (ω₂_tuned = ω₁)
- **Flux tuning completes before any coupling is turned on**
- During the entire gate (rise + flat-top + fall), both qubits are at ω₁

### Microwave drive frequencies (configurable detunings)

The drive frequencies for each qubit are **configurable parameters**, not fixed.
Define the detuning of each qubit's microwave drive from the gate frequency (ω₁):

```
Δ_mw1 = ω_drive1 - ω₁    (qubit 1 drive detuning from gate frame)
Δ_mw2 = ω_drive2 - ω₁    (qubit 2 drive detuning from gate frame)
```

#### Example configurations

| Config | ω_drive1 | ω_drive2 | Δ_mw1 | Δ_mw2 | Notes |
|--------|----------|----------|-------|-------|-------|
| A: q2 at idle | ω₁ | ω₂_idle | 0 | -δ₁₂ | q1 resonant, q2 avoids crosstalk |
| B: both detuned (symmetric) | ω₁ - Δ | ω₁ + Δ | -Δ | +Δ | e.g., Δ = 2π×5 MHz; both off-resonant, symmetric |
| C: both at idle | ω₁ | ω₂_idle | 0 | -δ₁₂ | same as A (q1 is already at idle) |
| D: custom | ω₁ + Δ₁ | ω₁ + Δ₂ | Δ₁ | Δ₂ | fully general |

Config B is attractive for crosstalk: both drives are off-resonant with both qubits
by ±Δ, providing symmetric isolation.

## IQ Mixing and Connection to Gate-Frame Hamiltonian

This section connects the physical AWG/mixer signal chain to the operators
that appear in the gate-frame Hamiltonian. The derivation follows Appendix F
of arXiv:2410.22603.

### Signal chain: AWG → IQ mixer → qubit

**Step 1: AWG** produces two baseband signals modulated at intermediate frequency ω_IF:
```
S_I(t) = I(t) cos(ω_IF·t + φ)
S_Q(t) = -Q(t) sin(ω_IF·t + φ)
```
where I(t) and Q(t) are the pulse envelopes (what the optimizer outputs)
and φ is a programmable phase.

**Step 2: IQ mixer** upconverts with local oscillator at ω_LO:
```
Ω(t) = S_I(t) cos(ω_LO·t) + S_Q(t) sin(ω_LO·t)
```

This produces sidebands at ω± = ω_LO ± ω_IF. By choosing I(t) = Q(t) = A(t),
the lower sideband cancels, leaving a single-frequency drive:
```
Ω(t) = A(t) cos(ω+·t + φ)
```
with ω+ = ω_LO + ω_IF = ω_drive (the intended drive frequency).

**Step 3: In the gate frame** (rotating at ω₁ for both qubits), applying the RWA,
the drive on qubit i with detuning Δ_mw = ω_drive - ω₁ becomes (cf. Eq. F8-F9):
```
H_d(t) = Ã(t)/2 · [σ_x cos(Δ_mw·t + φ) - σ_y sin(Δ_mw·t + φ)]
```

When Δ_mw = 0 (resonant drive), this reduces to Eq. (F9):
```
H_d(t) = Ã(t)/2 · [cos(φ)σ_x - sin(φ)σ_y]
```
which is the standard σ_x / σ_y drive controlled by the phase φ.

### Mapping to optimizer variables

The optimizer works with two real envelopes per qubit: u_X(t) and u_Y(t).
These map to the physical I/Q envelopes via:

**Resonant drive (Δ_mw = 0):**
```
u_X(t) = I(t)/2  →  drives σ_x (XI or IX)
u_Y(t) = Q(t)/2  →  drives σ_y (YI or IY)
```
No time-dependent modulation in the gate frame. The AWG carrier is set to
ω_drive = ω₁ via ω_LO + ω_IF = ω₁.

**Off-resonant drive (Δ_mw ≠ 0):**
```
u_X(t)·[σ_x cos(Δ_mw·t) + σ_y sin(Δ_mw·t)]  (effective X drive)
u_Y(t)·[σ_y cos(Δ_mw·t) - σ_x sin(Δ_mw·t)]  (effective Y drive)
```
The cos/sin modulation arises because the drive carrier is not at ω₁.
The AWG still outputs smooth envelopes I(t), Q(t) — the detuning shows up
in the gate-frame Hamiltonian, not on the AWG.

**To change Δ_mw experimentally**: adjust ω_IF on the AWG. This shifts ω_drive
without changing the envelope shapes. No hardware modification needed.

### Sign convention note
The exact signs on the cos/sin terms depend on:
- Sign convention for the rotating frame (exp(-iωt) vs exp(+iωt))
- Definition of σ± and the RWA
- Whether the drive couples to charge (n̂) or flux (φ̂)

The signs shown above follow Appendix F of arXiv:2410.22603 (charge coupling,
b† → σ+, standard RWA). They should be verified against the specific
experimental setup before implementation.

## Gate Frame Hamiltonian

The "gate frame" rotates at ω₁ for both qubits. Since both qubits are on-resonance
during the gate, there is no IZ detuning term.

### During rising/falling edges (no microwaves)
```
H_edge(t) = g(t)(XX + YY)
```
Pure exchange coupling. No detuning, no modulation, no drives.

### During flat-top (general drive detunings)
```
H_flat(t) = g_eff(XX + YY)
           + u_X1(t)·[XI cos(Δ_mw1·t) + YI sin(Δ_mw1·t)]
           + u_Y1(t)·[YI cos(Δ_mw1·t) - XI sin(Δ_mw1·t)]
           + u_X2(t)·[IX cos(Δ_mw2·t) + IY sin(Δ_mw2·t)]
           + u_Y2(t)·[IY cos(Δ_mw2·t) - IX sin(Δ_mw2·t)]
```

- Coupling drift: time-independent g_eff(XX+YY)
- Each qubit's drives are modulated at their respective detuning from the gate frame
- When Δ_mw = 0 (resonant), the cos/sin terms reduce to identity (no modulation)
- **This is the segment we optimize for robustness**
- Sign convention follows Appendix F derivation above; verify for specific hardware

### Special case: Config A (Δ_mw1 = 0, Δ_mw2 = -δ₁₂)
```
H_flat(t) = g_eff(XX + YY)
           + u_X1(t)·XI + u_Y1(t)·YI
           + u_X2(t)·[IX cos(δ₁₂t) - IY sin(δ₁₂t)]
           + u_Y2(t)·[-IX sin(δ₁₂t) + IY cos(δ₁₂t)]
```
(substituting Δ_mw2 = -δ₁₂ into the general form; note sign flips from negative detuning)

## Pulse Structure

```
[flux tune q2]  |<-- buffer -->|<--- flat-top --->|<-- buffer -->|  [flux tune q2 back]
                |  g(t) rises  | g_eff, u_i ≠ 0  |  g(t) falls  |
                |  u_i = 0     | OPTIMIZE HERE    |  u_i = 0     |
                |  → V_rise    | → U_goal         |  → V_fall    |
```

Total unitary in gate frame: `U_total = V_fall · U_goal · V_rise`

We want: `U_total = iSWAP` (in gate frame)

Therefore: `U_goal = V_fall† · iSWAP · V_rise†`

## V_rise and V_fall are Simple (area-only)

Since H_edge = g(t)(XX+YY) is time-independent in structure (no modulation,
no detuning — qubits are already on-resonance):

```
V_rise = exp(-i A_rise (XX+YY))    where A_rise = ∫ g_rise(t) dt
V_fall = exp(-i A_fall (XX+YY))    where A_fall = ∫ g_fall(t) dt
```

V depends **only on the integrated area** ∫g(t)dt, not on pulse shape.
Both V_rise and V_fall commute with iSWAP (all are XX+YY rotations).

Therefore:
```
U_goal = exp(-i(π/4 - A_rise - A_fall)(XX+YY))
```
The flat-top target is just an iSWAP with a reduced rotation angle.

## Edge Bandwidth and Distortion

The Gaussian rise/fall of g_eff should be low-bandwidth enough that the
250 MHz hardware filter does not distort it. A Gaussian envelope with
time-width σ_rise has frequency bandwidth B_rise ≈ 1/(2πσ_rise).

| σ_rise | Rise bandwidth | Passes 250 MHz filter? |
|--------|---------------|----------------------|
| 0.5 ns | ~320 MHz | Slightly distorted |
| 1 ns   | ~160 MHz | Clean |
| 2 ns   | ~80 MHz  | Very clean |
| 4 ns   | ~40 MHz  | Untouched |

**Design rule**: choose σ_rise ≥ 1-2 ns so edges pass through the hardware
filter undistorted. This guarantees ∫g(t)dt matches the design value and
V_rise/V_fall are exactly as computed.

## Frame Transformation and Virtual Z

### Why a frame correction is needed
The optimization is performed in the gate frame (rotating at ω₁ for both qubits).
After the gate, each qubit returns to its own tracking frame (typically at its
idle frequency). The phase mismatch accumulated during the gate must be corrected.

### General frame correction
The correction depends on the **qubit tracking frames**, not the drive frequencies.
Regardless of drive configuration, the qubit tracking software maintains phase
references for each qubit. The correction accounts for the difference between
the gate frame (ω₁ for both) and each qubit's tracking frequency during the gate.

For qubit 1 (stays at ω₁_idle = ω₁, tracked at ω₁):
```
φ₁ = 0    (gate frame = tracking frame)
```

For qubit 2 (flux-tuned to ω₁, but tracked at ω₂_idle):
```
φ₂ = (ω₁ - ω₂_idle) · T_total = δ₁₂ · T_total
```

The correction is:
```
R(T) = I ⊗ Rz(δ₁₂ · T_total)
```

This is the same regardless of drive configuration (A, B, C, or D).
Drive detuning affects the Hamiltonian during optimization, not the frame correction.

**Note:** if the control software tracks qubit phases at the drive frequencies
rather than the qubit idle frequencies, additional corrections are needed.
This is an implementation detail that should be confirmed with the experimental
setup.

### Correct placement (does NOT commute)
```
U_lab = R(T_total) · U_gate
```

R(T_total) goes on the LEFT (applied last). It does NOT commute with iSWAP because:
```
[IZ, XX+YY] ≠ 0
```

Explicitly: [IZ, XX] = I⊗Z · X⊗X - X⊗X · I⊗Z = X⊗(ZX - XZ) = X⊗(2iY) = 2i·XY ≠ 0

So the order matters — you cannot swap R(T) and U_gate.

### Why this frame change is valid
The frame transformation is a standard rotating frame / interaction picture transformation.
It is exact (not an approximation like RWA). The physics:

1. **Unitary equivalence**: a frame change is a time-dependent unitary transformation
   |ψ⟩_gate = R†(t)|ψ⟩_lab. Both frames describe the same physics. Expectation values
   of any observable are identical when the observable is also transformed.

2. **The Schrödinger equation transforms covariantly**: if i∂_t|ψ⟩ = H|ψ⟩ in one frame,
   then i∂_t|ψ̃⟩ = H̃|ψ̃⟩ in the other, with H̃ = R†HR - iR†∂_tR. This is exact.

3. **Gate fidelity is frame-independent**: F = |Tr(U_goal† U)|/d is the same in both
   frames as long as U_goal and U are expressed in the same frame. We optimize in the
   gate frame (where the Hamiltonian is simpler), then transform the result to the lab
   frame at the end.

4. **The correction is a virtual Z gate**: Rz on qubit 2, implemented as a software
   frame update on qubit 2's drive phase — zero duration, zero error, exact.

### Summary: lab-frame gate sequence
1. Flux-tune qubit 2 on-resonance with qubit 1
2. V_rise: Gaussian rise of g_eff (pure XX+YY, area = A_rise)
3. U_goal: optimized flat-top microwaves (robust iSWAP minus edge areas)
4. V_fall: Gaussian fall of g_eff (pure XX+YY, area = A_fall)
5. Flux-tune qubit 2 back to idle
6. Virtual Rz(δ₁₂ · T_total) on qubit 2

## Distortion Compensation

Since V_rise and V_fall depend only on ∫g(t)dt (not pulse shape):
- If distortion changes g(t) shape but preserves area → V unchanged → no correction needed
- If distortion changes area → adjust buffer padding to match nominal area
- IZ phase from any timing changes → absorb into virtual Z gate
- **No need to re-optimize the flat-top microwaves**

Key requirement: flux pulse must stabilize (true flat-top) before microwaves turn on.

## Implementation Steps

### Phase 1: Compute edge areas and U_goal target
- [ ] Define Gaussian rise/fall envelope parameters (σ_rise, buffer duration)
- [ ] Choose σ_rise ≥ 2 ns for clean edges (below 250 MHz filter bandwidth)
- [ ] Compute A_rise = ∫g_rise(t)dt, A_fall = ∫g_fall(t)dt
- [ ] Target rotation angle: θ_goal = π/4 - A_rise - A_fall
- [ ] U_goal = exp(-i θ_goal (XX+YY))
- [ ] Sanity check: θ_goal > 0 (edges don't over-rotate past iSWAP)

### Phase 2: Set up Piccolo optimization for flat-top
- [ ] Choose drive detuning configuration (A, B, C, or D)
- [ ] Choose Δ_mw1, Δ_mw2 values
- [ ] Gate frame Hamiltonian:
  - H_drift = g_eff(XX+YY) (time-independent)
  - H_drives: 4 channels with cos/sin modulation at Δ_mw1, Δ_mw2
  - When Δ_mw = 0, drives are simply XI, YI (no modulation)
- [ ] Target: U_goal = exp(-i θ_goal (XX+YY))
- [ ] Error operators: [ZI, IZ, ZZ]
- [ ] Variational problem with robustness (Q_r, hard fidelity constraint)
- [ ] Control bounds: |u_i| ≤ TBD (experimentally feasible)
- [ ] Bandwidth constraints: dda_bound matched to 250 MHz filter

### Phase 3: Verification
- [ ] Full simulation: V_fall · U_goal · V_rise ≈ iSWAP (gate frame)
- [ ] Apply R(T_total) → verify lab-frame gate
- [ ] Fidelity vs ε sweep for ZI, IZ, ZZ errors
- [ ] Gaussian filter (250 MHz) on microwaves → re-verify
- [ ] Compare robust vs default (no microwaves, just coupling) solution

### Phase 4: Experimental translation
- [ ] Convert gate-frame controls to AWG waveforms:
  - Optimizer outputs: u_X(t), u_Y(t) (baseband envelopes per qubit)
  - AWG signals: modulate at ω_IF chosen so ω_LO + ω_IF = ω_drive
  - ω_drive = ω₁ + Δ_mw for each qubit
  - To change Δ_mw: adjust ω_IF only (no hardware changes)
- [ ] Compute virtual Z correction: Rz(δ₁₂ · T_total) on qubit 2
- [ ] Verify with full lab-frame simulation including virtual Z
- [ ] Confirm sign conventions match experimental control software

## Open Questions

1. **Drive detuning configuration**
   - Which config (A, B, C, D) to start with?
   - Config A is simplest (q1 resonant). Config B (symmetric ±Δ) may be better
     for crosstalk. Can sweep configurations to compare.
   - Different configs change the optimizer's search space but not the frame correction.

2. **Sign conventions**
   - The cos/sin signs in the gate-frame Hamiltonian depend on frame rotation
     direction and coupling operator. Must verify against experimental control
     software before hardware implementation.
   - For simulation/optimization, any consistent convention works — just be
     consistent between the optimizer and the verification simulation.

3. **Qubit drive modulation in Piccolo**
   - The drives appear as time-dependent operators in the gate frame
   - Options: (a) include cos/sin modulation in the Hamiltonian explicitly,
     (b) use a bilinear time-dependent drive formulation
   - The existing modulate_iswap notebooks have examples of this

4. **Experimental parameters to confirm**
   - δ₁₂ value (idle detuning)
   - g_eff flat-top value (2π × 10 MHz?)
   - Gaussian σ_rise for rise/fall
   - Buffer duration
   - Microwave amplitude limits
   - Target fidelity threshold
   - Qubit tracking frame convention (idle freq vs drive freq?)

5. **Flat-top duration**
   - Determined by θ_goal = π/4 - A_rise - A_fall and g_eff:
     T_flat = θ_goal / g_eff (if no microwaves contributed to XX+YY)
   - But microwaves can also contribute — optimizer may find shorter/longer T_flat
   - Should T_flat be fixed or free?
