# Scoping note: restoring the missing `Im(a_{L,1})` degree of freedom

*Written 2026-09-01 as a plan. **IMPLEMENTED 2026-09-06** — Steps 1-5 are done and
the suite is green (130 passed, 1 skipped, ruff clean); Step 0 was skipped by
decision (see "What was actually done" at the end) and Step 6, the re-run, is the
outstanding work. The body below is kept as written so the plan and the outcome can
be compared. `ROADMAP.md` carries the live status.*

## The defect in one paragraph

`alm_utils.py::splittosingularalm` writes `complex(real, 0)` when `m == 0 or
m == 1`. Forcing `Im(a_{L,0}) = 0` is **correct and required** — a real field's
`m = 0` coefficient is real. Forcing `Im(a_{L,1}) = 0` is **not**: it is an
unintended consequence of the two cases sharing a branch. Every multipole
therefore carries `1 + 1 + 2(L-1) = 2L` real degrees of freedom where a general
real scalar field on the sphere has `2L+1`. The model **cannot represent a
general sky**: the fraction of missing modes is `1/(2L+1)` — 20% at ℓ=2, 9% at
ℓ=5, 4.8% at ℓ=10, 1.6% at ℓ=30, 0.8% at ℓ=63.

This is a *parameterisation* defect, not a statistics one. The inverse-Gamma
conditionals are now correct **for the packing as it stands** (`k_L = 2L`,
derived via `alm_utils.invgamma_shape_for_spectrum` since 2026-08-31), so
nothing is currently inconsistent — the sampler correctly samples a model that
is missing one mode per multipole.

## Why it is worth fixing, and why now

1. **It is the largest known open defect.** Everything else on the critical path
   is either validated or has a measured status.
2. **Its ℓ-dependence matches the project's chronic weak spot.** Low-ℓ φ has
   been the failure bin at every scale since lmax=128 — R̂ up to 1.50 in
   `[2,10)` in *every* configuration run, including the ones that otherwise
   pass. The missing-mode fraction is largest in exactly that bin. This is
   suggestive, **not** established: no test has yet linked the two, and
   designing that test is step 0 below.
3. **It would vindicate the original shape parameter.** With `2L+1` dof restored
   the flat-prior conditional becomes `InvGamma(L - 0.5, S_L/2)` again — the
   form the code used before 2026-08-31. Anyone reading the git history without
   this note will otherwise conclude the fix was reverted.
4. **A referee can check it in five minutes.** "Your parameterisation has 2L
   degrees of freedom per multipole" is a one-line objection to a paper whose
   entire claim is exactness.

## Blast radius, measured not estimated

Counted 2026-09-01 with `grep -rn "n_imag|(lmax-2)*(lmax-1)"` over `--include=*.py`,
excluding `.venv`:

| Surface | Sites | Files | Notes |
|---|---|---|---|
| Core package `diffcmb/diffcmb/` | 55 | 4 | `samplers.py` 37, `lensing.py` 12, `model.py` 4, `power.py` 2 |
| `scripts/` | ~150 | 24 | production/validation infrastructure |
| `tests/` | ~70 | 9 | must be re-derived, not relaxed |
| **Total** | **276** | **37** | |

The dominant literal is `n_imag = (lmax - 2) * (lmax - 1) // 2`, appearing
verbatim **75 times**, plus 7 `LMAX` variants and one `args.lmax` variant. Three
sites already use the equivalent `sum(l - 1 for l in range(2, lmax))`, and
`tests/test_phi_ancillary_move.py` has a local `_packed_sizes(lmax)` helper used
6 times — i.e. **the right abstraction already exists in the test suite and was
never promoted into the package.**

Saved output that the change invalidates: **224 `.npz` files, 12 GB**, of which
**107 are checkpoints**. Checkpoints are the hard part — `run_gibbs_chain`
resumes automatically from an existing checkpoint file, so a layout change makes
every in-flight campaign silently resume into a vector of the wrong length (best
case a shape error; worst case a silent broadcast).

## Proposed staged plan

**Step 0 — establish that it matters, before spending the effort.** Cheap, no
new sampling. Take an existing lmax=64 truth, project it onto the packed basis
(zeroing `Im(a_{L,1})`), and measure how much of the low-ℓ residual the missing
mode accounts for; separately, check whether the `[2,10)` R̂ pathology reproduces
in a *toy* sampler with the mode artificially removed and disappears with it
present. **If the pathology does not track the missing mode, this stays a
correctness fix and drops below the differentiator figures in priority.** Do not
skip this step: the project's standing lesson is that a plausible mechanism is
not a measured one.

**Step 1 — centralise the layout, changing nothing.** Add
`alm_utils.packed_sizes(lmax) -> (n_real, n_imag)` and
`alm_utils.packed_length(lmax)` (promoting the existing test helper), then
mechanically replace all 276 sites. This commit must be **bit-identical** in
behaviour: assert `packed_sizes` reproduces the old formula for lmax 4..512 and
confirm the full suite is unchanged. Nothing else in the plan is safe until the
layout has exactly one definition, and this step is independently valuable even
if Step 0 says stop.

**Step 2 — change the packing behind that single definition.** Move the
`m == 1` case out of the `m == 0` branch in `splittosingularalm`,
`splittosingularalm_tf`, `_alm_scatter_indices`, and the inverse
`singulartosplitalm`/`almtosplit` path. `n_imag` becomes
`Σ_{L=2}^{lmax-1} L = (lmax - 2)(lmax + 1)/2`, i.e. the old
`(lmax - 2)(lmax - 1)/2` plus exactly one extra imaginary slot per multipole
(`+ (lmax - 2)` in total; verified numerically for lmax = 4, 8, 64, 128). Everything downstream follows from `packed_sizes`, which is the point
of Step 1.

**Step 3 — let the dof follow automatically.** `packed_dof_per_multipole` must
return `2L+1` *because it counts the packing*, not because a constant was
edited. It was written for exactly this. Verify `<S_L>/C = 2L+1` and
`S_L/C ~ χ²_{2L+1}` in mean **and** variance, for the alm packing and the φ
packing separately — the same two checks that established `2L` in the first
place. The `nu > 0` properness threshold must be **re-derived**, not assumed to
carry over: that is the exact mistake that produced the stale `nu > 2`.

**Step 4 — re-derive the tests, and expect the suite to go red first.** Tests
that hardcode the old `n_imag` are *specifications of the defect*, not
regressions. Re-derive each; do not relax any. `test_alm_format_round_trip`
will keep passing throughout and proves nothing — a round-trip cannot see a
consistent restriction, exactly as it could not see the 2026-08-24 ordering bug.
The tests that matter are the absolute-(L,m) ones and a **new** test asserting
that a general `hp.synalm` draw survives pack→unpack with no power loss at any
multipole, which is precisely what fails today.

**Step 5 — version the checkpoints.** Write a `packing_version` field into every
checkpoint and chain `.npz`, and have the resume path refuse to load a
mismatched one with a clear message. Without this, Step 2 turns 107 existing
checkpoints into silent corruption. This is the single highest-risk item in the
plan and it is cheap.

**Step 6 — re-run, and treat every φ number as invalid until it is.** Every
saved chain predates the layout. The minimum re-run to restore the paper's
claims is the Option 1 ensemble (12 × lmax=64, Block 4 off) plus whichever
proper-prior configuration is current by then.

## Sequencing

Steps 0 and 1 are safe to do at any time and Step 1 is worth doing regardless of
the outcome of Step 0. Steps 2-6 should **not** start while a φ campaign is in
flight (currently job 11912088), because Step 5's checkpoint versioning is what
makes them safe and it lands after the change that needs it. The natural window
is immediately after the Option 2 long-trajectory question is settled.

## Standing caution

Restoring the mode changes what the model *is*, so it invalidates comparisons
against every number in `achievements.md` that was measured through the old
packing — which is all of them. Record the pre-change values of the headline
statistics (Option 1's φ 0.4688 / alm 0.5312) before starting, so the post-change
run can be compared against something rather than merely re-measured.


---

## What was actually done (2026-09-06)

**Step 0 was skipped.** The note argued for measuring that the missing mode
tracks the low-ℓ pathology before spending the effort. That measurement was
never made; the change was carried out as a correctness fix on its own merits.
The consequence to hold onto: **there is still no evidence linking the missing
dof to the `[10,30)` SBC residual or the chronic `[2,10)` R̂ pathology.** If the
post-change ensemble still shows them, that is not a surprise and not a failure
of this change — it closes a referee-visible correctness hole regardless.

**Steps 1-3, done as one change** rather than the planned bit-identical Step 1
followed by a behavioural Step 2. `alm_utils.packed_sizes`/`packed_length` were
added and all 32 files carrying the inline literal now derive from them;
`n_imag` is `(lmax-2)(lmax+1)/2`. `packed_dof_per_multipole` returns `2L+1`
because it counts the packing, and `invgamma_shape_for_spectrum` follows, so
Blocks 1 and 4 moved without either conditional being edited by hand.

**Two library sites the note did not enumerate**, both found by the suite:
- `alm_utils.hpalminit` realified the first `2*lmax - 1` healpy entries — the
  m=0 **and** m=1 blocks — so it imposed the defect on the *data and prior*
  alms, not only on the sampled vector. Now `lmax` entries (m=0 only).
- `lensing.sample_phi_amplitude_rescale`'s acceptance ratio is packing-
  dependent: the Jacobian exponent is `n_L + 2 - k_L`, which is `1` for
  `n_L = k_L = 2L` and `2` for `2L+1`. The brute-force test in
  `tests/test_phi_ancillary_move.py` had the matching `2L+2` hardcoded, so it
  agreed with the code and both were wrong together until the stationary-
  distribution test (which uses an independent exact Gibbs reference) caught
  the shift. That is the case for keeping an *independent* reference chain
  around: the analytic-vs-analytic check could not see it.

**Step 4, tests re-derived not relaxed.** The specs of the defect that had to
change: `test_splittosingularalm_roundtrip`, `test_almtomap_tf_matches_healpy`,
`test_alm_hp_to_packed_uses_true_healpy_ordering`,
`test_packed_sl_matches_healpy_power_per_multipole` (it used to zero
`Im(a_{L,1})` on the *reference* side), `test_packed_dof_per_multipole_...`
(`2L` → `2L+1`), and the two conjugate-prior shapes in `test_lensing.py`. The
new test the note asked for is
`test_general_synalm_draw_survives_pack_unpack_with_no_power_loss` — an
unrestricted `hp.synalm` draw through pack→unpack, exact per-multipole power.
Measured on one draw under the old rule, the lost fraction at L=2..6 was
0.356/0.008/0.222/0.002/0.047 against an expected `1/(2L+1)` of
0.200/0.143/0.111/0.091/0.077 — realization scatter, but unmistakably nonzero.

**Step 5, checkpoint versioning, is in.** `samplers.PACKING_VERSION = 2` is
written into every checkpoint; resume refuses on a version mismatch *and* on a
vector-length mismatch, with a test (`test_checkpoint_packing_version_mismatch`).
Checkpoints written before this carry no `packing_version` field and are read as
version 1, so they refuse rather than silently resuming.

**Step 6 is outstanding.** Every saved chain, every `.npz` in
`results/analysis/`, and every φ number in `achievements.md` and
`docs/paper/main.tex` predates the layout and describes a different model.
