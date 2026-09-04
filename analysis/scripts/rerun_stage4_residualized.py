"""Re-run Stage 4 (effect matrix -> null calibration -> BH-FDR calibration)
against the cell-cycle-residualized expression matrix produced by
`cell_cycle.py`, to test whether conditioning on cell-cycle state shrinks the
implausible k^out_g values flagged in LOG.md's Checkpoint D0 discussion
(some targets "significant" against ~97% of the filtered genome).

Writes a parallel set of outputs (suffixed `_resid`) rather than overwriting
the originals, so the raw and residualized k^out_g can be compared directly.
"""

from __future__ import annotations

from pathlib import Path

from effect_matrix import SCRATCH, compute_effect_matrix
from null_calibration import compute_null
from calibrate_effects import calibrate

RESID_PATH = SCRATCH / "X_resid_genemajor.npy"


def main():
    if not RESID_PATH.exists():
        raise SystemExit(f"{RESID_PATH} not found -- run cell_cycle.py first")

    print("=== Stage 4 (residualized): effect matrix ===")
    compute_effect_matrix(
        genemajor_path=RESID_PATH,
        out_name="E_matrix_resid.npy",
        meta_name="E_matrix_meta_resid.pkl",
    )

    print("=== Stage 4 (residualized): null calibration ===")
    compute_null(
        genemajor_path=RESID_PATH,
        mean_name="null_mean_by_size_resid.npy",
        sd_name="null_sd_by_size_resid.npy",
        meta_name="null_meta_resid.pkl",
    )

    print("=== Stage 4 (residualized): BH-FDR calibration + k_out ===")
    calibrate(
        load_kwargs=dict(
            e_matrix_name="E_matrix_resid.npy",
            e_meta_name="E_matrix_meta_resid.pkl",
            null_mean_name="null_mean_by_size_resid.npy",
            null_sd_name="null_sd_by_size_resid.npy",
            null_meta_name="null_meta_resid.pkl",
        ),
        q_name="E_qvalues_resid.npy",
        z_name="E_zscores_resid.npy",
        kout_name="k_out_resid.csv",
    )


if __name__ == "__main__":
    main()
