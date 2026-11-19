"""Architecture-compatible trainer names for inference with the study checkpoints.

The archived training script created each class dynamically without overriding
nnUNetTrainer methods. Expose those same names to nnU-Net's external trainer
discovery for predictions in a fresh process.
"""

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


def _register(seed: int, suffix: str = "") -> None:
    name = f"nnUNetTrainerPedGlioSeed{seed}{suffix}"
    globals()[name] = type(name, (nnUNetTrainer,), {"__module__": __name__})


for pediatric_seed in (2026, 2027, 2028):
    for condition in ("", "AdultInit", "SourceOnlyPlan"):
        _register(pediatric_seed, condition)

_register(2026, "AdultPre")
