# -*- coding: utf-8 -*-
"""Tests for checkpoint architecture compatibility checks.

Covers `_config_to_flat_arch` and `is_checkpoint_compatible_with_model_config`
in `elitefurretai.rl.learners`. The synthetic `_gen` key derived inside
`_config_to_flat_arch` makes cross-gen checkpoints architecturally incompatible
while letting same-gen checkpoints with different format strings load (the
embedder vocab is gen-keyed, not format-keyed).
"""


def test_config_to_flat_arch_derives_gen_from_battle_formats():
    """_config_to_flat_arch synthesizes _gen from the new battle_formats schema."""
    from elitefurretai.rl.learners import _config_to_flat_arch

    flat = _config_to_flat_arch(
        {
            "curriculum": {"battle_formats": {"gen9vgc2024regg": 1.0}},
            "training": {"embedder_feature_set": "raw"},
        }
    )
    assert flat["_gen"] == "9"


def test_checkpoint_compatibility_rejects_cross_gen(tmp_path):
    """A checkpoint built on gen8 should be rejected by a gen9 model config."""
    import torch

    from elitefurretai.rl.learners import is_checkpoint_compatible_with_model_config

    gen8_ckpt = tmp_path / "gen8.pt"
    torch.save(
        {"config": {"curriculum": {"battle_formats": {"gen8vgc2022": 1.0}}}},
        gen8_ckpt,
    )

    gen9_model_config = {"curriculum": {"battle_formats": {"gen9vgc2024regg": 1.0}}}

    assert not is_checkpoint_compatible_with_model_config(
        str(gen8_ckpt), gen9_model_config
    )


def test_checkpoint_compatibility_accepts_same_gen_different_formats(tmp_path):
    """Two checkpoints at the same gen but different format dicts ARE compatible."""
    import torch

    from elitefurretai.rl.learners import is_checkpoint_compatible_with_model_config

    regg_ckpt = tmp_path / "regg.pt"
    torch.save(
        {"config": {"curriculum": {"battle_formats": {"gen9vgc2024regg": 1.0}}}},
        regg_ckpt,
    )

    regh_model_config = {"curriculum": {"battle_formats": {"gen9vgc2024regh": 1.0}}}

    assert is_checkpoint_compatible_with_model_config(str(regg_ckpt), regh_model_config)
