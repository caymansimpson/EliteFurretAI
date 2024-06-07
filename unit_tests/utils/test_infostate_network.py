# -*- coding: utf-8 -*-
"""Unit Tests to ensure InfostateNetwork is good
"""

# -*- coding: utf-8 -*-
import pytest

from elitefurretai.utils.infostate_network import InfostateNetwork


# Will fail once I implement, which is intended
def test_embed_team_preview(double_battle_json):
    with pytest.raises(NotImplementedError):
        frisk = InfostateNetwork()
        frisk.get_speed_range()
