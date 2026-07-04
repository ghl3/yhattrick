"""Tests for the shared row-building primitives (yhattrick.models.generative_features).

The orientation math — the attacking-team zone flip and the lead sign — is load-bearing for every WAR
row and easy to get subtly wrong, so these pure functions are locked directly.
"""

from types import SimpleNamespace

from yhattrick.models import generative_features as F


def test_zone_flips_for_away_team():
    # Home O-zone faceoff: home attacker starts in O, away attacker starts in D.
    assert F.zone(True, "faceoff", "O") == (1.0, 0.0)  # home attacks → offensive
    assert F.zone(False, "faceoff", "O") == (0.0, 1.0)  # away attacks the same faceoff → defensive
    assert F.zone(True, "faceoff", "D") == (0.0, 1.0)
    assert F.zone(False, "faceoff", "D") == (1.0, 0.0)
    # Non-faceoff / neutral-zone starts carry no zone signal.
    assert F.zone(True, "fly", "O") == (0.0, 0.0)
    assert F.zone(True, "faceoff", "N") == (0.0, 0.0)


def test_base_ctx_orients_lead_and_home():
    # home_lead is home-perspective; it must flip sign for the away attacker.
    home_leading = SimpleNamespace(start_type="fly", start_zone="N", home_lead=2)
    # home attacker, home leading → lead col set, trail clear, home=1
    assert F.base_ctx(True, home_leading) == [1.0, 0.0, 0.0, 0.0, 1.0]
    # away attacker in the same stint → trailing, home=0
    assert F.base_ctx(False, home_leading) == [0.0, 0.0, 0.0, 1.0, 0.0]
    # tied → neither trail nor lead
    tied = SimpleNamespace(start_type="fly", start_zone="N", home_lead=0)
    assert F.base_ctx(True, tied) == [1.0, 0.0, 0.0, 0.0, 0.0]


def test_ratectx_names_match_base_ctx_width():
    tied = SimpleNamespace(start_type="fly", start_zone="N", home_lead=0)
    assert len(F.RATE_CTX) == len(F.base_ctx(True, tied)) == 5
    assert F.RATE_CTX == ["home", "ozone", "dzone", "trail", "lead"]
