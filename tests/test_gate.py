"""Tests for the CGAE core engine."""

import pytest
from cgae_engine.gate import GateFunction, RobustnessVector, Tier, TierThresholds


class TestRobustnessVector:
    def test_valid_vector(self):
        r = RobustnessVector(cc=0.5, er=0.6, as_=0.7, ih=0.8)
        assert r.cc == 0.5
        assert r.weakest == 0.5

    def test_out_of_range_raises(self):
        with pytest.raises(ValueError):
            RobustnessVector(cc=1.5, er=0.5, as_=0.5, ih=0.5)

    def test_primary_dimensions(self):
        r = RobustnessVector(cc=0.3, er=0.5, as_=0.4, ih=0.9)
        assert r.primary == (0.3, 0.5, 0.4)
        assert r.weakest == 0.3


class TestGateFunction:
    def setup_method(self):
        self.gate = GateFunction()

    def test_zero_robustness_gives_t0(self):
        r = RobustnessVector(cc=0.0, er=0.0, as_=0.0, ih=0.0)
        assert self.gate.evaluate(r) == Tier.T0

    def test_low_ih_forces_t0(self):
        r = RobustnessVector(cc=0.9, er=0.9, as_=0.9, ih=0.3)
        assert self.gate.evaluate(r) == Tier.T0

    def test_weakest_link(self):
        # High CC and ER but low AS should be gated by AS
        r = RobustnessVector(cc=0.9, er=0.9, as_=0.3, ih=0.9)
        tier = self.gate.evaluate(r)
        assert tier == Tier.T1  # AS=0.3 >= 0.25 (T1 threshold)

    def test_t5_requires_all_high(self):
        r = RobustnessVector(cc=0.95, er=0.95, as_=0.90, ih=0.95)
        assert self.gate.evaluate(r) == Tier.T5

    def test_evaluate_with_detail(self):
        r = RobustnessVector(cc=0.72, er=0.68, as_=0.55, ih=0.82)
        d = self.gate.evaluate_with_detail(r)
        assert "tier" in d
        assert "binding_dimension" in d
        assert d["ih_pass"] is True

    def test_chain_tier(self):
        r1 = RobustnessVector(cc=0.9, er=0.9, as_=0.9, ih=0.9)
        r2 = RobustnessVector(cc=0.4, er=0.4, as_=0.3, ih=0.6)
        chain = self.gate.chain_tier([r1, r2])
        assert chain <= self.gate.evaluate(r1)
        assert chain == self.gate.evaluate(r2)

    def test_budget_ceiling_increases_with_tier(self):
        ceilings = [self.gate.budget_ceiling(Tier(i)) for i in range(6)]
        for i in range(1, 6):
            assert ceilings[i] > ceilings[i - 1]


class TestTierThresholds:
    def test_default_thresholds_valid(self):
        t = TierThresholds()
        assert len(t.cc) == 6
        assert t.cc[0] == 0.0

    def test_non_increasing_raises(self):
        with pytest.raises(ValueError):
            TierThresholds(cc=[0.0, 0.5, 0.4, 0.6, 0.8, 0.9])
