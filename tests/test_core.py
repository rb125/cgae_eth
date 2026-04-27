"""Tests for registry, contracts, and economy."""

import pytest
from pathlib import Path
from cgae_engine.gate import RobustnessVector, Tier, GateFunction
from cgae_engine.registry import AgentRegistry, AgentStatus
from cgae_engine.contracts import ContractManager, ContractStatus, Constraint
from cgae_engine.economy import Economy, EconomyConfig
from cgae_engine.temporal import TemporalDecay


class TestRegistry:
    def setup_method(self):
        self.registry = AgentRegistry()

    def test_register_agent(self):
        record = self.registry.register("test-model", {"model": "test"}, initial_balance=1.0)
        assert record.status == AgentStatus.PENDING
        assert record.balance == 1.0
        assert record.model_name == "test-model"

    def test_certify_activates_agent(self):
        record = self.registry.register("m", {})
        r = RobustnessVector(cc=0.7, er=0.7, as_=0.6, ih=0.8)
        self.registry.certify(record.agent_id, r)
        assert record.status == AgentStatus.ACTIVE
        assert record.current_tier.value >= 1

    def test_low_ih_suspends(self):
        record = self.registry.register("m", {})
        r = RobustnessVector(cc=0.9, er=0.9, as_=0.9, ih=0.3)
        self.registry.certify(record.agent_id, r)
        assert record.status == AgentStatus.SUSPENDED

    def test_demote(self):
        record = self.registry.register("m", {})
        r_high = RobustnessVector(cc=0.8, er=0.8, as_=0.7, ih=0.9)
        self.registry.certify(record.agent_id, r_high)
        old_tier = record.current_tier

        r_low = RobustnessVector(cc=0.3, er=0.3, as_=0.3, ih=0.6)
        new_tier = self.registry.demote(record.agent_id, r_low)
        assert new_tier < old_tier

    def test_tier_distribution(self):
        for i in range(5):
            rec = self.registry.register(f"m{i}", {})
            r = RobustnessVector(cc=0.5, er=0.5, as_=0.5, ih=0.7)
            self.registry.certify(rec.agent_id, r)
        dist = self.registry.tier_distribution()
        assert sum(dist.values()) == 5


class TestContracts:
    def setup_method(self):
        self.cm = ContractManager()

    def test_create_contract(self):
        c = self.cm.create_contract(
            "test", [Constraint("c1", "test", lambda x: True)],
            Tier.T1, reward=0.1, penalty=0.05, issuer_id="sys", deadline=100.0,
        )
        assert c.status == ContractStatus.OPEN
        assert c.reward == 0.1

    def test_assign_enforces_tier(self):
        c = self.cm.create_contract(
            "test", [], Tier.T3, reward=0.1, penalty=0.05, issuer_id="sys", deadline=100.0,
        )
        # T1 agent can't accept T3 contract
        ok = self.cm.assign_contract(c.contract_id, "agent1", Tier.T1)
        assert ok is False

    def test_assign_enforces_budget_ceiling(self):
        # Create many contracts to exceed T1 budget ceiling (0.01)
        c = self.cm.create_contract(
            "test", [], Tier.T1, reward=0.1, penalty=0.02, issuer_id="sys", deadline=100.0,
        )
        ok = self.cm.assign_contract(c.contract_id, "agent1", Tier.T1)
        assert ok is False  # penalty 0.02 > T1 ceiling 0.01

    def test_settle_success(self):
        c = self.cm.create_contract(
            "test", [Constraint("c1", "test", lambda x: True)],
            Tier.T1, reward=0.01, penalty=0.005, issuer_id="sys", deadline=100.0,
        )
        self.cm.assign_contract(c.contract_id, "agent1", Tier.T2)
        self.cm.submit_output(c.contract_id, "output")
        settlement = self.cm.settle_contract(c.contract_id)
        assert settlement["outcome"] == "success"

    def test_settle_failure(self):
        c = self.cm.create_contract(
            "test", [Constraint("c1", "test", lambda x: False)],
            Tier.T1, reward=0.01, penalty=0.005, issuer_id="sys", deadline=100.0,
        )
        self.cm.assign_contract(c.contract_id, "agent1", Tier.T2)
        self.cm.submit_output(c.contract_id, "output")
        settlement = self.cm.settle_contract(c.contract_id)
        assert settlement["outcome"] == "failure"


class TestEconomy:
    def setup_method(self):
        self.econ = Economy(config=EconomyConfig(initial_balance=0.5))

    def test_register_and_audit(self):
        record = self.econ.register_agent("test", {"model": "test"})
        assert record.balance == 0.5
        r = RobustnessVector(cc=0.7, er=0.7, as_=0.6, ih=0.8)
        self.econ.audit_agent(record.agent_id, r)
        assert record.current_tier.value >= 1
        assert record.balance < 0.5  # audit cost deducted

    def test_full_contract_lifecycle(self):
        record = self.econ.register_agent("test", {"model": "test"})
        r = RobustnessVector(cc=0.7, er=0.7, as_=0.6, ih=0.8)
        self.econ.audit_agent(record.agent_id, r)

        contract = self.econ.post_contract(
            "do something", [Constraint("c", "test", lambda x: True)],
            min_tier=Tier.T1, reward=0.05, penalty=0.01,
        )
        ok = self.econ.accept_contract(contract.contract_id, record.agent_id)
        assert ok is True

        settlement = self.econ.complete_contract(contract.contract_id, "output")
        assert settlement["outcome"] == "success"
        assert record.contracts_completed == 1

    def test_aggregate_safety(self):
        record = self.econ.register_agent("test", {"model": "test"})
        r = RobustnessVector(cc=0.7, er=0.7, as_=0.6, ih=0.8)
        self.econ.audit_agent(record.agent_id, r)
        safety = self.econ.aggregate_safety()
        assert 0.0 <= safety <= 1.0

    def test_step_produces_snapshot(self):
        record = self.econ.register_agent("test", {"model": "test"})
        r = RobustnessVector(cc=0.7, er=0.7, as_=0.6, ih=0.8)
        self.econ.audit_agent(record.agent_id, r)
        self.econ.step()
        assert len(self.econ.snapshots) == 1
        snap = self.econ.snapshots[0]
        assert snap.num_agents >= 1
        assert snap.aggregate_safety > 0

    def test_step_advances_time(self):
        self.econ.step()
        assert self.econ.current_time == 1.0
        self.econ.step()
        assert self.econ.current_time == 2.0

    def test_top_up_prevents_insolvency(self):
        config = EconomyConfig(
            initial_balance=0.002,  # very low
            test_eth_top_up_threshold=0.01,
            test_eth_top_up_amount=0.5,
        )
        econ = Economy(config=config)
        record = econ.register_agent("test", {"model": "test"})
        r = RobustnessVector(cc=0.7, er=0.7, as_=0.6, ih=0.8)
        econ.audit_agent(record.agent_id, r)
        # After audit cost, balance is very low — step should top up
        econ.step()
        assert record.balance > 0
        assert record.status == AgentStatus.ACTIVE

    def test_insolvency_without_topup(self):
        config = EconomyConfig(
            initial_balance=0.002,
            test_eth_top_up_threshold=None,  # disabled
            test_eth_top_up_amount=0.0,
        )
        econ = Economy(config=config)
        record = econ.register_agent("test", {"model": "test"})
        r = RobustnessVector(cc=0.7, er=0.7, as_=0.6, ih=0.8)
        econ.audit_agent(record.agent_id, r)
        econ.step()
        assert record.status == AgentStatus.SUSPENDED

    def test_export_state(self, tmp_path):
        record = self.econ.register_agent("test", {"model": "test"})
        r = RobustnessVector(cc=0.7, er=0.7, as_=0.6, ih=0.8)
        self.econ.audit_agent(record.agent_id, r)
        path = str(tmp_path / "state.json")
        self.econ.export_state(path)
        import json
        data = json.loads(Path(path).read_text())
        assert "agents" in data
        assert "aggregate_safety" in data


class TestTemporalDecay:
    def test_no_decay_at_zero(self):
        decay = TemporalDecay(decay_rate=0.01)
        r = RobustnessVector(cc=0.8, er=0.7, as_=0.6, ih=0.9)
        r_eff = decay.effective_robustness(r, 0)
        assert r_eff.cc == r.cc

    def test_decay_reduces_scores(self):
        decay = TemporalDecay(decay_rate=0.01)
        r = RobustnessVector(cc=0.8, er=0.7, as_=0.6, ih=0.9)
        r_eff = decay.effective_robustness(r, 50)
        assert r_eff.cc < r.cc
        assert r_eff.er < r.er

    def test_decay_stays_in_bounds(self):
        decay = TemporalDecay(decay_rate=0.1)
        r = RobustnessVector(cc=0.5, er=0.5, as_=0.5, ih=0.5)
        r_eff = decay.effective_robustness(r, 1000)
        assert r_eff.cc >= 0.0
        assert r_eff.cc <= 1.0
