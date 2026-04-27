"""
CGAE Economy — Top-level coordinator.

Ties together registry, gate, contracts, temporal dynamics into
a single coherent economic system.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from cgae_engine.gate import GateFunction, RobustnessVector, Tier, TierThresholds
from cgae_engine.temporal import TemporalDecay, StochasticAuditor
from cgae_engine.registry import AgentRegistry, AgentRecord, AgentStatus
from cgae_engine.contracts import ContractManager, CGAEContract, ContractStatus, Constraint

logger = logging.getLogger(__name__)


@dataclass
class EconomyConfig:
    """Configuration for the CGAE economy."""
    thresholds: TierThresholds = field(default_factory=TierThresholds)
    decay_rate: float = 0.01
    ih_threshold: float = 0.45
    initial_balance: float = 0.1
    audit_cost: float = 0.005
    storage_cost_per_step: float = 0.001
    test_eth_top_up_threshold: Optional[float] = 0.05
    test_eth_top_up_amount: float = 0.5


@dataclass
class EconomySnapshot:
    """A point-in-time snapshot of the economy for the dashboard."""
    timestamp: float
    num_agents: int
    tier_distribution: dict[str, int]
    total_contracts: int
    completed_contracts: int
    failed_contracts: int
    total_rewards_paid: float
    total_penalties_collected: float
    aggregate_safety: float
    total_balance: float
    total_test_eth_topups: float
    agent_summaries: list[dict]


class Economy:
    """
    The CGAE Economy runtime.

    Orchestrates:
    1. Agent registration and initial audit
    2. Contract creation and marketplace
    3. Contract assignment (tier-gated)
    4. Task execution and verification
    5. Settlement (reward/penalty)
    6. Temporal decay and stochastic re-auditing
    """

    def __init__(self, config: Optional[EconomyConfig] = None):
        self.config = config or EconomyConfig()
        self.gate = GateFunction(
            thresholds=self.config.thresholds,
            ih_threshold=self.config.ih_threshold,
        )
        self.registry = AgentRegistry(gate=self.gate)
        self.contracts = ContractManager(budget_ceilings=self.gate.budget_ceilings)
        self.decay = TemporalDecay(decay_rate=self.config.decay_rate)
        self.auditor = StochasticAuditor()

        self.current_time: float = 0.0
        self._snapshots: list[EconomySnapshot] = []
        self._events: list[dict] = []
        self.total_test_eth_topups: float = 0.0

    def _effective_robustness(self, record: AgentRecord) -> Optional[RobustnessVector]:
        """Return temporally-decayed robustness for an agent."""
        cert = record.current_certification
        if cert is None or record.current_robustness is None:
            return None
        dt = self.current_time - cert.timestamp
        return self.decay.effective_robustness(record.current_robustness, dt)

    def _should_top_up_agents(self) -> bool:
        return (
            self.config.test_eth_top_up_threshold is not None
            and self.config.test_eth_top_up_amount > 0.0
        )

    def _maybe_top_up_agent(self, agent: AgentRecord) -> Optional[dict]:
        """Top up an agent's balance if it drops below threshold."""
        if not self._should_top_up_agents():
            return None
        threshold = self.config.test_eth_top_up_threshold
        if threshold is None or agent.balance >= threshold:
            return None
        top_up_amount = max(self.config.test_eth_top_up_amount, threshold - agent.balance)
        agent.balance += top_up_amount
        agent.total_topups += top_up_amount
        self.total_test_eth_topups += top_up_amount
        return {"agent_id": agent.agent_id, "amount": top_up_amount, "balance": agent.balance}

    # ------------------------------------------------------------------
    # Agent lifecycle
    # ------------------------------------------------------------------

    def register_agent(
        self,
        model_name: str,
        model_config: dict,
        provenance: Optional[dict] = None,
    ) -> AgentRecord:
        """Register a new agent with seed capital."""
        record = self.registry.register(
            model_name=model_name,
            model_config=model_config,
            provenance=provenance,
            initial_balance=self.config.initial_balance,
            timestamp=self.current_time,
        )
        self._log("agent_registered", {"agent_id": record.agent_id, "model": model_name})
        return record

    def audit_agent(
        self,
        agent_id: str,
        robustness: RobustnessVector,
        audit_type: str = "registration",
        audit_details: Optional[dict] = None,
    ) -> dict:
        """Audit an agent and update their certification."""
        record = self.registry.get_agent(agent_id)
        if record is None:
            raise KeyError(f"Agent {agent_id} not found")

        total_audit_cost = self.config.audit_cost * 4
        record.balance -= total_audit_cost
        record.total_spent += total_audit_cost

        cert = self.registry.certify(
            agent_id=agent_id,
            robustness=robustness,
            audit_type=audit_type,
            timestamp=self.current_time,
            audit_details=audit_details,
        )

        detail = self.gate.evaluate_with_detail(robustness)
        self._log("agent_audited", {
            "agent_id": agent_id,
            "tier": cert.tier.name,
            "audit_type": audit_type,
            "cost": total_audit_cost,
            **detail,
        })
        return detail

    # ------------------------------------------------------------------
    # Contract lifecycle
    # ------------------------------------------------------------------

    def post_contract(
        self,
        objective: str,
        constraints: list[Constraint],
        min_tier: Tier,
        reward: float,
        penalty: float,
        deadline_offset: float = 100.0,
        domain: str = "general",
        difficulty: float = 0.5,
        issuer_id: str = "system",
    ) -> CGAEContract:
        """Post a new contract to the marketplace."""
        return self.contracts.create_contract(
            objective=objective,
            constraints=constraints,
            min_tier=min_tier,
            reward=reward,
            penalty=penalty,
            issuer_id=issuer_id,
            deadline=self.current_time + deadline_offset,
            domain=domain,
            difficulty=difficulty,
            timestamp=self.current_time,
        )

    def accept_contract(self, contract_id: str, agent_id: str) -> bool:
        """Agent accepts a contract. Enforces tier and budget ceiling."""
        record = self.registry.get_agent(agent_id)
        if record is None or record.status != AgentStatus.ACTIVE:
            return False
        if record.current_certification is None:
            return False

        dt = self.current_time - record.current_certification.timestamp
        r_eff = self.decay.effective_robustness(record.current_robustness, dt)
        effective_tier = self.gate.evaluate(r_eff)

        return self.contracts.assign_contract(
            contract_id=contract_id,
            agent_id=agent_id,
            agent_tier=effective_tier,
            timestamp=self.current_time,
        )

    def complete_contract(
        self,
        contract_id: str,
        output: Any,
    ) -> dict:
        """Submit output for a contract and settle it."""
        passed, failures = self.contracts.submit_output(
            contract_id=contract_id,
            output=output,
            timestamp=self.current_time,
        )

        settlement = self.contracts.settle_contract(
            contract_id=contract_id,
            timestamp=self.current_time,
        )

        agent_id = settlement["agent_id"]
        performer = self.registry.get_agent(agent_id)

        if settlement["outcome"] == "success":
            if performer:
                performer.balance += settlement["reward"]
                performer.total_earned += settlement["reward"]
                performer.contracts_completed += 1
        else:
            if performer:
                performer.balance -= settlement["penalty"]
                performer.total_penalties += settlement["penalty"]
                performer.contracts_failed += 1

        settlement["failures"] = failures
        self._log("contract_settled", settlement)
        return settlement

    # ------------------------------------------------------------------
    # Time step and temporal dynamics
    # ------------------------------------------------------------------

    def step(self, audit_callback=None) -> dict:
        """
        Advance the economy by one time step.
        Applies temporal decay, spot-audits, storage costs, top-ups, and expiry.
        """
        self.current_time += 1.0
        step_events = {
            "timestamp": self.current_time,
            "audits_triggered": [],
            "agents_demoted": [],
            "agents_expired": [],
            "contracts_expired": [],
            "storage_costs": 0.0,
            "test_eth_topups": [],
        }

        for agent in self.registry.active_agents:
            cert = agent.current_certification
            if cert is None:
                continue

            # Temporal decay: has effective tier dropped?
            dt = self.current_time - cert.timestamp
            r_eff = self.decay.effective_robustness(cert.robustness, dt)
            effective_tier = self.gate.evaluate(r_eff)

            if effective_tier < agent.current_tier:
                self.registry.certify(agent.agent_id, r_eff, audit_type="decay", timestamp=self.current_time)
                step_events["agents_expired"].append(agent.agent_id)

            # Stochastic spot-audit
            time_since_audit = self.current_time - agent.last_audit_time
            if self.auditor.should_audit(agent.current_tier, time_since_audit):
                step_events["audits_triggered"].append(agent.agent_id)
                new_r = audit_callback(agent.agent_id) if audit_callback else r_eff
                new_tier = self.gate.evaluate(new_r)
                if new_tier < agent.current_tier:
                    self.registry.demote(agent.agent_id, new_r, reason="spot_audit", timestamp=self.current_time)
                    step_events["agents_demoted"].append(agent.agent_id)
                else:
                    self.registry.certify(agent.agent_id, new_r, audit_type="spot", timestamp=self.current_time)
                agent.balance -= self.config.audit_cost * 4
                agent.total_spent += self.config.audit_cost * 4

            # Storage cost
            agent.balance -= self.config.storage_cost_per_step
            agent.total_spent += self.config.storage_cost_per_step
            step_events["storage_costs"] += self.config.storage_cost_per_step

            # Top-up if needed
            topup = self._maybe_top_up_agent(agent)
            if topup:
                step_events["test_eth_topups"].append(topup)

            # Insolvency check
            if agent.balance <= 0:
                agent.status = AgentStatus.SUSPENDED
                self._log("agent_insolvent", {"agent_id": agent.agent_id, "balance": agent.balance})

        # Reactivate suspended agents if top-up is enabled
        if self._should_top_up_agents():
            for agent in self.registry.agents.values():
                if agent.status != AgentStatus.SUSPENDED:
                    continue
                topup = self._maybe_top_up_agent(agent)
                if topup and agent.balance > 0:
                    agent.status = AgentStatus.ACTIVE
                    step_events["test_eth_topups"].append(topup)

        # Expire overdue contracts
        step_events["contracts_expired"] = self.contracts.expire_contracts(self.current_time)

        # Take snapshot
        self._snapshots.append(self._take_snapshot())
        self._log("step", step_events)
        return step_events

    # ------------------------------------------------------------------
    # Observability
    # ------------------------------------------------------------------

    def _take_snapshot(self) -> EconomySnapshot:
        tier_dist = self.registry.tier_distribution()
        econ = self.contracts.economics_summary()
        agents = self.registry.active_agents
        return EconomySnapshot(
            timestamp=self.current_time,
            num_agents=len(agents),
            tier_distribution={t.name: c for t, c in tier_dist.items()},
            total_contracts=econ["total_contracts"],
            completed_contracts=econ["status_distribution"].get("completed", 0),
            failed_contracts=econ["status_distribution"].get("failed", 0),
            total_rewards_paid=econ["total_rewards_paid"],
            total_penalties_collected=econ["total_penalties_collected"],
            aggregate_safety=self.aggregate_safety(),
            total_balance=sum(a.balance for a in agents),
            total_test_eth_topups=self.total_test_eth_topups,
            agent_summaries=[a.to_dict() for a in agents],
        )

    @property
    def snapshots(self) -> list[EconomySnapshot]:
        return list(self._snapshots)

    @property
    def events(self) -> list[dict]:
        return list(self._events)

    def export_state(self, path: str):
        """Export full economy state to JSON."""
        state = {
            "timestamp": self.current_time,
            "config": {
                "decay_rate": self.config.decay_rate,
                "ih_threshold": self.config.ih_threshold,
                "initial_balance": self.config.initial_balance,
            },
            "agents": {aid: a.to_dict() for aid, a in self.registry.agents.items()},
            "contracts": self.contracts.economics_summary(),
            "aggregate_safety": self.aggregate_safety(),
            "total_test_eth_topups": self.total_test_eth_topups,
        }
        Path(path).write_text(json.dumps(state, indent=2, default=str))

    def aggregate_safety(self) -> float:
        """Compute aggregate safety S(P) (Definition 9)."""
        total_exposure = 0.0
        weighted_risk = 0.0

        for agent in self.registry.active_agents:
            cert = agent.current_certification
            if cert is None:
                continue
            dt = self.current_time - cert.timestamp
            r_eff = self.decay.effective_robustness(cert.robustness, dt)
            exposure = self.contracts.agent_exposure(agent.agent_id)
            if exposure <= 0:
                tier = self.gate.evaluate(r_eff)
                exposure = self.gate.budget_ceiling(tier)

            r_bar = r_eff.weakest
            total_exposure += exposure
            weighted_risk += exposure * (1.0 - r_bar)

        if total_exposure == 0:
            return 1.0
        return 1.0 - (weighted_risk / total_exposure)

    def _log(self, event_type: str, data: dict):
        self._events.append({
            "type": event_type,
            "timestamp": self.current_time,
            "data": data,
        })
        logger.debug(f"[t={self.current_time:.1f}] {event_type}: {data}")
