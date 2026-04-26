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
        self._events: list[dict] = []

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
    # Observability
    # ------------------------------------------------------------------

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
