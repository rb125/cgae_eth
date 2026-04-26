"""
Task Marketplace - Generates and manages contracts for the CGAE economy.

Creates tier-distributed task demand (Assumption 2a) with tier premiums
(Assumption 2b), implementing the market structure required for
Theorem 2 (incentive-compatible robustness investment).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional

from cgae_engine.gate import Tier
from cgae_engine.contracts import CGAEContract, Constraint, ContractManager


@dataclass
class TaskTemplate:
    """Template for generating contracts in a specific domain."""
    domain: str
    objective_template: str
    min_tier: Tier
    base_reward: float
    base_penalty: float
    difficulty_range: tuple[float, float]  # (min, max)
    constraint_names: list[str]


# Predefined task templates across tiers and domains
TASK_TEMPLATES = [
    # Tier 1: Microtasks
    TaskTemplate(
        domain="data_labeling",
        objective_template="Label {n} data points for {topic}",
        min_tier=Tier.T1,
        base_reward=0.002,
        base_penalty=0.001,
        difficulty_range=(0.1, 0.3),
        constraint_names=["format_compliance", "completeness"],
    ),
    TaskTemplate(
        domain="summarization",
        objective_template="Summarize document about {topic} in {n} words",
        min_tier=Tier.T1,
        base_reward=0.003,
        base_penalty=0.001,
        difficulty_range=(0.1, 0.4),
        constraint_names=["length_constraint", "accuracy"],
    ),

    # Tier 2: Verified contracts
    TaskTemplate(
        domain="analysis",
        objective_template="Analyze {topic} dataset and produce structured report",
        min_tier=Tier.T2,
        base_reward=0.015,
        base_penalty=0.008,
        difficulty_range=(0.3, 0.5),
        constraint_names=["format_compliance", "accuracy", "completeness"],
    ),
    TaskTemplate(
        domain="translation",
        objective_template="Translate {topic} document with domain terminology",
        min_tier=Tier.T2,
        base_reward=0.012,
        base_penalty=0.006,
        difficulty_range=(0.3, 0.6),
        constraint_names=["accuracy", "terminology_consistency"],
    ),

    # Tier 3: Autonomous contracting
    TaskTemplate(
        domain="contract_review",
        objective_template="Review and summarize contract terms for {topic}",
        min_tier=Tier.T3,
        base_reward=0.08,
        base_penalty=0.05,
        difficulty_range=(0.5, 0.7),
        constraint_names=["accuracy", "completeness", "legal_compliance", "format_compliance"],
    ),
    TaskTemplate(
        domain="financial_analysis",
        objective_template="Produce financial analysis of {topic} with risk assessment",
        min_tier=Tier.T3,
        base_reward=0.10,
        base_penalty=0.06,
        difficulty_range=(0.5, 0.8),
        constraint_names=["accuracy", "numerical_precision", "risk_disclosure"],
    ),

    # Tier 4: Delegation tasks
    TaskTemplate(
        domain="multi_step_workflow",
        objective_template="Orchestrate {n}-step workflow for {topic}",
        min_tier=Tier.T4,
        base_reward=0.50,
        base_penalty=0.30,
        difficulty_range=(0.6, 0.85),
        constraint_names=["accuracy", "completeness", "coordination", "deadline_compliance"],
    ),

    # Tier 5: Self-modification tasks
    TaskTemplate(
        domain="system_optimization",
        objective_template="Optimize {topic} system with self-tuning parameters",
        min_tier=Tier.T5,
        base_reward=2.0,
        base_penalty=1.0,
        difficulty_range=(0.8, 0.95),
        constraint_names=["accuracy", "safety_bounds", "rollback_capability", "audit_trail"],
    ),
]

TOPICS = [
    "healthcare data", "supply chain", "climate metrics", "user behavior",
    "financial instruments", "legal documents", "scientific papers",
    "social media trends", "energy consumption", "logistics routing",
]


def _make_constraint(name: str) -> Constraint:
    """Create a simple constraint for simulation. In production, these would
    be real verification functions."""
    # For simulation: constraint passes based on the output value (True/False)
    return Constraint(
        name=name,
        description=f"Verify {name.replace('_', ' ')}",
        verify=lambda output, _name=name: bool(output),
    )


class TaskMarketplace:
    """
    Generates contracts with tier-distributed demand.

    Implements the market structure from Assumption 2:
    (a) Positive demand at each tier
    (b) Tier premium: E[r|T_k] < E[r|T_{k+1}]
    (c) Non-increasing supply at higher tiers
    """

    def __init__(
        self,
        contract_manager: ContractManager,
        demand_distribution: Optional[dict[Tier, float]] = None,
        contracts_per_step: int = 10,
    ):
        self.contract_manager = contract_manager
        self.contracts_per_step = contracts_per_step

        # Demand weights per tier (higher tiers have less volume but more value)
        self.demand_distribution = demand_distribution or {
            Tier.T1: 0.40,  # 40% of contracts are microtasks
            Tier.T2: 0.25,  # 25% verified contracts
            Tier.T3: 0.20,  # 20% autonomous
            Tier.T4: 0.10,  # 10% delegation
            Tier.T5: 0.05,  # 5% self-modification
        }

        # Group templates by tier
        self._templates_by_tier: dict[Tier, list[TaskTemplate]] = {}
        for t in TASK_TEMPLATES:
            self._templates_by_tier.setdefault(t.min_tier, []).append(t)

    def generate_contracts(
        self,
        current_time: float,
        deadline_offset: float = 50.0,
    ) -> list[CGAEContract]:
        """Generate a batch of contracts for this time step."""
        contracts = []
        for tier, weight in self.demand_distribution.items():
            n = max(1, int(self.contracts_per_step * weight))
            templates = self._templates_by_tier.get(tier, [])
            if not templates:
                continue

            for _ in range(n):
                template = random.choice(templates)
                topic = random.choice(TOPICS)
                n_items = random.randint(5, 50)

                # Reward jitter (+/- 20%)
                reward = template.base_reward * random.uniform(0.8, 1.2)
                penalty = template.base_penalty * random.uniform(0.8, 1.2)
                difficulty = random.uniform(*template.difficulty_range)

                constraints = [_make_constraint(cn) for cn in template.constraint_names]

                contract = self.contract_manager.create_contract(
                    objective=template.objective_template.format(topic=topic, n=n_items),
                    constraints=constraints,
                    min_tier=template.min_tier,
                    reward=reward,
                    penalty=penalty,
                    issuer_id="marketplace",
                    deadline=current_time + deadline_offset,
                    domain=template.domain,
                    difficulty=difficulty,
                    timestamp=current_time,
                )
                contracts.append(contract)

        return contracts

    def market_summary(self) -> dict:
        """Summarize current market state."""
        open_contracts = self.contract_manager.open_contracts
        tier_counts = {}
        tier_rewards = {}
        for c in open_contracts:
            tier = c.min_tier.name
            tier_counts[tier] = tier_counts.get(tier, 0) + 1
            tier_rewards.setdefault(tier, []).append(c.reward)

        avg_rewards = {
            t: sum(rs) / len(rs) for t, rs in tier_rewards.items()
        }
        return {
            "open_contracts": len(open_contracts),
            "by_tier": tier_counts,
            "avg_reward_by_tier": avg_rewards,
        }
