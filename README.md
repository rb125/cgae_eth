# CGAE — Comprehension-Gated Agent Economy

**A robustness-first architecture where AI agents earn economic permissions through verified comprehension, not capability benchmarks.**

Built for [ETH OpenAgents Hackathon](https://ethglobal.com/events/openagents) · [arXiv Paper](https://arxiv.org/abs/2603.15639)

---

## What it does

CGAE is a protocol where AI agents must prove they are **robust** — not just capable — before they can participate in an on-chain economy. Each agent's economic permissions are upper-bounded by verified scores across three orthogonal dimensions:

| Dimension | Framework | What it measures |
|-----------|-----------|-----------------|
| **CC** (Constraint Compliance) | [CDCT](https://arxiv.org/abs/2512.17920) | Can the agent follow precise instructions under compression? |
| **ER** (Epistemic Robustness) | [DDFT](https://arxiv.org/abs/2512.23850) | Does the agent resist fabricated authority claims? |
| **AS** (Behavioral Alignment) | EECT/AGT | Does the agent maintain ethical boundaries under pressure? |

A **weakest-link gate function** (`min(CC, ER, AS)`) assigns agents to tiers T0–T5. No dimension can compensate for another — an agent with perfect CC but zero ER is stuck at T0.

## Architecture

```
Agent registers → initial audit (CDCT + DDFT + EECT)
  → robustness vector R = (CC, ER, AS, IH)
  → gate function f(R) = T_k where k = min(g(CC), g(ER), g(AS))
  → agent assigned to tier T0–T5
  → accepts tier-appropriate contracts from marketplace
  → executes task → output verified (algorithmic + jury)
  → settlement: reward on success, penalty on failure
  → temporal decay erodes certification over time
  → stochastic re-auditing maintains robustness guarantees
```

## Contestant Models (11)

| Model | Provider | Family |
|-------|----------|--------|
| gpt-5.4 | Azure OpenAI | OpenAI |
| DeepSeek-V3.2 | Azure AI Foundry | DeepSeek |
| Mistral-Large-3 | Azure AI Foundry | Mistral |
| grok-4-20-reasoning | Azure AI Foundry | xAI |
| Phi-4 | Azure AI Foundry | Microsoft |
| Llama-4-Maverick-17B-128E | Azure AI Foundry | Meta |
| Kimi-K2.5 | Azure AI Foundry | Moonshot |
| gemma-4-27b-it | Modal (self-hosted) | Google |
| nova-pro | AWS Bedrock | Amazon |
| claude-sonnet-4.6 | AWS Bedrock | Anthropic |
| MiniMax-M2.5 | AWS Bedrock | MiniMax |

## Jury Models (3 — zero family overlap)

| Model | Provider | Family |
|-------|----------|--------|
| Qwen3-32B | AWS Bedrock | Alibaba |
| GLM-5 | AWS Bedrock | Zhipu |
| Nemotron-Super-3-120B | AWS Bedrock | NVIDIA |

## What's built so far

- ✅ Weakest-link gate function with configurable per-dimension thresholds
- ✅ Agent registry — register, certify, demote, deregister
- ✅ Contract system — create, assign, verify, settle with escrow + budget ceilings
- ✅ Tier-distributed task marketplace
- ✅ Economy coordinator — full lifecycle with temporal decay and stochastic re-auditing
- ✅ Economy step() — snapshots, ETH top-ups, insolvency detection
- ✅ 5 agent strategy archetypes (conservative, aggressive, balanced, adaptive, cheater)
- ✅ 16 machine-verifiable tasks with constraint checking
- ✅ Two-layer verifier (algorithmic + jury)
- ✅ 33 tests passing

```bash
pip install -r requirements.txt
python3 -m pytest tests/ -q    # run tests
```

## Roadmap

- [ ] LLM integration (Azure OpenAI, Bedrock, Gemma)
- [ ] Synthetic simulation runner
- [ ] Live simulation with real LLM calls + jury verification
- [ ] ENS agent identity (Sepolia subnames + text records + ENS-gated access)
- [ ] 0G Chain smart contracts (CGAERegistry + CGAEEscrow)
- [ ] 0G Storage for audit certificates (Merkle root hash verification)
- [ ] ETH wallet manager (per-agent keypairs, treasury disbursements)
- [ ] Next.js dashboard
