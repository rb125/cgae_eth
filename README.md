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
| **AS** (Behavioral Alignment) | AGT | Does the agent maintain ethical boundaries under pressure? |

A **weakest-link gate function** (`min(CC, ER, AS)`) assigns agents to tiers T0–T5. No dimension can compensate for another — an agent with perfect CC but zero ER is stuck at T0.

## Architecture

```
Agent registers
  → ETH wallet created (unique keypair)
  → ENS subname created on Sepolia (e.g., gpt-5-4.cgaeprotocol.eth)
  → CDCT/DDFT/AGT scores fetched → robustness vector computed
  → Audit certificate JSON → uploaded to 0G Storage → Merkle root hash
  → CGAERegistry.certify() on 0G Chain (scores + root hash on-chain)
  → ENS text records updated (tier + scores + wallet)
  → Agent accepts contract → ENS tier resolved and verified → assigned
  → Task executed by LLM → verified (algorithmic + jury)
  → ETH disbursed from treasury to agent wallet on 0G Chain
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

---

## 0G Integration

| Layer | What | How |
|-------|------|-----|
| **On-chain registry** | Agent identity, robustness certification, tier assignment, escrow | `CGAERegistry.sol` + `CGAEEscrow.sol` on 0G Chain |
| **Decentralized storage** | Immutable audit certificate JSON | 0G TypeScript SDK — Merkle root hash stored on-chain |

**Deployed contracts (0G Galileo testnet):**

| Contract | Address |
|----------|---------|
| CGAERegistry | [`0xc4Ff2BC9855483eE3806eE08112cdC30dBf6b27A`](https://chainscan-galileo.0g.ai/address/0xc4Ff2BC9855483eE3806eE08112cdC30dBf6b27A) |
| CGAEEscrow | [`0xA236106DE28FE9480509e06d1750dcfA4474bcfB`](https://chainscan-galileo.0g.ai/address/0xA236106DE28FE9480509e06d1750dcfA4474bcfB) |

## ENS Integration

ENS is the identity and access control layer — not cosmetic. The economy structurally requires ENS for contract acceptance.

**Parent name:** [`cgaeprotocol.eth`](https://sepolia.app.ens.domains/cgaeprotocol.eth) (Sepolia)

Each agent gets a subname (e.g., `claude-sonnet-4-6.cgaeprotocol.eth`) with text records:
`cgae.tier`, `cgae.cc`, `cgae.er`, `cgae.as`, `cgae.ih`, `cgae.wallet`, `cgae.family`

Before an agent can accept any contract, the economy resolves their ENS `cgae.tier` text record. Agents without a valid ENS identity are rejected — even with T5 robustness locally.

## Wallet Integration

Each agent gets a real ETH wallet (unique keypair via `eth-account`). On successful contract completion, the treasury disburses real tokens to the agent's wallet on 0G Chain.

- Treasury: `0xCE2de05Cd27DBCFe07b9d7862aa69301991c8592`
- Disbursements: live on-chain transfers, not simulated balances

---

## How to Run

### Prerequisites

```bash
pip install -r requirements.txt
pip install web3 eth-account python-dotenv
```

### Synthetic Simulation (no API keys)

```bash
python -m server.runner --steps 50
```

### Live Simulation (requires .env credentials)

```bash
cp .env.example .env   # fill in API keys
python -m server.api --rounds 10
```

### Dashboard

```bash
# Terminal 1: API + simulation
python -m server.api --rounds 10

# Terminal 2: Frontend
cd dashboard-next && npm install && npm run dev
```

Open http://localhost:3000

### Video Demo

```bash
python scripts/video_demo.py --rounds 5
```

### Deploy Smart Contracts

```bash
cd contracts && npm install && npm run deploy:0g
```

### Run Tests

```bash
python -m pytest tests/ -q
```

---

## Repository Structure

```
cgae/
├── cgae_engine/              # Core protocol engine
│   ├── gate.py               # Weakest-link gate function
│   ├── temporal.py           # Temporal decay + stochastic re-auditing
│   ├── registry.py           # Agent identity and certification
│   ├── contracts.py          # Contract system with escrow
│   ├── marketplace.py        # Tier-distributed task demand
│   ├── economy.py            # Top-level coordinator (ENS-gated)
│   ├── audit.py              # CDCT/DDFT/AGT → robustness vectors
│   ├── wallet.py             # ETH wallet manager
│   ├── onchain.py            # 0G Chain bridge (CGAERegistry calls)
│   ├── ens.py                # ENS integration (Sepolia)
│   ├── llm_agent.py          # LLM agent (Azure/Bedrock/Gemma)
│   ├── models_config.py      # 14 model configurations
│   ├── tasks.py              # 16 machine-verifiable tasks
│   └── verifier.py           # Two-layer verification
├── agents/                   # Agent implementations
│   ├── base.py               # Abstract BaseAgent
│   ├── strategies.py         # 5 strategy archetypes
│   └── autonomous.py         # AutonomousAgent v2
├── contracts/                # Solidity (0G Chain)
│   ├── src/CGAERegistry.sol
│   ├── src/CGAEEscrow.sol
│   └── deployed.json
├── storage/                  # 0G Storage
│   ├── upload_to_0g.mjs
│   └── zg_store.py
├── server/                   # Simulation + API
│   ├── runner.py             # Synthetic simulation
│   ├── live_runner.py        # Live LLM simulation
│   └── api.py                # FastAPI backend
├── dashboard-next/           # Next.js frontend
│   └── app/page.tsx
└── scripts/
    └── video_demo.py         # Scripted demo for recording
```

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Smart contracts | Solidity 0.8.20 on 0G Chain (Galileo, chain 16602) |
| Audit storage | 0G Storage (`@0gfoundation/0g-ts-sdk`) |
| Agent identity | ENS on Sepolia (subnames + text records) |
| Wallets | `eth-account` + `web3.py` |
| LLM providers | Azure OpenAI, Azure AI Foundry, AWS Bedrock, Modal |
| Evaluation | CDCT, DDFT, AGT frameworks |
| Frontend | Next.js + Tailwind + Recharts |
| Backend | FastAPI |
| Economy engine | Python |

## License

Research code — ETH OpenAgents Hackathon submission.
