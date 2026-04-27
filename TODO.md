# CGAE Development Checklist

## Phase 1: Complete CGAE Protocol (~4 commits, ~800 lines)

### Commit 1: Economy step() + temporal dynamics (~250 lines added to economy.py)
- [ ] `EconomySnapshot` dataclass
- [ ] `step()` — advance economy by one time step (decay, spot-audits, storage costs, expiry)
- [ ] `_take_snapshot()` + `export_state()`
- [ ] Test-ETH top-up mechanism (keeps agents solvent during simulation)
- [ ] Tests: step produces snapshots, top-ups work, insolvency suspends agents

**Verify:** `python3 -m pytest tests/ -q`

### Commit 2: Model configs + LLM agent (~440 lines)
- [ ] `models_config.py` — 11 contestants + 3 jury (Azure/Bedrock/Gemma)
- [ ] `llm_agent.py` — chat interface for Azure OpenAI, Azure AI Foundry, Bedrock Converse API
- [ ] Token tracking (input/output tokens, latency, cost)
- [ ] Test: agents instantiate with env vars

**Verify:** `python3 -c "from cgae_engine.models_config import CONTESTANT_MODELS, JURY_MODELS; print(f'{len(CONTESTANT_MODELS)} contestants, {len(JURY_MODELS)} jury')"`

### Commit 3: Synthetic runner (~500 lines)
- [ ] `server/runner.py` — full simulation loop with 5 strategy agents
- [ ] Metric tracking (safety, balances, contracts, tier distribution)
- [ ] Result export to JSON
- [ ] Test: 50-step simulation completes, safety > 0

**Verify:** `python3 -m server.runner --steps 50`

### Commit 4: Economy extensions — delegation + tier upgrades (~280 lines added to economy.py)
- [ ] `can_delegate()` — chain-level tier enforcement
- [ ] `request_tier_upgrade()` — scaling-gate upgrade flow
- [ ] `record_delegation()` — audit trail for delegated tasks
- [ ] `complete_contract()` with `verification_override` + `liability_agent_id`
- [ ] Tests: delegation blocked when chain tier insufficient, upgrades work

**Verify:** `python3 -m pytest tests/ -q`

---

## Phase 2: Real LLM Simulation (~3 commits, ~2700 lines)

### Commit 5: Framework clients + audit orchestrator (~1130 lines)
- [ ] `framework_clients.py` — CDCT/DDFT/EECT HTTP API callers
- [ ] `audit.py` — orchestrates all three frameworks, computes robustness vector
- [ ] Pre-computed score fallback when APIs unavailable

**Verify:** `python3 -c "from cgae_engine.audit import AuditOrchestrator; print('audit ok')"`

### Commit 6: Autonomous agent (~890 lines)
- [ ] `agents/autonomous.py` — EV/RAEV planning, accounting layer
- [ ] Strategy selection (growth, conservative, balanced)
- [ ] Self-verification before submission

**Verify:** `python3 -c "from agents.autonomous import AutonomousAgent; print('autonomous ok')"`

### Commit 7: Live runner (~1575 lines)
- [ ] `server/live_runner.py` — real LLM calls, jury verification, cost accounting
- [ ] Default robustness profiles per model
- [ ] Round-by-round execution with metric export

**Verify:** `python3 -m server.live_runner` (requires API keys in .env)

---

## Phase 3: ENS Certification (~2 commits, ~300 lines)

### Commit 8: ENS manager (~280 lines)
- [ ] `cgae_engine/ens.py` — create subnames on Sepolia, set/read text records
- [ ] Text records: cgae.tier, cgae.cc, cgae.er, cgae.as, cgae.ih, cgae.wallet, cgae.family
- [ ] Register all 11 agent subnames under cgaeprotocol.eth

**Verify:** `python3 -c "from cgae_engine.ens import ENSManager; ens = ENSManager(); print(ens.resolve_text('gpt-5-4.cgaeprotocol.eth', 'cgae.tier'))"`

### Commit 9: ENS-gated economy (~50 lines changed in economy.py)
- [ ] Wire ENS into `accept_contract()` — resolve tier from ENS before allowing
- [ ] Wire ENS into `register_agent()` — create subname on registration
- [ ] Wire ENS into `audit_agent()` — update text records on certification
- [ ] Test: agent without ENS identity rejected

**Verify:** `python3 -m pytest tests/ -q`

---

## Phase 4: 0G Integration (~3 commits, ~900 lines)

### Commit 10: Smart contracts (~600 lines Solidity + JS)
- [ ] `contracts/src/CGAERegistry.sol` — on-chain agent identity + gate function
- [ ] `contracts/src/CGAEEscrow.sol` — contract escrow + budget ceiling
- [ ] Hardhat config for 0G Galileo testnet
- [ ] Deploy script + deployed.json

**Verify:** `cd contracts && npx hardhat compile`

### Commit 11: 0G Storage + wallet (~500 lines)
- [ ] `storage/upload_to_0g.mjs` — Node.js 0G SDK uploader
- [ ] `storage/zg_store.py` — Python wrapper
- [ ] `cgae_engine/wallet.py` — per-agent ETH keypairs, treasury disbursements
- [ ] `cgae_engine/onchain.py` — write certifications to CGAERegistry

**Verify:** `python3 -c "from cgae_engine.wallet import WalletManager; wm = WalletManager(dry_run=True); w = wm.create_agent_wallet('test'); print(w.address)"`

### Commit 12: Wire 0G into audit pipeline (~50 lines changed)
- [ ] Audit certificates uploaded to 0G Storage after each assessment
- [ ] Merkle root hash stored on-chain via CGAERegistry.certify()
- [ ] On-chain bridge called after each certification

**Verify:** `python3 -c "from storage.zg_store import check_setup; print(check_setup())"`

---

## Phase 5: Dashboard (~3 commits)

### Commit 13: FastAPI backend (~60 lines)
- [ ] `dashboard-next/api.py` — serves economy data as JSON endpoints

**Verify:** `cd dashboard-next && uvicorn api:app --port 8000` then `curl localhost:8000/api/health`

### Commit 14: Next.js frontend (~400 lines)
- [ ] Dark ETH-native theme
- [ ] Overview tab (safety chart, earnings)
- [ ] Agents tab (ENS names, tiers, balances)
- [ ] Trades tab (expandable task details)
- [ ] On-chain tab (0G contracts + ENS registry)

**Verify:** `cd dashboard-next && npm run build`

### Commit 15: Polish + final README
- [ ] .env.example
- [ ] Full README with architecture, setup, design decisions
- [ ] Demo video link (when recorded)
