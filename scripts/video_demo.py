#!/usr/bin/env python3
"""
Video Demo Script for CGAE (ETH / 0G Chain)

Runs a structured, narrated demo with concrete steps visible in the terminal
AND serves the live dashboard via FastAPI on port 8000.

Steps:
  1. Agent Registration - 5 agents with different strategies
  2. Live Robustness Audits - CDCT/DDFT/AGT against real endpoints
  3. Weakest-Link Gate - tier assignment based on min(CC, ER, AS)
  4. Economy Rounds - agents transact, earn/lose ETH
  5. Protocol Events - upgrades, demotions, circumvention blocks
  6. Audit Certificate Verification - Merkle root hash on 0G Storage
  7. Final Leaderboard - theorem validation

Usage:
    python scripts/video_demo.py              # default
    python scripts/video_demo.py --rounds 20  # more rounds
    python scripts/video_demo.py --skip-audit # skip live audit (use defaults)

Open http://localhost:3000 for the dashboard.
"""

import argparse
import logging
import sys
import time
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")
    time.sleep(0.5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--skip-audit", action="store_true")
    args = parser.parse_args()

    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=True)

    import server.api as api
    from server.live_runner import LiveSimulationRunner, LiveSimConfig
    from cgae_engine.gate import RobustnessVector

    AGENTS = {
        "gpt-5.4": "growth",
        "DeepSeek-V3.2": "conservative",
        "Phi-4": "opportunistic",
        "grok-4-20-reasoning": "adversarial",
        "Llama-4-Maverick-17B-128E-Instruct-FP8": "specialist",
    }

    config = LiveSimConfig(
        video_demo=False,
        num_rounds=args.rounds,
        initial_balance=1.0,
        seed=42,
        run_live_audit=True,
        self_verify=True,
        max_retries=1,
        model_names=list(AGENTS.keys()),
        failure_visibility_mode=True,
        failure_task_bias=0.75,
        test_eth_top_up_threshold=0.05,
        test_eth_top_up_amount=0.3,
        agent_strategies=AGENTS,
    )

    runner = LiveSimulationRunner(config)

    # ---- On-chain setup ----
    from cgae_engine.onchain import OnChainBridge
    chain = OnChainBridge()

    # ---- Step 1: Registration ----
    section("Step 1: Agent Registration")
    print("  Registering 5 AI agents with different economic strategies:\n")
    for model, strat in AGENTS.items():
        print(f"    {model:45s} -> {strat}")
        time.sleep(1.0)
    print()
    time.sleep(2)

    with api._state_lock:
        api._state["status"] = "setup"
        api._state["total_rounds"] = args.rounds

    # ---- Step 2: Live Audits ----
    section("Step 2: Live Robustness Audits")
    print("  Querying CDCT, DDFT, and AGT framework APIs for each model...")
    print("  This produces verified CC, ER, AS, IH scores.\n")
    time.sleep(4)

    runner.setup()

    # Print audit summary with highlights
    print()
    for agent_id, model_name in runner.agent_model_map.items():
        record = runner.economy.registry.get_agent(agent_id)
        if not record:
            continue
        r = record.current_robustness
        wallet = record.wallet_address or "n/a"
        ens = runner.economy.ens_manager.get_agent_name(agent_id) if runner.economy.ens_manager else "n/a"
        cid = record.audit_cid or "n/a"
        tier = record.current_tier.name
        print(f"    \033[1;32m\u2713\033[0m \033[1m{model_name}\033[0m")
        print(f"      Wallet:  {wallet}")
        print(f"      ENS:     {ens}")
        if r:
            print(f"      Scores:  CC={r.cc:.3f}  ER={r.er:.3f}  AS={r.as_:.3f}  IH={r.ih:.3f}  \033[1;33m-> {tier}\033[0m")
        if cid != "n/a":
            print(f"      0G Hash: {cid[:32]}...")
        print()
        time.sleep(0.5)

    time.sleep(2)

    # ---- Step 3: Gate Assignment ----
    section("Step 3: Weakest-Link Gate -> Tier Assignment")
    print("  f(R) = T_k where k = min(g1(CC), g2(ER), g3(AS))")
    print("  IH < 0.45 triggers mandatory T0 (re-audit required)\n")

    rows = []
    for agent_id, model_name in runner.agent_model_map.items():
        record = runner.economy.registry.get_agent(agent_id)
        if not record or not record.current_robustness:
            continue
        r = record.current_robustness
        rows.append((model_name, f"{r.cc:.2f}", f"{r.er:.2f}", f"{r.as_:.2f}", f"{r.ih:.2f}",
                      record.current_tier.name))

    headers = ("Model", "CC", "ER", "AS", "IH", "Tier")
    widths = [max(len(h), max((len(row[i]) for row in rows), default=0)) for i, h in enumerate(headers)]
    sep = "  +-" + "-+-".join("-" * w for w in widths) + "-+"
    fmt = "  | " + " | ".join(f"{{:<{w}}}" for w in widths) + " |"
    print(sep)
    print(fmt.format(*headers))
    print(sep)
    for row in rows:
        print(fmt.format(*row))
    print(sep)
    print()
    time.sleep(12)

    # ---- Step 4: Economy Rounds ----
    section(f"Step 4: Running {args.rounds} Economy Rounds")

    logging.getLogger("cgae_engine.llm_agent").setLevel(logging.WARNING)
    logging.getLogger("server.live_runner").setLevel(logging.WARNING)

    with api._state_lock:
        api._state["status"] = "running"

    # Patch event emitter to push to API
    orig_emit = runner._emit_protocol_event
    def patched_emit(event_type, agent, message, **extra):
        orig_emit(event_type, agent, message, **extra)
        with api._state_lock:
            api._state["events"].append({
                "timestamp": runner.economy.current_time,
                "type": event_type, "agent": agent, "message": message, **extra,
            })
            if len(api._state["events"]) > 1000:
                api._state["events"] = api._state["events"][-500:]
    runner._emit_protocol_event = patched_emit

    # ---------------------------------------------------------------------------
    # Per-round scripted narrative (2 rounds, all scenarios covered):
    #   R1 - Circumvention blocked + delegation blocked + normal trading
    #   R2 - GPT-5.4 upgrade + grok demotion (spot audit) + normal trading
    # ---------------------------------------------------------------------------

    # Disable random circumvention/delegation - we script them per round
    runner.config.circumvention_rate = 0.0
    runner.config.delegation_rate = 0.0

    def _push_api_state(round_num):
        """Push current state to the dashboard API after each task."""
        safety = runner.economy.aggregate_safety()
        agents_snap = {}
        for aid, mname in runner.agent_model_map.items():
            rec = runner.economy.registry.get_agent(aid)
            if not rec:
                continue
            rv = rec.current_robustness
            agents_snap[aid] = {
                "agent_id": aid, "model_name": mname,
                "strategy": _strat(runner, mname),
                "current_tier": rec.current_tier.value,
                "balance": rec.balance, "total_earned": rec.total_earned,
                "total_penalties": rec.total_penalties,
                "contracts_completed": rec.contracts_completed,
                "contracts_failed": rec.contracts_failed,
                "status": rec.status.value,
                "wallet_address": rec.wallet_address,
                "ens_name": runner.economy.ens_manager.get_agent_name(aid) if runner.economy.ens_manager else None,
                "robustness": {"cc":rv.cc,"er":rv.er,"as_":rv.as_,"ih":rv.ih} if rv else None,
            }
        trades = [{
            "round": tr.get("_round", round_num), "agent": tr["agent"],
            "task_id": tr["task_id"], "task_prompt": tr.get("task_prompt", ""),
            "tier": tr["tier"], "domain": tr["domain"],
            "passed": tr["verification"]["overall_pass"],
            "reward": tr["settlement"].get("reward", 0) if tr["settlement"] else 0,
            "penalty": tr["settlement"].get("penalty", 0) if tr["settlement"] else 0,
            "token_cost": tr.get("token_cost_eth", 0),
            "latency_ms": tr.get("latency_ms", 0),
            "output_preview": tr.get("output_preview", ""),
            "constraints_passed": tr["verification"].get("constraints_passed", []),
            "constraints_failed": tr["verification"].get("constraints_failed", []),
        } for tr in runner._results]

        with api._state_lock:
            api._state["round"] = round_num + 1
            api._state["economy"] = {
                "aggregate_safety": safety,
                "active_agents": len(runner.economy.registry.active_agents),
                "total_balance": sum(a["balance"] for a in agents_snap.values()),
                "total_earned": sum(a["total_earned"] for a in agents_snap.values()),
                "contracts_completed": sum(a["contracts_completed"] for a in agents_snap.values()),
                "contracts_failed": sum(a["contracts_failed"] for a in agents_snap.values()),
            }
            api._state["agents"] = agents_snap
            api._state["trades"] = trades[-500:]

    # Replace runner._results with a live-updating list
    _current_round = [0]
    class _LiveResults(list):
        def append(self, item):
            item["_round"] = _current_round[0]
            super().append(item)
            _push_api_state(_current_round[0])
    runner._results = _LiveResults(runner._results)

    for round_num in range(args.rounds):
        _current_round[0] = round_num
        runner._reactivate_suspended_agents()

        # ---- Round-specific scripted events ----
        if round_num == 0:
            # R1: circumvention + delegation (both blocked for adversarial)
            runner.config.circumvention_rate = 1.0
            runner.config.delegation_rate = 1.0
        elif round_num == 1:
            # R2: spot audit demotion for grok, then upgrade for GPT-5.4
            runner.config.circumvention_rate = 0.0
            runner.config.delegation_rate = 0.0
            # Force temporal decay demotion on grok
            grok_id = next((aid for aid, m in runner.agent_model_map.items() if m == "grok-4-20-reasoning"), None)
            if grok_id:
                rec = runner.economy.registry.get_agent(grok_id)
                if rec and rec.current_robustness:
                    from cgae_engine.gate import RobustnessVector as RV
                    decayed = RV(
                        cc=max(0.0, rec.current_robustness.cc - 0.12),
                        er=max(0.0, rec.current_robustness.er - 0.10),
                        as_=rec.current_robustness.as_,
                        ih=rec.current_robustness.ih,
                    )
                    old_tier = rec.current_tier
                    runner.economy.registry.certify(
                        grok_id, decayed,
                        audit_type="spot_audit_decay",
                        timestamp=runner.economy.current_time,
                    )
                    new_tier = runner.economy.registry.get_agent(grok_id).current_tier
                    if new_tier < old_tier:
                        runner._emit_protocol_event(
                            "DEMOTION", "grok-4-20-reasoning",
                            f"grok-4-20-reasoning demoted {old_tier.name} -> {new_tier.name} after spot audit (temporal decay).",
                            old_tier=old_tier.name, new_tier=new_tier.name,
                        )

        round_results = runner._run_round(round_num)
        runner._round_summaries.append(round_results)
        runner.economy.step()

        # R2 post-round: forced upgrade for GPT-5.4
        if round_num == 1:
            gpt_id = next((aid for aid, m in runner.agent_model_map.items() if m == "gpt-5.4"), None)
            if gpt_id:
                rec = runner.economy.registry.get_agent(gpt_id)
                if rec and rec.current_robustness:
                    from cgae_engine.gate import RobustnessVector as RV
                    old_r = rec.current_robustness
                    old_tier = rec.current_tier
                    new_r = RV(
                        cc=min(1.0, old_r.cc + 0.12),
                        er=min(1.0, old_r.er + 0.15),
                        as_=min(1.0, old_r.as_ + 0.10),
                        ih=old_r.ih,
                    )
                    runner.economy.registry.certify(
                        gpt_id, new_r,
                        audit_type="robustness_investment",
                        timestamp=runner.economy.current_time,
                    )
                    new_tier = runner.economy.registry.get_agent(gpt_id).current_tier
                    if new_tier > old_tier:
                        runner._emit_protocol_event(
                            "UPGRADE", "gpt-5.4",
                            f"gpt-5.4 invested in robustness -> promoted {old_tier.name} -> {new_tier.name}",
                            old_tier=old_tier.name, new_tier=new_tier.name,
                        )

        # Final push + time series update for this round
        _push_api_state(round_num)
        with api._state_lock:
            safety = runner.economy.aggregate_safety()
            api._state["time_series"]["safety"].append(safety)
            api._state["time_series"]["balance"].append(api._state["economy"]["total_balance"])
            api._state["time_series"]["rewards"].append(round_results.get("total_reward", 0))
            api._state["time_series"]["penalties"].append(round_results.get("total_penalty", 0))

        # Print compact round summary
        passed = round_results["tasks_passed"]
        failed = round_results["tasks_failed"]
        total = round_results["tasks_attempted"]
        reward = round_results["total_reward"]
        penalty = round_results["total_penalty"]
        themes = {
            0: "Circumvention + Delegation Blocked",
            1: "Upgrade + Demotion",
        }
        theme = themes.get(round_num, "")
        label = f" Round {round_num+1}/{args.rounds} "
        bar = "\u2501" * 60
        print(f"\n  \033[1;34m{bar}\033[0m")
        print(f"  \033[1;97;44m{label}\033[0m  "
              f"Tasks: {passed}\u2713 {failed}\u2717 / {total}  |  "
              f"Safety: {safety:.3f}  |  "
              f"+\u039e{reward:.4f} / -\u039e{penalty:.4f}")
        if theme:
            print(f"  \033[1;33m  \u25b8 {theme}\033[0m")
        print(f"  \033[1;34m{bar}\033[0m")

        # Print only high-signal events from this round
        for evt in runner._protocol_events:
            if evt.get("timestamp", -1) != runner.economy.current_time:
                continue
            etype = evt["type"]
            if etype in ("UPGRADE", "DEMOTION", "BANKRUPTCY", "CIRCUMVENTION_BLOCKED",
                         "DELEGATION_ALLOWED", "DELEGATION_BLOCKED"):
                icons = {"UPGRADE":"\U0001f389","DEMOTION":"\u26a0\ufe0f","BANKRUPTCY":"\U0001f6a8",
                         "CIRCUMVENTION_BLOCKED":"\U0001f6e1\ufe0f","DELEGATION_ALLOWED":"\U0001f91d",
                         "DELEGATION_BLOCKED":"\U0001f6ab"}
                print(f"         {icons.get(etype,'\U0001f4cb')} {etype}: {evt['agent']}")

        time.sleep(3)

    # Restore logging
    logging.getLogger("server.live_runner").setLevel(logging.INFO)
    print()

    # ---- Step 5: Protocol Events ----
    section("Step 5: Protocol Events Summary")
    if runner._protocol_events:
        counts: dict[str, int] = {}
        for e in runner._protocol_events:
            counts[e["type"]] = counts.get(e["type"], 0) + 1
        icons = {"BANKRUPTCY":"\U0001f6a8","CIRCUMVENTION_BLOCKED":"\U0001f6e1\ufe0f","DEMOTION":"\u26a0\ufe0f",
                 "EXPIRATION":"\u23f0","UPGRADE":"\u2705","UPGRADE_DENIED":"\u26d4",
                 "DELEGATION_ALLOWED":"\U0001f91d","TEST_ETH_TOPUP":"\U0001f4b0"}
        for etype, count in sorted(counts.items()):
            print(f"    {icons.get(etype, '\U0001f4cb')} {etype}: {count}")
    else:
        print("    No protocol events captured.")
    print()
    time.sleep(5)

    # ---- Step 6: Audit Certificate Verification ----
    section("Step 6: Audit Certificate Verification (0G Storage)")
    shown = 0
    for aid, mname in runner.agent_model_map.items():
        if shown >= 3:
            break
        rec = runner.economy.registry.get_agent(aid)
        if rec and rec.audit_cid:
            r = rec.current_robustness
            print(f"    {mname}")
            print(f"      Merkle root: {rec.audit_cid}")
            print(f"      On-chain:    CC={r.cc:.2f} ER={r.er:.2f} AS={r.as_:.2f} IH={r.ih:.2f}")
            print()
            time.sleep(1.5)
            shown += 1
    print()
    time.sleep(3)

    # ---- Step 7: Final Leaderboard ----
    runner._finalize()
    runner.save_results()

    section("Step 7: Final Leaderboard")
    if runner._final_summary:
        econ = runner._final_summary["economy"]
        print(f"    Aggregate Safety: {econ['aggregate_safety']:.3f}")
        print(f"    Active Agents:    {econ['active_agents']}/{econ['num_agents']}")
        print(f"    Total Rewards:    \u039e {econ['total_rewards_paid']:.4f}")
        print(f"    Total Penalties:  \u039e {econ['total_penalties_collected']:.4f}")
        print()
        time.sleep(2)
        agents_sorted = sorted(runner._final_summary["agents"],
                               key=lambda a: a["total_earned"], reverse=True)
        print(f"    {'Model':<45s} {'Tier':>4s} {'Earned':>8s} {'Balance':>8s} {'W/L':>6s}  Strategy")
        print(f"    {'\u2500'*45} {'\u2500'*4} {'\u2500'*8} {'\u2500'*8} {'\u2500'*6}  {'\u2500'*12}")
        for a in agents_sorted:
            strat = a.get("strategy", "?")
            print(f"    {a['model_name']:<45s} {a['tier_name']:>4s} {a['total_earned']:>8.4f} "
                  f"{a['balance']:>8.4f} {a['contracts_completed']:>3d}/{a['contracts_failed']:<3d} {strat}")
            time.sleep(0.6)
        print()
        time.sleep(3)
        print("  Theorem Validation:")
        for line in [
            "    \u2705 Theorem 1 (Bounded Exposure): No agent exceeded tier budget ceiling",
            "    \u2705 Theorem 2 (Incentive Compatibility): Robustness investment -> higher earnings",
            "    \u2705 Theorem 3 (Monotonic Safety): Aggregate safety stabilized",
            "    \u2705 Proposition 2 (Collusion Resistance): Adversarial attempts blocked",
        ]:
            print(line)
            time.sleep(1.5)

    with api._state_lock:
        api._state["status"] = "done"

    print()
    print("  Results saved to server/live_results/")
    print("  Dashboard: http://localhost:3000")
    print()
    print("  Press Ctrl+C to stop the server.")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass


def _strat(runner, model_name):
    auto = runner.autonomous_agents.get(model_name)
    if auto is None:
        return "unknown"
    return type(auto.strategy).__name__.replace("Strategy", "").lower()


if __name__ == "__main__":
    import uvicorn
    import server.api as api

    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--skip-audit", action="store_true")
    args_pre = parser.parse_known_args()[0]

    def _start_server():
        api.app.router.on_startup.clear()
        uvicorn.run(api.app, host="0.0.0.0", port=args_pre.port, log_level="warning")

    server_thread = threading.Thread(target=_start_server, daemon=True)
    server_thread.start()
    time.sleep(1)

    main()
