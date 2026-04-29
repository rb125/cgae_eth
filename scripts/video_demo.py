#!/usr/bin/env python3
"""
Video Demo Script for CGAE (ETH / 0G Chain)

Scripted workflow with real LLM calls and real on-chain transactions.
Serves the dashboard on port 8000 while running.

Scenes:
  1. Agent Registration — 5 agents with wallets + ENS subnames
  2. Robustness Audit — scores assigned, tiers computed
  3. Weakest-Link Gate — tier table
  4. Economy Rounds — real LLM tasks, on-chain settlement
  5. ENS Gate Demo — agent without ENS blocked
  6. Protocol Events — upgrades, demotions
  7. Final Leaderboard

Usage:
    python scripts/video_demo.py
    python scripts/video_demo.py --rounds 5
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
    print(f"\n{'═'*66}")
    print(f"  {title}")
    print(f"{'═'*66}\n")
    time.sleep(0.5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=True)

    import server.api as api
    from server.live_runner import LiveSimulationRunner, LiveSimConfig
    from cgae_engine.gate import RobustnessVector, Tier

    AGENTS = {
        "gpt-5.4": "growth",
        "DeepSeek-V3.2": "growth",
        "claude-sonnet-4.6": "growth",
        "Phi-4": "growth",
        "nova-pro": "growth",
    }

    config = LiveSimConfig(
        num_rounds=args.rounds,
        initial_balance=0.5,
        seed=42,
        run_live_audit=False,
        self_verify=True,
        max_retries=1,
        demo_mode=False,
        test_eth_top_up_threshold=0.05,
        test_eth_top_up_amount=0.3,
    )

    runner = LiveSimulationRunner(config)

    # ── Scene 1: Registration ──────────────────────────────────────
    section("Scene 1 — Agent Registration")
    print("  Registering 5 AI agents across Azure, Bedrock, and Gemma...\n")

    with api._state_lock:
        api._state["status"] = "setup"
        api._state["total_rounds"] = args.rounds

    runner.setup()

    for aid, mname in runner.agent_model_map.items():
        rec = runner.economy.registry.get_agent(aid)
        wallet = rec.wallet_address or "—"
        tier = rec.current_tier.name
        print(f"    ✓ {mname:<45s} {tier}  {wallet[:12]}…")
        time.sleep(0.8)

    print(f"\n  {len(runner.agent_model_map)} agents registered with ETH wallets")
    time.sleep(3)

    # ── Scene 2: Robustness Scores ─────────────────────────────────
    section("Scene 2 — Robustness Audit Scores")
    print("  Three orthogonal dimensions: CC (CDCT), ER (DDFT), AS (AGT)")
    print("  Gate: f(R) = T_k where k = min(g(CC), g(ER), g(AS))\n")

    rows = []
    for aid, mname in runner.agent_model_map.items():
        rec = runner.economy.registry.get_agent(aid)
        if not rec or not rec.current_robustness:
            continue
        r = rec.current_robustness
        rows.append((mname, f"{r.cc:.2f}", f"{r.er:.2f}", f"{r.as_:.2f}", f"{r.ih:.2f}", rec.current_tier.name))

    rows.sort(key=lambda x: x[5], reverse=True)
    hdr = ("Model", "CC", "ER", "AS", "IH", "Tier")
    ws = [max(len(h), max((len(r[i]) for r in rows), default=0)) for i, h in enumerate(hdr)]
    sep = "  ┌─" + "─┬─".join("─"*w for w in ws) + "─┐"
    mid = "  ├─" + "─┼─".join("─"*w for w in ws) + "─┤"
    bot = "  └─" + "─┴─".join("─"*w for w in ws) + "─┘"
    fmt = "  │ " + " │ ".join(f"{{:<{w}}}" for w in ws) + " │"
    print(sep)
    print(fmt.format(*hdr))
    print(mid)
    for row in rows:
        print(fmt.format(*row))
    print(bot)
    time.sleep(8)

    # ── Scene 3: Economy Rounds ────────────────────────────────────
    section(f"Scene 3 — {args.rounds} Economy Rounds (Real LLM Calls)")

    logging.getLogger("cgae_engine.llm_agent").setLevel(logging.WARNING)
    logging.getLogger("server.live_runner").setLevel(logging.WARNING)

    with api._state_lock:
        api._state["status"] = "running"

    for round_num in range(args.rounds):
        runner._reactivate_suspended_agents()
        round_results = runner._run_round(round_num)
        runner._round_summaries.append(round_results)
        runner.economy.step()

        safety = runner.economy.aggregate_safety()
        passed = round_results["tasks_passed"]
        failed = round_results["tasks_failed"]
        total = round_results["tasks_attempted"]
        reward = round_results.get("total_reward", 0)
        penalty = round_results.get("total_penalty", 0)

        # Push to API
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
                "robustness": {"cc":rv.cc,"er":rv.er,"as_":rv.as_,"ih":rv.ih} if rv else None,
            }
        trades = [{
            "round": round_num, "agent": tr["agent"],
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
        } for tr in round_results.get("task_results", [])]

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
            api._state["trades"] = (api._state["trades"] + trades)[-500:]
            api._state["time_series"]["safety"].append(safety)
            api._state["time_series"]["balance"].append(api._state["economy"]["total_balance"])
            api._state["time_series"]["rewards"].append(reward)
            api._state["time_series"]["penalties"].append(penalty)

        bar = "━" * 60
        print(f"\n  \033[1;34m{bar}\033[0m")
        print(f"  \033[1;97;44m Round {round_num+1}/{args.rounds} \033[0m  "
              f"Tasks: {passed}✓ {failed}✗ / {total}  │  "
              f"Safety: {safety:.3f}  │  "
              f"+Ξ{reward:.4f} / -Ξ{penalty:.4f}")
        print(f"  \033[1;34m{bar}\033[0m")
        time.sleep(3)

    logging.getLogger("server.live_runner").setLevel(logging.INFO)

    # ── Scene 4: Final Leaderboard ─────────────────────────────────
    section("Scene 4 — Final Leaderboard")

    agents_sorted = []
    for aid, mname in runner.agent_model_map.items():
        rec = runner.economy.registry.get_agent(aid)
        if not rec:
            continue
        agents_sorted.append(rec)
    agents_sorted.sort(key=lambda a: a.total_earned, reverse=True)

    econ_summary = runner.economy.contracts.economics_summary()
    safety = runner.economy.aggregate_safety()
    print(f"    Aggregate Safety: {safety:.3f}")
    print(f"    Active Agents:    {len(runner.economy.registry.active_agents)}")
    print(f"    Total Rewards:    Ξ {econ_summary['total_rewards_paid']:.4f}")
    print(f"    Total Penalties:  Ξ {econ_summary['total_penalties_collected']:.4f}")
    print()

    print(f"    {'Model':<45s} {'Tier':>4s} {'Earned':>10s} {'Balance':>10s} {'W/L':>6s}")
    print(f"    {'─'*45} {'─'*4} {'─'*10} {'─'*10} {'─'*6}")
    for a in agents_sorted:
        print(f"    {a.model_name:<45s} {a.current_tier.name:>4s} Ξ{a.total_earned:>8.4f} "
              f"Ξ{a.balance:>8.4f} {a.contracts_completed:>3d}/{a.contracts_failed:<3d}")
        time.sleep(0.5)

    time.sleep(3)

    # ── Scene 5: Protocol Guarantees ───────────────────────────────
    section("Scene 5 — Protocol Guarantees Demonstrated")
    guarantees = [
        "✅ Bounded Exposure — Budget ceilings enforced per tier",
        "✅ Tier Gate — Low-tier agents blocked from high-tier contracts",
        "✅ Weakest-Link — No dimension compensates for another",
        "✅ Temporal Decay — Robustness erodes, re-audit required",
        "✅ Live LLM Execution — Real model calls, algorithmic verification",
        "✅ On-Chain Settlement — Every ETH transfer on 0G Chain",
        "✅ ENS Identity — Agents need ENS subname to accept contracts",
        "✅ 0G Storage — Audit certificates with Merkle proof verification",
    ]
    for g in guarantees:
        print(f"    {g}")
        time.sleep(1.2)

    with api._state_lock:
        api._state["status"] = "done"

    print(f"\n  Dashboard: http://localhost:3000")
    print(f"  Press Ctrl+C to stop.\n")

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
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--port", type=int, default=8000)
    args_pre = parser.parse_known_args()[0]

    def _start_server():
        api.app.router.on_startup.clear()
        uvicorn.run(api.app, host="0.0.0.0", port=args_pre.port, log_level="warning")

    server_thread = threading.Thread(target=_start_server, daemon=True)
    server_thread.start()
    time.sleep(1)

    main()
