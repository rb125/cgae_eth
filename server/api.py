"""
CGAE Live Economy Server — ETH / 0G Chain

Runs the LiveSimulationRunner in a background thread and exposes
real-time state via REST endpoints for the dashboard.

Usage:
    python -m server.api                     # default 20 rounds
    python -m server.api --rounds 50
    python -m server.api --rounds -1         # infinite
"""

import argparse
import json
import logging
import threading
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

app = FastAPI(title="CGAE Live Economy")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

logger = logging.getLogger("cgae.api")

_state: dict = {
    "status": "idle",
    "round": 0,
    "total_rounds": 0,
    "economy": None,
    "agents": {},
    "trades": [],
    "events": [],
    "time_series": {"safety": [], "balance": [], "rewards": [], "penalties": []},
}
_state_lock = threading.Lock()
MAX_TRADES = 500

DEPLOYED = Path(__file__).resolve().parents[1] / "contracts" / "deployed.json"


def _run_economy(num_rounds: int, initial_balance: float):
    import sys, os
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=True)

    from server.live_runner import LiveSimulationRunner, LiveSimConfig

    config = LiveSimConfig(
        num_rounds=num_rounds,
        initial_balance=initial_balance,
        run_live_audit=False,
        self_verify=True,
        max_retries=1,
        test_eth_top_up_threshold=0.05,
        test_eth_top_up_amount=0.3,
    )

    runner = LiveSimulationRunner(config)

    with _state_lock:
        _state["status"] = "setup"
        _state["total_rounds"] = num_rounds

    runner.setup()

    with _state_lock:
        _state["status"] = "running"

    round_num = 0
    infinite = num_rounds == -1

    try:
        while infinite or round_num < num_rounds:
            runner._reactivate_suspended_agents()
            round_results = runner._run_round(round_num)
            runner._round_summaries.append(round_results)
            step_events = runner.economy.step()

            safety = runner.economy.aggregate_safety()
            agents_snapshot = {}
            for aid, mname in runner.agent_model_map.items():
                rec = runner.economy.registry.get_agent(aid)
                if not rec:
                    continue
                r = rec.current_robustness
                agents_snapshot[aid] = {
                    "agent_id": aid,
                    "model_name": mname,
                    "strategy": _get_strategy(runner, mname),
                    "current_tier": rec.current_tier.value,
                    "balance": rec.balance,
                    "total_earned": rec.total_earned,
                    "total_penalties": rec.total_penalties,
                    "contracts_completed": rec.contracts_completed,
                    "contracts_failed": rec.contracts_failed,
                    "status": rec.status.value,
                    "wallet_address": rec.wallet_address,
                    "ens_name": runner.economy.ens_manager.get_agent_name(aid) if runner.economy.ens_manager else None,
                    "robustness": {
                        "cc": r.cc, "er": r.er, "as_": r.as_, "ih": r.ih,
                    } if r else None,
                }

            trades = []
            for tr in round_results.get("task_results", []):
                trades.append({
                    "round": round_num,
                    "agent": tr["agent"],
                    "task_id": tr["task_id"],
                    "task_prompt": tr.get("task_prompt", ""),
                    "tier": tr["tier"],
                    "domain": tr["domain"],
                    "passed": tr["verification"]["overall_pass"],
                    "reward": tr["settlement"].get("reward", 0) if tr["settlement"] else 0,
                    "penalty": tr["settlement"].get("penalty", 0) if tr["settlement"] else 0,
                    "token_cost": tr.get("token_cost_eth", 0),
                    "latency_ms": tr.get("latency_ms", 0),
                    "output_preview": tr.get("output_preview", ""),
                    "constraints_passed": tr["verification"].get("constraints_passed", []),
                    "constraints_failed": tr["verification"].get("constraints_failed", []),
                })

            # Capture protocol events from step
            for aid in step_events.get("agents_demoted", []):
                mname = runner.agent_model_map.get(aid, aid)
                with _state_lock:
                    _state["events"].append({"timestamp": runner.economy.current_time, "type": "DEMOTION", "agent": mname, "message": f"{mname} demoted after spot-audit failure"})

            with _state_lock:
                _state["round"] = round_num + 1
                _state["economy"] = {
                    "aggregate_safety": safety,
                    "active_agents": len(runner.economy.registry.active_agents),
                    "total_balance": sum(a["balance"] for a in agents_snapshot.values()),
                    "total_earned": sum(a["total_earned"] for a in agents_snapshot.values()),
                    "contracts_completed": sum(a["contracts_completed"] for a in agents_snapshot.values()),
                    "contracts_failed": sum(a["contracts_failed"] for a in agents_snapshot.values()),
                }
                _state["agents"] = agents_snapshot
                _state["trades"] = (_state["trades"] + trades)[-MAX_TRADES:]
                _state["time_series"]["safety"].append(safety)
                _state["time_series"]["balance"].append(_state["economy"]["total_balance"])
                _state["time_series"]["rewards"].append(round_results.get("total_reward", 0))
                _state["time_series"]["penalties"].append(round_results.get("total_penalty", 0))

            round_num += 1

    except Exception as e:
        logger.exception(f"Economy runner failed: {e}")
    finally:
        with _state_lock:
            _state["status"] = "done"


def _get_strategy(runner, model_name: str) -> str:
    auto = runner.autonomous_agents.get(model_name)
    if auto is None:
        return "unknown"
    return type(auto.strategy).__name__.replace("Strategy", "").lower()


@app.get("/api/state")
def get_state():
    with _state_lock:
        return {"status": _state["status"], "round": _state["round"], "total_rounds": _state["total_rounds"], "economy": _state["economy"]}


@app.get("/api/agents")
def get_agents():
    with _state_lock:
        return {"agents": list(_state["agents"].values())}


@app.get("/api/trades")
def get_trades(limit: int = 100):
    with _state_lock:
        return {"trades": _state["trades"][-limit:]}


@app.get("/api/events")
def get_events(limit: int = 100):
    with _state_lock:
        return {"events": _state["events"][-limit:]}


@app.get("/api/timeseries")
def get_timeseries():
    with _state_lock:
        return _state["time_series"]


@app.get("/api/contracts")
def get_contracts():
    if DEPLOYED.exists():
        return json.loads(DEPLOYED.read_text())
    return {}


_runner_thread = None


def start_economy(rounds: int = 20, balance: float = 0.5):
    global _runner_thread
    if _runner_thread and _runner_thread.is_alive():
        return
    _runner_thread = threading.Thread(target=_run_economy, args=(rounds, balance), daemon=True)
    _runner_thread.start()


@app.on_event("startup")
async def on_startup():
    import sys
    rounds = 20
    for i, arg in enumerate(sys.argv):
        if arg == "--rounds" and i + 1 < len(sys.argv):
            rounds = int(sys.argv[i + 1])
    start_economy(rounds=rounds)


# Serve Next.js static export if available (for HF Spaces / single-container deploy)
_frontend_dir = Path(__file__).resolve().parent.parent / "dashboard-next" / "out"
if _frontend_dir.exists():
    @app.get("/")
    async def serve_index():
        return FileResponse(str(_frontend_dir / "index.html"))
    app.mount("/", StaticFiles(directory=str(_frontend_dir), html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    uvicorn.run(app, host="0.0.0.0", port=args.port)
