"""
CGAE Wallet Manager — Real ETH wallet integration for agents.

Each agent gets a unique Ethereum keypair on registration.
A treasury wallet funds agents and disburses ETH on successful contract completion.

Usage:
    from cgae_engine.wallet import WalletManager

    wm = WalletManager(rpc_url="https://evmrpc-testnet.0g.ai",
                        treasury_private_key=os.getenv("PRIVATE_KEY"))
    wallet = wm.create_agent_wallet("agent_abc123")
    tx = wm.disburse(wallet.address, amount_wei=10**15)  # 0.001 ETH
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from eth_account import Account
from web3 import Web3

logger = logging.getLogger(__name__)


@dataclass
class AgentWallet:
    """An agent's Ethereum wallet."""
    agent_id: str
    address: str
    private_key: str  # hex with 0x prefix

    def to_dict(self, redact_key: bool = True) -> dict:
        return {
            "agent_id": self.agent_id,
            "address": self.address,
            "private_key": self.private_key[:10] + "..." if redact_key else self.private_key,
        }


class WalletManager:
    """
    Manages ETH wallets for CGAE agents.

    - Creates a unique keypair per agent
    - Sends real ETH from a treasury wallet on contract settlement
    - Tracks all disbursements for audit trail
    """

    def __init__(
        self,
        rpc_url: Optional[str] = None,
        treasury_private_key: Optional[str] = None,
        dry_run: bool = False,
        wallet_store_path: Optional[str] = None,
    ):
        self.rpc_url = rpc_url or os.getenv("ZG_RPC_URL", "https://evmrpc-testnet.0g.ai")
        self._treasury_key = treasury_private_key or os.getenv("PRIVATE_KEY")
        self.dry_run = dry_run

        self.w3 = Web3(Web3.HTTPProvider(self.rpc_url))
        self._wallets: dict[str, AgentWallet] = {}  # agent_id -> wallet
        self._disbursements: list[dict] = []
        self._store_path = Path(wallet_store_path) if wallet_store_path else Path("server/live_results/wallets.json")

        if self._treasury_key:
            key = self._treasury_key if self._treasury_key.startswith("0x") else f"0x{self._treasury_key}"
            self._treasury_account = Account.from_key(key)
            self.treasury_address = self._treasury_account.address
        else:
            self._treasury_account = None
            self.treasury_address = None

        # Load persisted wallets from disk
        self._model_wallets: dict[str, AgentWallet] = {}  # model_name -> wallet
        self._load_wallets()

    @property
    def is_live(self) -> bool:
        """True if we can send real transactions."""
        return self._treasury_account is not None and not self.dry_run

    def create_agent_wallet(self, agent_id: str, model_name: str = "") -> AgentWallet:
        """Get existing wallet for this model or generate a new keypair."""
        if agent_id in self._wallets:
            return self._wallets[agent_id]

        # Reuse persisted wallet for this model if it exists
        if model_name and model_name in self._model_wallets:
            wallet = AgentWallet(
                agent_id=agent_id,
                address=self._model_wallets[model_name].address,
                private_key=self._model_wallets[model_name].private_key,
            )
            self._wallets[agent_id] = wallet
            logger.info(f"  [wallet] Loaded existing wallet for {agent_id}: {wallet.address}")
            return wallet

        acct = Account.create()
        wallet = AgentWallet(
            agent_id=agent_id,
            address=acct.address,
            private_key=acct.key.hex() if isinstance(acct.key, bytes) else acct.key,
        )
        self._wallets[agent_id] = wallet
        if model_name:
            self._model_wallets[model_name] = wallet
            self._save_wallets()
        logger.info(f"  [wallet] Created wallet for {agent_id}: {wallet.address}")
        return wallet

    def _load_wallets(self):
        """Load persisted model->wallet mapping from disk."""
        if self._store_path.exists():
            try:
                data = json.loads(self._store_path.read_text())
                for model_name, w in data.items():
                    self._model_wallets[model_name] = AgentWallet(
                        agent_id=w.get("agent_id", ""),
                        address=w["address"],
                        private_key=w["private_key"],
                    )
                logger.info(f"  [wallet] Loaded {len(self._model_wallets)} persisted wallets")
            except Exception as e:
                logger.warning(f"  [wallet] Could not load wallets: {e}")

    def _save_wallets(self):
        """Persist model->wallet mapping to disk (unredacted keys)."""
        self._store_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            model: {"agent_id": w.agent_id, "address": w.address, "private_key": w.private_key}
            for model, w in self._model_wallets.items()
        }
        self._store_path.write_text(json.dumps(data, indent=2))

    def get_wallet(self, agent_id: str) -> Optional[AgentWallet]:
        return self._wallets.get(agent_id)

    def get_balance(self, address: str) -> float:
        """Get balance in ETH."""
        try:
            wei = self.w3.eth.get_balance(Web3.to_checksum_address(address))
            return float(Web3.from_wei(wei, "ether"))
        except Exception as e:
            logger.warning(f"  [wallet] Balance check failed for {address}: {e}")
            return 0.0

    def get_treasury_balance(self) -> float:
        if not self.treasury_address:
            return 0.0
        return self.get_balance(self.treasury_address)

    def disburse(self, to_address: str, amount_eth: float, reason: str = "") -> Optional[dict]:
        """
        Send ETH from treasury to an agent wallet.
        Returns tx receipt dict or None if dry_run / failed.
        """
        amount_wei = Web3.to_wei(amount_eth, "ether")

        record = {
            "to": to_address,
            "amount_eth": amount_eth,
            "reason": reason,
            "tx_hash": None,
            "status": "pending",
        }

        if not self.is_live:
            record["status"] = "dry_run"
            self._disbursements.append(record)
            logger.info(f"  [wallet] DRY RUN disburse {amount_eth:.6f} ETH → {to_address} ({reason})")
            return record

        try:
            nonce = self.w3.eth.get_transaction_count(self.treasury_address)
            tx = {
                "from": self.treasury_address,
                "to": Web3.to_checksum_address(to_address),
                "value": amount_wei,
                "gas": 21000,
                "gasPrice": self.w3.eth.gas_price,
                "nonce": nonce,
                "chainId": self.w3.eth.chain_id,
            }
            signed = self._treasury_account.sign_transaction(tx)
            tx_hash = self.w3.eth.send_raw_transaction(signed.raw_transaction)
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash, timeout=60)

            record["tx_hash"] = tx_hash.hex()
            record["status"] = "confirmed" if receipt["status"] == 1 else "failed"
            self._disbursements.append(record)

            logger.info(
                f"  [wallet] Disbursed {amount_eth:.6f} ETH → {to_address} "
                f"(tx: {tx_hash.hex()[:16]}… status: {record['status']})"
            )
            return record

        except Exception as e:
            record["status"] = f"error: {e}"
            self._disbursements.append(record)
            logger.error(f"  [wallet] Disburse failed: {e}")
            return record

    def fund_agent(self, agent_id: str, amount_eth: float) -> Optional[dict]:
        """Convenience: disburse ETH to an agent by agent_id."""
        wallet = self._wallets.get(agent_id)
        if not wallet:
            logger.warning(f"  [wallet] No wallet for {agent_id}")
            return None
        return self.disburse(wallet.address, amount_eth, reason=f"fund:{agent_id}")

    def disburse_reward(self, agent_id: str, amount_eth: float, contract_id: str = "") -> Optional[dict]:
        """Disburse contract reward to agent."""
        wallet = self._wallets.get(agent_id)
        if not wallet:
            return None
        return self.disburse(wallet.address, amount_eth, reason=f"reward:{contract_id}")

    @property
    def disbursements(self) -> list[dict]:
        return list(self._disbursements)

    def summary(self) -> dict:
        total_disbursed = sum(d["amount_eth"] for d in self._disbursements if d["status"] in ("confirmed", "dry_run"))
        return {
            "treasury_address": self.treasury_address,
            "treasury_balance_eth": self.get_treasury_balance() if self.is_live else None,
            "agents_with_wallets": len(self._wallets),
            "total_disbursements": len(self._disbursements),
            "total_eth_disbursed": total_disbursed,
            "is_live": self.is_live,
            "rpc_url": self.rpc_url,
        }

    def export_wallets(self, path: str, redact_keys: bool = True):
        """Export wallet mapping to JSON."""
        data = {aid: w.to_dict(redact_key=redact_keys) for aid, w in self._wallets.items()}
        Path(path).write_text(json.dumps(data, indent=2))
