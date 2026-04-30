"""
CGAE ENS Integration — Real ENS names for AI agents on Sepolia.

Each agent gets a subname under a parent ENS name (e.g., gpt5.cgaeprotocol.eth)
with text records storing robustness scores, tier, wallet address, and 0G audit hash.

This uses the ENS NameWrapper + PublicResolver on Sepolia to:
1. Create subnames for each agent (setSubnodeRecord)
2. Set text records with robustness credentials
3. Enable resolution: anyone can look up gpt5.cgaeprotocol.eth → get scores + wallet

Requirements:
  - Parent ENS name registered on Sepolia (e.g., cgaeprotocol.eth)
  - Parent name wrapped in NameWrapper
  - Sepolia ETH for gas
  - SEPOLIA_RPC_URL and PRIVATE_KEY in env
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Optional

from eth_account import Account
from web3 import Web3

logger = logging.getLogger(__name__)

# Sepolia ENS contract addresses (from docs.ens.domains/learn/deployments)
ENS_REGISTRY = "0x00000000000C2E074eC69A0dFb2997BA6C7d2e1e"
NAME_WRAPPER = "0x0635513f179D50A207757E05759CbD106d7dFcE8"
PUBLIC_RESOLVER = "0xE99638b40E4Fff0129D56f03b55b6bbC4BBE49b5"

# Minimal ABIs for the functions we need
NAME_WRAPPER_ABI = json.loads("""[
  {
    "inputs": [
      {"name": "parentNode", "type": "bytes32"},
      {"name": "label", "type": "string"},
      {"name": "owner", "type": "address"},
      {"name": "fuses", "type": "uint32"},
      {"name": "expiry", "type": "uint64"}
    ],
    "name": "setSubnodeOwner",
    "outputs": [{"name": "node", "type": "bytes32"}],
    "stateMutability": "nonpayable",
    "type": "function"
  },
  {
    "inputs": [
      {"name": "parentNode", "type": "bytes32"},
      {"name": "label", "type": "string"},
      {"name": "owner", "type": "address"},
      {"name": "resolver", "type": "address"},
      {"name": "ttl", "type": "uint64"},
      {"name": "fuses", "type": "uint32"},
      {"name": "expiry", "type": "uint64"}
    ],
    "name": "setSubnodeRecord",
    "outputs": [{"name": "node", "type": "bytes32"}],
    "stateMutability": "nonpayable",
    "type": "function"
  }
]""")

RESOLVER_ABI = json.loads("""[
  {
    "inputs": [
      {"name": "node", "type": "bytes32"},
      {"name": "key", "type": "string"},
      {"name": "value", "type": "string"}
    ],
    "name": "setText",
    "outputs": [],
    "stateMutability": "nonpayable",
    "type": "function"
  },
  {
    "inputs": [
      {"name": "node", "type": "bytes32"},
      {"name": "key", "type": "string"}
    ],
    "name": "text",
    "outputs": [{"name": "", "type": "string"}],
    "stateMutability": "view",
    "type": "function"
  },
  {
    "inputs": [
      {"name": "node", "type": "bytes32"},
      {"name": "coinType", "type": "uint256"}
    ],
    "name": "addr",
    "outputs": [{"name": "", "type": "bytes"}],
    "stateMutability": "view",
    "type": "function"
  }
]""")


def namehash(name: str) -> bytes:
    """Compute ENS namehash (EIP-137)."""
    node = b"\x00" * 32
    if name:
        labels = name.split(".")
        for label in reversed(labels):
            label_hash = Web3.keccak(text=label)
            node = Web3.keccak(node + label_hash)
    return node


def _slugify(name: str) -> str:
    s = name.lower().replace("_", "-").replace(" ", "-").replace(".", "-")
    s = re.sub(r"[^a-z0-9-]", "", s)
    return re.sub(r"-+", "-", s).strip("-") or "agent"


class ENSManager:
    """
    Manages ENS subnames for CGAE agents on Sepolia.

    Creates subnames under a parent name and sets text records
    with robustness scores, tier, and 0G audit provenance.
    """

    def __init__(
        self,
        parent_name: str = "cgaeprotocol.eth",
        rpc_url: Optional[str] = None,
        private_key: Optional[str] = None,
    ):
        self.parent_name = parent_name
        self.rpc_url = rpc_url or os.getenv("SEPOLIA_RPC_URL", "https://ethereum-sepolia-rpc.publicnode.com")
        self._key = private_key or os.getenv("PRIVATE_KEY")

        self.w3 = Web3(Web3.HTTPProvider(self.rpc_url))

        if self._key:
            key = self._key if self._key.startswith("0x") else f"0x{self._key}"
            self._account = Account.from_key(key)
        else:
            self._account = None

        self.name_wrapper = self.w3.eth.contract(
            address=Web3.to_checksum_address(NAME_WRAPPER), abi=NAME_WRAPPER_ABI
        )
        self.resolver = self.w3.eth.contract(
            address=Web3.to_checksum_address(PUBLIC_RESOLVER), abi=RESOLVER_ABI
        )
        self.parent_node = namehash(parent_name)
        self._subnames: dict[str, str] = {}  # agent_id -> full ENS name

    @property
    def is_live(self) -> bool:
        return self._account is not None

    def create_subname(self, agent_id: str, model_name: str, owner: str) -> Optional[str]:
        """
        Create a subname like gpt5.cgaeprotocol.eth for an agent.
        If the subname already exists (has a cgae.tier record), reuse it.
        Returns the full ENS name or None on failure.
        """
        label = _slugify(model_name)
        full_name = f"{label}.{self.parent_name}"

        if not self.is_live:
            logger.info(f"  [ens] Dry run: would create {full_name}")
            self._subnames[agent_id] = full_name
            return full_name

        # Check if subname already exists by reading a text record
        existing_tier = self.resolve_text(full_name, "cgae.tier")
        if existing_tier:
            logger.info(f"  [ens] Reusing existing {full_name} (tier={existing_tier})")
            self._subnames[agent_id] = full_name
            return full_name

        try:
            nonce = self.w3.eth.get_transaction_count(self._account.address)
            # setSubnodeRecord creates the subname + sets resolver in one tx
            tx = self.name_wrapper.functions.setSubnodeRecord(
                self.parent_node,
                label,
                Web3.to_checksum_address(owner),
                Web3.to_checksum_address(PUBLIC_RESOLVER),
                0,   # ttl
                0,   # fuses (no restrictions)
                2**64 - 1,  # max expiry
            ).build_transaction({
                "from": self._account.address,
                "nonce": nonce,
                "gas": 300_000,
                "gasPrice": self.w3.eth.gas_price,
                "chainId": self.w3.eth.chain_id,
            })
            signed = self._account.sign_transaction(tx)
            tx_hash = self.w3.eth.send_raw_transaction(signed.raw_transaction)
            self.w3.eth.wait_for_transaction_receipt(tx_hash, timeout=60)

            self._subnames[agent_id] = full_name
            logger.info(f"  [ens] Created {full_name} tx={tx_hash.hex()[:16]}…")
            return full_name

        except Exception as e:
            logger.error(f"  [ens] Failed to create {full_name}: {e}")
            self._subnames[agent_id] = full_name  # store anyway for display
            return None

    def set_text_records(
        self,
        agent_id: str,
        records: dict[str, str],
    ) -> int:
        """
        Set multiple text records on an agent's ENS subname.
        Returns number of records successfully set.
        """
        full_name = self._subnames.get(agent_id)
        if not full_name or not self.is_live:
            return 0

        node = namehash(full_name)
        count = 0

        for key, value in records.items():
            try:
                nonce = self.w3.eth.get_transaction_count(self._account.address)
                tx = self.resolver.functions.setText(
                    node, key, value
                ).build_transaction({
                    "from": self._account.address,
                    "nonce": nonce,
                    "gas": 100_000,
                    "gasPrice": self.w3.eth.gas_price,
                    "chainId": self.w3.eth.chain_id,
                })
                signed = self._account.sign_transaction(tx)
                tx_hash = self.w3.eth.send_raw_transaction(signed.raw_transaction)
                self.w3.eth.wait_for_transaction_receipt(tx_hash, timeout=60)
                count += 1
            except Exception as e:
                logger.warning(f"  [ens] setText({key}) failed for {full_name}: {e}")

        if count:
            logger.info(f"  [ens] Set {count}/{len(records)} text records on {full_name}")
        return count

    def set_agent_credentials(
        self,
        agent_id: str,
        tier: str,
        cc: float, er: float, as_: float, ih: float,
        wallet_address: str = "",
        audit_hash: str = "",
        family: str = "",
    ) -> int:
        """Set robustness credentials as ENS text records."""
        records = {
            "cgae.tier": tier,
            "cgae.cc": f"{cc:.4f}",
            "cgae.er": f"{er:.4f}",
            "cgae.as": f"{as_:.4f}",
            "cgae.ih": f"{ih:.4f}",
        }
        if wallet_address:
            records["cgae.wallet"] = wallet_address
        if audit_hash:
            records["cgae.0g-audit-hash"] = audit_hash
        if family:
            records["cgae.family"] = family
        return self.set_text_records(agent_id, records)

    def resolve_text(self, ens_name: str, key: str) -> str:
        """Read a text record from an ENS name."""
        node = namehash(ens_name)
        try:
            return self.resolver.functions.text(node, key).call()
        except Exception:
            return ""

    def get_agent_name(self, agent_id: str) -> Optional[str]:
        return self._subnames.get(agent_id)

    def all_subnames(self) -> dict[str, str]:
        return dict(self._subnames)
