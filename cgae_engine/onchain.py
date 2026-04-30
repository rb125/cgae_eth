"""
CGAE On-Chain Bridge — Writes certifications to CGAERegistry and settles
contracts through CGAEEscrow on 0G Chain.

- CGAERegistry.certify(): robustness vector + 0G Storage root hash on-chain
- CGAEEscrow: full contract lifecycle (create/accept/complete/fail) on-chain
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional

from web3 import Web3
from eth_account import Account

logger = logging.getLogger(__name__)

_CONTRACTS_DIR = Path(__file__).resolve().parent.parent / "contracts"


def _load_registry_abi() -> list:
    abi_path = _CONTRACTS_DIR / "artifacts" / "src" / "CGAERegistry.sol" / "CGAERegistry.json"
    if not abi_path.exists():
        raise FileNotFoundError(f"Registry ABI not found at {abi_path}. Run: cd contracts && npx hardhat compile")
    return json.loads(abi_path.read_text())["abi"]


def _load_deployed() -> dict:
    path = _CONTRACTS_DIR / "deployed.json"
    if not path.exists():
        raise FileNotFoundError("contracts/deployed.json not found. Run: npm run deploy:0g")
    return json.loads(path.read_text())


class OnChainBridge:
    """
    Bridges Python-side certifications to the on-chain CGAERegistry.

    On each certify() call, sends a tx to CGAERegistry.certify() with
    the robustness vector (scaled to uint16) and the 0G Storage root hash.
    """

    def __init__(
        self,
        rpc_url: Optional[str] = None,
        private_key: Optional[str] = None,
        registry_address: Optional[str] = None,
    ):
        self.rpc_url = rpc_url or os.getenv("ZG_RPC_URL", "https://evmrpc-testnet.0g.ai")
        self._key = private_key or os.getenv("PRIVATE_KEY")
        self.w3 = Web3(Web3.HTTPProvider(self.rpc_url))

        if self._key:
            key = self._key if self._key.startswith("0x") else f"0x{self._key}"
            self._account = Account.from_key(key)
        else:
            self._account = None

        # Load registry contract
        if registry_address:
            self._registry_addr = registry_address
        else:
            deployed = _load_deployed()
            self._registry_addr = deployed["contracts"]["CGAERegistry"]["address"]

        abi = _load_registry_abi()
        self.registry = self.w3.eth.contract(
            address=Web3.to_checksum_address(self._registry_addr), abi=abi
        )
        self._tx_log: list[dict] = []

    @property
    def is_live(self) -> bool:
        return self._account is not None

    def certify_agent(
        self,
        agent_address: str,
        cc: float, er: float, as_: float, ih: float,
        audit_type: str = "registration",
        audit_hash: str = "",
    ) -> Optional[str]:
        """
        Register (if needed) then certify an agent on-chain.

        Scores are floats in [0,1], scaled to uint16 [0,10000].
        Returns tx hash or None on failure.
        """
        if not self.is_live:
            logger.info(f"  [onchain] Dry run certify {agent_address[:10]}… (no key)")
            return None

        agent_addr = Web3.to_checksum_address(agent_address)

        # Auto-register if not yet on-chain
        try:
            record = self.registry.functions.getAgent(agent_addr).call()
            if record[4] == 0:  # registrationTime == 0 means not registered
                self._register_agent_onchain(agent_addr, audit_type)
        except Exception:
            self._register_agent_onchain(agent_addr, audit_type)

        # Scale [0,1] → [0,10000]
        cc_u = min(10000, int(cc * 10000))
        er_u = min(10000, int(er * 10000))
        as_u = min(10000, int(as_ * 10000))
        ih_u = min(10000, int(ih * 10000))

        try:
            nonce = self.w3.eth.get_transaction_count(self._account.address)
            tx = self.registry.functions.certify(
                Web3.to_checksum_address(agent_address),
                cc_u, er_u, as_u, ih_u,
                audit_type,
                audit_hash or "",
            ).build_transaction({
                "from": self._account.address,
                "nonce": nonce,
                "gas": 300_000,
                "gasPrice": self.w3.eth.gas_price,
                "chainId": self.w3.eth.chain_id,
            })

            signed = self._account.sign_transaction(tx)
            tx_hash = self.w3.eth.send_raw_transaction(signed.raw_transaction)
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash, timeout=60)

            result = {
                "agent": agent_address,
                "tx_hash": tx_hash.hex(),
                "status": "confirmed" if receipt["status"] == 1 else "failed",
                "scores": {"cc": cc, "er": er, "as": as_, "ih": ih},
                "audit_hash": audit_hash,
            }
            self._tx_log.append(result)
            logger.info(f"  [onchain] Certified {agent_address[:10]}… tx={tx_hash.hex()[:16]}…")
            return tx_hash.hex()

        except Exception as e:
            logger.error(f"  [onchain] Certify failed for {agent_address[:10]}…: {e}")
            self._tx_log.append({"agent": agent_address, "error": str(e)})
            return None

    @property
    def tx_log(self) -> list[dict]:
        return list(self._tx_log)

    def _register_agent_onchain(self, agent_addr: str, model_name: str = "cgae-agent"):
        """Register an agent on-chain via registerAgent()."""
        try:
            arch_hash = Web3.keccak(text=model_name)[:16]  # first 16 bytes of keccak(model_name)
            nonce = self.w3.eth.get_transaction_count(self._account.address)
            tx = self.registry.functions.registerAgent(
                agent_addr, arch_hash, model_name
            ).build_transaction({
                "from": self._account.address,
                "nonce": nonce,
                "gas": 200_000,
                "gasPrice": self.w3.eth.gas_price,
                "chainId": self.w3.eth.chain_id,
            })
            signed = self._account.sign_transaction(tx)
            tx_hash = self.w3.eth.send_raw_transaction(signed.raw_transaction)
            self.w3.eth.wait_for_transaction_receipt(tx_hash, timeout=60)
            logger.info(f"  [onchain] Registered {agent_addr[:10]}… tx={tx_hash.hex()[:16]}…")
        except Exception as e:
            logger.warning(f"  [onchain] Register failed for {agent_addr[:10]}…: {e}")


def _load_escrow_abi() -> list:
    abi_path = _CONTRACTS_DIR / "artifacts" / "src" / "CGAEEscrow.sol" / "CGAEEscrow.json"
    if not abi_path.exists():
        raise FileNotFoundError(f"Escrow ABI not found at {abi_path}. Run: cd contracts && npx hardhat compile")
    return json.loads(abi_path.read_text())["abi"]


class EscrowBridge:
    """
    Bridges Python-side contract lifecycle to CGAEEscrow on 0G Chain.

    Full on-chain settlement: createContract (payable, escrows reward),
    acceptContract (payable, agent deposits penalty collateral),
    completeContract / failContract.
    """

    def __init__(
        self,
        rpc_url: Optional[str] = None,
        private_key: Optional[str] = None,
        escrow_address: Optional[str] = None,
    ):
        self.rpc_url = rpc_url or os.getenv("ZG_RPC_URL", "https://evmrpc-testnet.0g.ai")
        self._key = private_key or os.getenv("PRIVATE_KEY")
        self.w3 = Web3(Web3.HTTPProvider(self.rpc_url))

        if self._key:
            key = self._key if self._key.startswith("0x") else f"0x{self._key}"
            self._account = Account.from_key(key)
        else:
            self._account = None

        if escrow_address:
            self._escrow_addr = escrow_address
        else:
            self._escrow_addr = os.getenv("CGAE_ESCROW_ADDRESS")
            if not self._escrow_addr:
                deployed = _load_deployed()
                self._escrow_addr = deployed["contracts"]["CGAEEscrow"]["address"]

        abi = _load_escrow_abi()
        self.escrow = self.w3.eth.contract(
            address=Web3.to_checksum_address(self._escrow_addr), abi=abi
        )
        self._tx_log: list[dict] = []

    @property
    def is_live(self) -> bool:
        return self._account is not None

    def _send_tx(self, fn, value_wei: int = 0, gas: int = 500_000) -> Optional[str]:
        if not self.is_live:
            return None
        try:
            nonce = self.w3.eth.get_transaction_count(self._account.address)
            tx = fn.build_transaction({
                "from": self._account.address,
                "nonce": nonce,
                "gas": gas,
                "gasPrice": self.w3.eth.gas_price,
                "chainId": self.w3.eth.chain_id,
                "value": value_wei,
            })
            signed = self._account.sign_transaction(tx)
            tx_hash = self.w3.eth.send_raw_transaction(signed.raw_transaction)
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash, timeout=60)
            status = "confirmed" if receipt["status"] == 1 else "failed"
            self._tx_log.append({"tx_hash": tx_hash.hex(), "status": status})
            return tx_hash.hex()
        except Exception as e:
            logger.warning(f"  [escrow] tx skipped (insufficient gas or network error): {e}")
            self._tx_log.append({"error": str(e)})
            return None

    def create_contract(
        self,
        objective: str,
        constraints_hash: bytes,
        verifier_spec_hash: str,
        min_tier: int,
        reward_wei: int,
        penalty_wei: int,
        deadline: int,
        domain: str,
    ) -> Optional[tuple[str, bytes]]:
        """
        Create a contract on-chain. Sends reward_wei as escrow.
        Returns (tx_hash, contract_id) or None.
        """
        if not self.is_live:
            logger.info(f"  [escrow] Dry run createContract (no key)")
            return None

        fn = self.escrow.functions.createContract(
            objective[:200],
            constraints_hash,
            verifier_spec_hash,
            min_tier,
            penalty_wei,
            deadline,
            domain,
        )
        tx_hash = self._send_tx(fn, value_wei=reward_wei)
        if not tx_hash:
            return None

        # Extract contract_id from ContractCreated event
        receipt = self.w3.eth.get_transaction_receipt(tx_hash)
        logs = self.escrow.events.ContractCreated().process_receipt(receipt)
        if logs:
            contract_id = logs[0]["args"]["contractId"]
            logger.info(f"  [escrow] Created contract tx={tx_hash[:16]}... id={contract_id.hex()[:16]}...")
            return tx_hash, contract_id
        logger.info(f"  [escrow] Created contract tx={tx_hash[:16]}...")
        return tx_hash, None

    def accept_contract(self, contract_id: bytes, penalty_wei: int) -> Optional[str]:
        """Agent accepts contract, depositing penalty as collateral."""
        fn = self.escrow.functions.acceptContract(contract_id)
        tx_hash = self._send_tx(fn, value_wei=penalty_wei)
        if tx_hash:
            logger.info(f"  [escrow] Accepted contract tx={tx_hash[:16]}...")
        return tx_hash

    def complete_contract(self, contract_id: bytes) -> Optional[str]:
        """Mark contract completed. Releases reward to agent + returns collateral."""
        fn = self.escrow.functions.completeContract(contract_id)
        tx_hash = self._send_tx(fn)
        if tx_hash:
            logger.info(f"  [escrow] Completed contract tx={tx_hash[:16]}...")
        return tx_hash

    def fail_contract(self, contract_id: bytes) -> Optional[str]:
        """Mark contract failed. Penalty forfeited, reward returned to issuer."""
        fn = self.escrow.functions.failContract(contract_id)
        tx_hash = self._send_tx(fn)
        if tx_hash:
            logger.info(f"  [escrow] Failed contract tx={tx_hash[:16]}...")
        return tx_hash

    def get_economics_summary(self) -> Optional[dict]:
        """Read on-chain economics summary."""
        try:
            result = self.escrow.functions.getEconomicsSummary().call()
            return {
                "total_rewards_paid": result[0],
                "total_penalties_collected": result[1],
                "total_escrowed": result[2],
                "contract_count": result[3],
            }
        except Exception as e:
            logger.error(f"  [escrow] getEconomicsSummary failed: {e}")
            return None

    @property
    def tx_log(self) -> list[dict]:
        return list(self._tx_log)
