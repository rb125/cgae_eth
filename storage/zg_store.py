"""
CGAE 0G Storage — Python Interface
====================================
Uploads CGAE audit certificates to 0G decentralized storage via the
Node.js uploader script (storage/upload_to_0g.mjs).

0G Storage returns a Merkle root hash (bytes32) as the content identifier,
which is stored on-chain in CGAERegistry.certify(). Anyone can verify by
downloading from 0G via the root hash and checking the Merkle proof.

Usage:
    from storage.zg_store import ZgStore, StoreResult

    store = ZgStore()
    result = store.store_audit_result(model_name, audit_json_path)
    print(result.root_hash)   # "0xabc..." or deterministic fallback

0G Integration:
    Real uploads require:
      1. Node.js 18+ with `@0gfoundation/0g-ts-sdk` installed in storage/
      2. ZG_PRIVATE_KEY env var (hex, no 0x prefix)
      3. Wallet funded with testnet tokens from faucet.0g.ai

    Without credentials the store falls back to a deterministic
    content-addressed hash (SHA-256) so the pipeline always has a
    root hash to work with. The 'real' field on StoreResult tells
    callers which mode was used.

Network:
    Default: 0G Testnet
    EVM RPC: https://evmrpc-testnet.0g.ai
    Indexer: https://indexer-storage-testnet-turbo.0g.ai
    Scan:    https://storagescan.0g.ai
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logger = logging.getLogger(__name__)

_STORAGE_DIR = Path(__file__).resolve().parent
_UPLOADER_SCRIPT = _STORAGE_DIR / "upload_to_0g.mjs"

ZG_RPC_URL = "https://evmrpc-testnet.0g.ai"
ZG_INDEXER_RPC = "https://indexer-storage-testnet-turbo.0g.ai"
ZG_STORAGE_SCAN = "https://storagescan.0g.ai"
ZG_FAUCET = "https://faucet.0g.ai"


@dataclass
class StoreResult:
    """Result of a 0G Storage operation."""
    root_hash: str              # Merkle root hash (real) or sha256-derived (fallback)
    real: bool                  # True = uploaded to 0G; False = fallback
    model_name: str
    file_path: str
    size_bytes: int = 0
    error: Optional[str] = None

    @property
    def scan_url(self) -> Optional[str]:
        if self.real:
            return f"{ZG_STORAGE_SCAN}/tx/{self.root_hash}"
        return None

    def to_dict(self) -> dict:
        return {
            "root_hash": self.root_hash,
            "real": self.real,
            "model_name": self.model_name,
            "file_path": self.file_path,
            "size_bytes": self.size_bytes,
            "error": self.error,
            "scan_url": self.scan_url,
        }


class ZgStore:
    """
    Uploads audit JSON files to 0G decentralized storage.

    Falls back to deterministic SHA-256 hash when upload is unavailable.
    """

    def __init__(
        self,
        private_key: Optional[str] = None,
        node_cmd: Optional[str] = None,
        fallback_ok: bool = True,
    ):
        self._private_key = private_key or os.getenv("ZG_PRIVATE_KEY")
        self._node = node_cmd or _find_node()
        self.fallback_ok = fallback_ok

    def store_audit_result(self, model_name: str, json_path: str | Path) -> StoreResult:
        json_path = Path(json_path)
        if not json_path.exists():
            raise FileNotFoundError(f"Audit file not found: {json_path}")

        if self._can_upload():
            try:
                return self._upload_via_0g(model_name, json_path)
            except Exception as e:
                msg = str(e)
                if not self.fallback_ok:
                    raise RuntimeError(f"0G Storage upload failed for {model_name}: {msg}") from e
                logger.warning(f"  [0g] Upload failed for {model_name}: {msg}. Using fallback hash.")
                return self._fallback_result(model_name, json_path, error=msg)
        else:
            reason = self._unavailable_reason()
            if not self.fallback_ok:
                raise RuntimeError(f"0G Storage unavailable: {reason}")
            logger.info(f"  [0g] Upload unavailable ({reason}). Using deterministic hash for {model_name}.")
            return self._fallback_result(model_name, json_path, error=reason)

    def _can_upload(self) -> bool:
        return (
            self._node is not None
            and _UPLOADER_SCRIPT.exists()
            and self._private_key is not None
        )

    def _unavailable_reason(self) -> str:
        if self._node is None:
            return "node.js not found in PATH"
        if not _UPLOADER_SCRIPT.exists():
            return f"uploader script missing: {_UPLOADER_SCRIPT}"
        if self._private_key is None:
            return "ZG_PRIVATE_KEY not set"
        return "unknown"

    def _upload_via_0g(self, model_name: str, json_path: Path) -> StoreResult:
        env = {**os.environ}
        if self._private_key:
            env["ZG_PRIVATE_KEY"] = self._private_key

        cmd = [self._node, str(_UPLOADER_SCRIPT), str(json_path)]

        logger.info(f"  [0g] Uploading {json_path.name} for {model_name}...")
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300, env=env)

        if proc.returncode == 2:
            raise RuntimeError(
                "0G SDK not installed. Run: cd storage && npm install @0gfoundation/0g-ts-sdk ethers"
            )

        if proc.returncode != 0:
            stderr = proc.stderr.strip()
            try:
                err_data = json.loads(stderr)
                raise RuntimeError(err_data.get("error", stderr))
            except (json.JSONDecodeError, KeyError):
                raise RuntimeError(stderr or f"exit code {proc.returncode}")

        data = json.loads(proc.stdout.strip())
        if not data.get("ok"):
            raise RuntimeError(data.get("error", "Unknown upload error"))

        root_hash = data["rootHash"]
        size = data.get("size", json_path.stat().st_size)

        logger.info(f"  [0g] Uploaded {json_path.name} → rootHash {root_hash} ({size} bytes)")
        return StoreResult(
            root_hash=root_hash, real=True, model_name=model_name,
            file_path=str(json_path), size_bytes=size,
        )

    @staticmethod
    def _fallback_result(model_name: str, json_path: Path, error: Optional[str] = None) -> StoreResult:
        content = json_path.read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        pseudo_hash = f"0x{digest}"
        return StoreResult(
            root_hash=pseudo_hash, real=False, model_name=model_name,
            file_path=str(json_path), size_bytes=len(content), error=error,
        )


def _find_node() -> Optional[str]:
    for name in ["node", "nodejs"]:
        try:
            result = subprocess.run([name, "--version"], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                return name
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
    return None


def check_setup() -> dict:
    node = _find_node()
    sdk_installed = False
    if node:
        nm = _STORAGE_DIR / "node_modules" / "@0gfoundation" / "0g-ts-sdk"
        sdk_installed = nm.exists()
    has_key = bool(os.getenv("ZG_PRIVATE_KEY"))
    script_ok = _UPLOADER_SCRIPT.exists()
    ready = node and sdk_installed and has_key and script_ok
    return {
        "ready": ready,
        "node_found": node,
        "sdk_installed": sdk_installed,
        "private_key_set": has_key,
        "uploader_script": script_ok,
        "instructions": (
            None if ready else
            "To enable real 0G uploads:\n"
            "  1. cd storage && npm install @0gfoundation/0g-ts-sdk ethers\n"
            f"  2. Get testnet tokens: {ZG_FAUCET}\n"
            "  3. export ZG_PRIVATE_KEY=<your_hex_private_key>\n"
            "  4. Re-run the simulation"
        ),
    }


if __name__ == "__main__":
    print(json.dumps(check_setup(), indent=2))
