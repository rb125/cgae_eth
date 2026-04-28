#!/usr/bin/env node
/**
 * CGAE 0G Storage — Audit Certificate Uploader
 * ==============================================
 * Uploads a JSON audit certificate to 0G decentralized storage
 * and prints the resulting Merkle root hash to stdout.
 *
 * Usage:
 *   node upload_to_0g.mjs <file_path>
 *
 * Required env vars:
 *   ZG_PRIVATE_KEY   — hex private key (no 0x prefix) of the funding wallet
 *
 * Optional env vars:
 *   ZG_RPC_URL       — override EVM RPC (default: 0G testnet)
 *   ZG_INDEXER_RPC   — override indexer RPC
 *
 * Output (stdout, JSON):
 *   { "rootHash": "0xabc...", "size": 1234, "ok": true }
 *
 * On error (stderr + exit 1):
 *   { "error": "...", "ok": false }
 *
 * Install deps (from storage/):
 *   npm install @0gfoundation/0g-ts-sdk ethers
 */

import { readFileSync, statSync } from "fs";
import { resolve } from "path";

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

const args = process.argv.slice(2);
if (args.length === 0 || args[0] === "--help") {
  console.error("Usage: node upload_to_0g.mjs <file_path>");
  process.exit(1);
}

const filePath = resolve(args[0]);

// ---------------------------------------------------------------------------
// Environment
// ---------------------------------------------------------------------------

const privateKey = process.env.ZG_PRIVATE_KEY;
if (!privateKey) {
  writeError("ZG_PRIVATE_KEY environment variable not set");
  process.exit(1);
}

const RPC_URL = process.env.ZG_RPC_URL || "https://evmrpc-testnet.0g.ai";
const INDEXER_RPC = process.env.ZG_INDEXER_RPC || "https://indexer-storage-testnet-turbo.0g.ai";

// ---------------------------------------------------------------------------
// Main upload
// ---------------------------------------------------------------------------

async function main() {
  let fileBytes;
  let fileSize;
  try {
    fileBytes = readFileSync(filePath);
    fileSize = statSync(filePath).size;
  } catch (e) {
    writeError(`Cannot read file: ${filePath} — ${e.message}`);
    process.exit(1);
  }

  let Indexer, MemData, ethers;
  try {
    const zgSdk = await import("@0gfoundation/0g-ts-sdk");
    Indexer = zgSdk.Indexer;
    MemData = zgSdk.MemData;
    ethers = await import("ethers");
  } catch (e) {
    writeError(
      `Cannot load @0gfoundation/0g-ts-sdk or ethers. ` +
      `Run: npm install @0gfoundation/0g-ts-sdk ethers  (in storage/ directory)\n${e.message}`
    );
    process.exit(2); // Exit code 2 = SDK not installed (Python wrapper treats as soft fail)
  }

  const provider = new ethers.JsonRpcProvider(RPC_URL);
  const signer = new ethers.Wallet(`0x${privateKey}`, provider);
  const indexer = new Indexer(INDEXER_RPC);

  console.error(`Uploading ${filePath} (${fileSize} bytes) to 0G Storage...`);

  const file = new MemData(fileBytes);

  let rootHash;
  try {
    const [hash, uploadErr] = await indexer.upload(file, RPC_URL, signer);
    if (uploadErr) throw new Error(String(uploadErr));
    rootHash = hash;
  } catch (e) {
    writeError(`Upload failed: ${e.message}`);
    process.exit(1);
  }

  const output = {
    ok: true,
    rootHash: rootHash,
    size: fileSize,
    file: filePath,
    rpc: RPC_URL,
    indexer: INDEXER_RPC,
  };
  process.stdout.write(JSON.stringify(output) + "\n");
}

function writeError(msg) {
  process.stderr.write(JSON.stringify({ ok: false, error: msg }) + "\n");
}

main().catch((e) => {
  writeError(`Unexpected error: ${e.message}\n${e.stack}`);
  process.exit(1);
});
