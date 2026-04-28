require("@nomicfoundation/hardhat-toolbox");

/**
 * Hardhat configuration for CGAE smart contracts.
 *
 * Targets:
 *   zgTestnet  — 0G Chain Testnet (EVM-compatible)
 *   localhost  — Local Hardhat node for development
 *
 * Required env vars for 0G Testnet deployment:
 *   PRIVATE_KEY  — hex private key (no 0x prefix) of the deployer wallet
 *
 * Testnet resources:
 *   Token Faucet: https://faucet.0g.ai
 *   Explorer:     https://chainscan-galileo.0g.ai
 *   EVM RPC:      https://evmrpc-testnet.0g.ai
 *
 * Usage:
 *   cd contracts
 *   npm install
 *   export PRIVATE_KEY=<your_hex_key>
 *   npm run deploy:0g
 */

const PRIVATE_KEY = process.env.PRIVATE_KEY ||
  // Fallback zero key for compilation/testing only — never deploy with this
  "0000000000000000000000000000000000000000000000000000000000000001";

/** @type import('hardhat/config').HardhatUserConfig */
module.exports = {
  solidity: {
    version: "0.8.20",
    settings: {
      optimizer: {
        enabled: true,
        runs: 200,
      },
      viaIR: true,
    },
  },

  networks: {
    // 0G Chain Testnet (EVM-compatible)
    zgTestnet: {
      url: process.env.ZG_RPC_URL || "https://evmrpc-testnet.0g.ai",
      chainId: 16602,
      accounts: [`0x${PRIVATE_KEY}`],
      gas: 10_000_000,
      gasPrice: 2_500_000_000,
      timeout: 120_000,
    },

    // Local development
    localhost: {
      url: "http://127.0.0.1:8545",
      chainId: 31337,
    },
  },

  paths: {
    sources: "./src",
    artifacts: "./artifacts",
    cache: "./cache",
  },
};
