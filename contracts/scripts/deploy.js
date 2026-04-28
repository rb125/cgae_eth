/**
 * CGAE Deployment Script — 0G Chain Testnet
 * ==========================================
 * Deploys CGAERegistry and CGAEEscrow to 0G Chain and writes
 * the resulting contract addresses to deployed.json.
 *
 * Usage:
 *   cd contracts
 *   npm install
 *   export PRIVATE_KEY=<hex_private_key_no_0x>
 *   npm run deploy:0g
 */

const { ethers } = require("hardhat");
const fs = require("fs");
const path = require("path");

async function main() {
  const [deployer] = await ethers.getSigners();
  const network = await ethers.provider.getNetwork();
  const chainId = Number(network.chainId);

  console.log("=".repeat(60));
  console.log("CGAE Contract Deployment — 0G Chain");
  console.log("=".repeat(60));
  console.log(`Network:  ${network.name} (chain ${chainId})`);
  console.log(`Deployer: ${deployer.address}`);

  const balance = await ethers.provider.getBalance(deployer.address);
  console.log(`Balance:  ${ethers.formatEther(balance)} A0GI\n`);

  if (balance === 0n) {
    console.error("ERROR: Deployer wallet has 0 tokens.");
    console.error("Get testnet tokens from: https://faucet.0g.ai");
    process.exit(1);
  }

  // Deploy CGAERegistry
  console.log("Deploying CGAERegistry...");
  const RegistryFactory = await ethers.getContractFactory("CGAERegistry");
  const registry = await RegistryFactory.deploy();
  await registry.waitForDeployment();
  const registryAddress = await registry.getAddress();
  console.log(`  CGAERegistry deployed to: ${registryAddress}`);

  // Deploy CGAEEscrow
  console.log("Deploying CGAEEscrow...");
  const EscrowFactory = await ethers.getContractFactory("CGAEEscrow");
  const escrow = await EscrowFactory.deploy(registryAddress);
  await escrow.waitForDeployment();
  const escrowAddress = await escrow.getAddress();
  console.log(`  CGAEEscrow deployed to:   ${escrowAddress}`);

  // Authorize escrow as auditor
  console.log("Authorizing CGAEEscrow as auditor in CGAERegistry...");
  const authTx = await registry.authorizeAuditor(escrowAddress);
  await authTx.wait();
  console.log(`  Authorized (tx: ${authTx.hash})`);

  // Write deployment manifest
  const explorer = chainId === 16602
    ? "https://chainscan-galileo.0g.ai"
    : "http://localhost";

  const manifest = {
    network: network.name,
    chainId,
    deployedAt: new Date().toISOString(),
    deployer: deployer.address,
    contracts: {
      CGAERegistry: {
        address: registryAddress,
        deployTx: registry.deploymentTransaction()?.hash || null,
      },
      CGAEEscrow: {
        address: escrowAddress,
        deployTx: escrow.deploymentTransaction()?.hash || null,
      },
    },
    rpc: process.env.ZG_RPC_URL || "https://evmrpc-testnet.0g.ai",
    explorer,
    storage: {
      indexer: "https://indexer-storage-testnet-turbo.0g.ai",
      scan: "https://storagescan-galileo.0g.ai",
    },
  };

  const deployedPath = path.join(__dirname, "..", "deployed.json");
  fs.writeFileSync(deployedPath, JSON.stringify(manifest, null, 2));
  console.log(`\nDeployment manifest written to: deployed.json`);

  console.log("\n" + "=".repeat(60));
  console.log("Deployment complete!");
  console.log("=".repeat(60));
  console.log(`CGAERegistry : ${explorer}/address/${registryAddress}`);
  console.log(`CGAEEscrow   : ${explorer}/address/${escrowAddress}`);
  console.log("\nNext steps:");
  console.log(`  1. CGAE_REGISTRY_ADDRESS=${registryAddress}`);
  console.log(`     CGAE_ESCROW_ADDRESS=${escrowAddress}`);
  console.log("  2. python -m server.live_runner");
  console.log("=".repeat(60));
}

main()
  .then(() => process.exit(0))
  .catch((err) => {
    console.error("Deployment failed:", err);
    process.exit(1);
  });
