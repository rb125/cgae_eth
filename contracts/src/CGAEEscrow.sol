// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "./CGAERegistry.sol";

/**
 * @title CGAEEscrow
 * @notice Contract management and escrow for the CGAE economy.
 *         Implements tier-gated contract assignment with budget ceiling enforcement
 *         (Theorem 1: Bounded Economic Exposure).
 *
 * @dev All economic activity is mediated through formally specified contracts.
 *      Rewards are escrowed until verification. Penalties are enforced on failure.
 */
contract CGAEEscrow {

    // -----------------------------------------------------------------------
    // Types
    // -----------------------------------------------------------------------

    enum ContractStatus {
        Open,
        Assigned,
        Completed,
        Failed,
        Expired,
        Cancelled
    }

    struct EconomicContract {
        bytes32 contractId;
        address issuer;
        address assignedAgent;
        string objective;
        bytes32 constraintsHash;   // Hash of machine-verifiable constraint set (Phi)
        string verifierSpecHash;   // 0G Storage hash or pointer to verifier specification
        uint8 minTier;
        uint256 reward;
        uint256 penalty;
        uint64 deadline;
        uint64 createdAt;
        ContractStatus status;
        string domain;
    }

    // -----------------------------------------------------------------------
    // State
    // -----------------------------------------------------------------------

    CGAERegistry public registry;

    mapping(bytes32 => EconomicContract) public contracts;
    bytes32[] public contractIds;

    // Agent -> total active penalty exposure
    mapping(address => uint256) public activeExposure;

    // Economic accounting
    uint256 public totalRewardsPaid;
    uint256 public totalPenaltiesCollected;
    uint256 public totalEscrowed;

    address public admin;

    // -----------------------------------------------------------------------
    // Events
    // -----------------------------------------------------------------------

    event ContractCreated(bytes32 indexed contractId, uint8 minTier, uint256 reward, string domain);
    event ContractAssigned(bytes32 indexed contractId, address indexed agent);
    event ContractCompleted(bytes32 indexed contractId, address indexed agent, uint256 reward);
    event ContractFailed(bytes32 indexed contractId, address indexed agent, uint256 penalty);
    event ContractExpired(bytes32 indexed contractId);

    // -----------------------------------------------------------------------
    // Constructor
    // -----------------------------------------------------------------------

    constructor(address _registry) {
        registry = CGAERegistry(_registry);
        admin = msg.sender;
    }

    // -----------------------------------------------------------------------
    // Contract Lifecycle
    // -----------------------------------------------------------------------

    /**
     * @notice Create a new contract. Issuer deposits reward as escrow.
     */
    function createContract(
        string calldata objective,
        bytes32 constraintsHash,
        string calldata verifierSpecHash,
        uint8 minTier,
        uint256 penalty,
        uint64 deadline,
        string calldata domain
    ) external payable {
        require(msg.value > 0, "Must escrow reward");
        require(minTier > 0 && minTier <= 5, "Invalid tier");
        require(deadline > block.timestamp, "Deadline must be in future");
        require(constraintsHash != bytes32(0), "Missing constraints hash");
        require(bytes(verifierSpecHash).length > 0, "Missing verifier spec");

        bytes32 contractId = keccak256(abi.encodePacked(
            msg.sender, block.timestamp, objective, contractIds.length
        ));

        contracts[contractId] = EconomicContract({
            contractId: contractId,
            issuer: msg.sender,
            assignedAgent: address(0),
            objective: objective,
            constraintsHash: constraintsHash,
            verifierSpecHash: verifierSpecHash,
            minTier: minTier,
            reward: msg.value,
            penalty: penalty,
            deadline: deadline,
            createdAt: uint64(block.timestamp),
            status: ContractStatus.Open,
            domain: domain
        });

        contractIds.push(contractId);
        totalEscrowed += msg.value;

        emit ContractCreated(contractId, minTier, msg.value, domain);
    }

    /**
     * @notice Agent accepts a contract. Enforces:
     *         1. Agent tier >= min_tier
     *         2. Agent exposure + penalty <= budget ceiling (Theorem 1)
     */
    function acceptContract(bytes32 contractId) external payable {
        EconomicContract storage c = contracts[contractId];
        require(c.status == ContractStatus.Open, "Not open");
        require(block.timestamp < c.deadline, "Past deadline");

        // Tier check
        CGAERegistry.AgentRecord memory agent = registry.getAgent(msg.sender);
        require(agent.active, "Agent not active");
        require(agent.currentTier >= c.minTier, "Tier too low");

        // Budget ceiling check (Theorem 1: Bounded Economic Exposure)
        uint256 ceiling = registry.getBudgetCeiling(agent.currentTier);
        require(
            activeExposure[msg.sender] + c.penalty <= ceiling,
            "Would exceed budget ceiling"
        );

        // Agent must deposit penalty as collateral
        require(msg.value >= c.penalty, "Insufficient penalty collateral");

        c.assignedAgent = msg.sender;
        c.status = ContractStatus.Assigned;
        activeExposure[msg.sender] += c.penalty;

        emit ContractAssigned(contractId, msg.sender);
    }

    /**
     * @notice Mark a contract as completed. Called by admin/verifier after
     *         output verification. Releases reward to agent, returns collateral.
     */
    function completeContract(bytes32 contractId) external {
        require(msg.sender == admin, "Only admin/verifier");
        EconomicContract storage c = contracts[contractId];
        require(c.status == ContractStatus.Assigned, "Not assigned");

        c.status = ContractStatus.Completed;

        // Release exposure
        activeExposure[c.assignedAgent] -= c.penalty;

        // Pay reward to agent
        uint256 reward = c.reward;
        totalEscrowed -= reward;
        totalRewardsPaid += reward;

        // Return penalty collateral + pay reward
        uint256 payout = reward + c.penalty;
        payable(c.assignedAgent).transfer(payout);

        // Record on registry
        registry.recordContractOutcome(c.assignedAgent, true, reward);

        emit ContractCompleted(contractId, c.assignedAgent, reward);
    }

    /**
     * @notice Mark a contract as failed. Penalty is forfeited.
     */
    function failContract(bytes32 contractId) external {
        require(msg.sender == admin, "Only admin/verifier");
        EconomicContract storage c = contracts[contractId];
        require(c.status == ContractStatus.Assigned, "Not assigned");

        c.status = ContractStatus.Failed;

        // Release exposure
        activeExposure[c.assignedAgent] -= c.penalty;

        // Forfeit penalty collateral
        totalPenaltiesCollected += c.penalty;

        // Return escrowed reward to issuer
        totalEscrowed -= c.reward;
        payable(c.issuer).transfer(c.reward);

        // Record on registry
        registry.recordContractOutcome(c.assignedAgent, false, c.penalty);

        emit ContractFailed(contractId, c.assignedAgent, c.penalty);
    }

    /**
     * @notice Expire contracts past deadline. Anyone can call this.
     */
    function expireContract(bytes32 contractId) external {
        EconomicContract storage c = contracts[contractId];
        require(c.status == ContractStatus.Open, "Not open");
        require(block.timestamp >= c.deadline, "Not expired yet");

        c.status = ContractStatus.Expired;
        totalEscrowed -= c.reward;
        payable(c.issuer).transfer(c.reward);

        emit ContractExpired(contractId);
    }

    // -----------------------------------------------------------------------
    // Views
    // -----------------------------------------------------------------------

    function getContract(bytes32 contractId) external view returns (EconomicContract memory) {
        return contracts[contractId];
    }

    function getContractCount() external view returns (uint256) {
        return contractIds.length;
    }

    function getExposure(address agent) external view returns (uint256) {
        return activeExposure[agent];
    }

    function getEconomicsSummary() external view returns (
        uint256 _totalRewards,
        uint256 _totalPenalties,
        uint256 _totalEscrowed,
        uint256 _contractCount
    ) {
        return (totalRewardsPaid, totalPenaltiesCollected, totalEscrowed, contractIds.length);
    }

    // -----------------------------------------------------------------------
    // Admin
    // -----------------------------------------------------------------------

    function updateAdmin(address newAdmin) external {
        require(msg.sender == admin, "Only admin");
        admin = newAdmin;
    }

    receive() external payable {}
}
