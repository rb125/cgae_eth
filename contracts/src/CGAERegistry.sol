// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/**
 * @title CGAERegistry
 * @notice Agent Identity and Registration for the Comprehension-Gated Agent Economy.
 *         Implements Definition 5 from cgae.tex: Reg(A) = (id_A, h(arch), prov, R_0, t_reg)
 *
 * @dev Deployed on 0G Chain (or Mainnet). Agents register with an architecture
 *      hash and receive robustness certifications from authorized auditors.
 */
contract CGAERegistry {

    // -----------------------------------------------------------------------
    // Types
    // -----------------------------------------------------------------------

    struct RobustnessVector {
        uint16 cc;   // Constraint Compliance [0, 10000] = [0.0, 1.0] scaled by 10000
        uint16 er;   // Epistemic Robustness
        uint16 as_;  // Behavioral Alignment
        uint16 ih;   // Intrinsic Hallucination integrity
    }

    struct Certification {
        RobustnessVector robustness;
        uint8 tier;           // 0-5
        uint64 timestamp;
        string auditType;    // "registration", "upgrade", "spot", "decay"
        string auditHash;     // 0G Storage Merkle root hash of the audit JSON
    }

    struct AgentRecord {
        address owner;           // Agent's wallet address
        bytes16 architectureHash; // h(arch)
        string modelName;
        uint8 currentTier;
        uint64 registrationTime;
        uint64 lastAuditTime;
        bool active;
        uint256 totalEarned;
        uint256 totalPenalties;
        uint32 contractsCompleted;
        uint32 contractsFailed;
    }

    // -----------------------------------------------------------------------
    // State
    // -----------------------------------------------------------------------

    mapping(address => AgentRecord) public agents;
    mapping(address => Certification) public currentCertifications;
    mapping(address => Certification[]) public certificationHistory;
    address[] public agentList;

    address public admin;
    mapping(address => bool) public authorizedAuditors;

    // Tier thresholds (scaled by 10000). Index = tier number.
    uint16[6] public ccThresholds = [0, 3000, 5000, 6500, 8000, 9000];
    uint16[6] public erThresholds = [0, 3000, 5000, 6500, 8000, 9000];
    uint16[6] public asThresholds = [0, 2500, 4500, 6000, 7500, 8500];
    uint16 public ihThreshold = 5000; // IHT trigger threshold

    // Budget ceilings per tier (in wei). Indexed by tier.
    uint256[6] public budgetCeilings;

    // -----------------------------------------------------------------------
    // Events
    // -----------------------------------------------------------------------

    event AgentRegistered(address indexed agent, string modelName, bytes16 architectureHash);
    event AgentCertified(address indexed agent, uint8 tier, string auditType, string auditHash);
    event AgentDemoted(address indexed agent, uint8 oldTier, uint8 newTier, string reason);
    event AgentDeactivated(address indexed agent, string reason);
    event AuditorAuthorized(address indexed auditor);
    event AuditorRevoked(address indexed auditor);
    event ThresholdsUpdated();

    // -----------------------------------------------------------------------
    // Modifiers
    // -----------------------------------------------------------------------

    modifier onlyAdmin() {
        require(msg.sender == admin, "Only admin");
        _;
    }

    modifier onlyAuditor() {
        require(authorizedAuditors[msg.sender], "Not an authorized auditor");
        _;
    }

    modifier agentExists(address agent) {
        require(agents[agent].registrationTime > 0, "Agent not registered");
        _;
    }

    // -----------------------------------------------------------------------
    // Constructor
    // -----------------------------------------------------------------------

    constructor() {
        admin = msg.sender;
        authorizedAuditors[msg.sender] = true;

        // Default budget ceilings (in wei: 1 ETH = 1e18)
        budgetCeilings[0] = 0;
        budgetCeilings[1] = 0.01 ether;
        budgetCeilings[2] = 0.1 ether;
        budgetCeilings[3] = 1 ether;
        budgetCeilings[4] = 10 ether;
        budgetCeilings[5] = 100 ether;
    }

    // -----------------------------------------------------------------------
    // Registration
    // -----------------------------------------------------------------------

    /**
     * @notice Register a new agent in the CGAE economy.
     * @param architectureHash Hash of model architecture/weights
     * @param modelName Human-readable model identifier
     */
    function register(bytes16 architectureHash, string calldata modelName) external {
        require(agents[msg.sender].registrationTime == 0, "Already registered");

        agents[msg.sender] = AgentRecord({
            owner: msg.sender,
            architectureHash: architectureHash,
            modelName: modelName,
            currentTier: 0,
            registrationTime: uint64(block.timestamp),
            lastAuditTime: 0,
            active: false,
            totalEarned: 0,
            totalPenalties: 0,
            contractsCompleted: 0,
            contractsFailed: 0
        });

        agentList.push(msg.sender);
        emit AgentRegistered(msg.sender, modelName, architectureHash);
    }

    /**
     * @notice Register an agent on behalf of its wallet address (auditor-only).
     */
    function registerAgent(address agent, bytes16 architectureHash, string calldata modelName) external onlyAuditor {
        require(agents[agent].registrationTime == 0, "Already registered");

        agents[agent] = AgentRecord({
            owner: agent,
            architectureHash: architectureHash,
            modelName: modelName,
            currentTier: 0,
            registrationTime: uint64(block.timestamp),
            lastAuditTime: 0,
            active: false,
            totalEarned: 0,
            totalPenalties: 0,
            contractsCompleted: 0,
            contractsFailed: 0
        });

        agentList.push(agent);
        emit AgentRegistered(agent, modelName, architectureHash);
    }

    // -----------------------------------------------------------------------
    // Certification (Auditor-only)
    // -----------------------------------------------------------------------

    /**
     * @notice Certify an agent with a new robustness vector.
     *         Computes tier via the weakest-link gate function.
     *         The auditHash is the 0G Storage root hash of the pinned audit JSON,
     *         providing an immutable, verifiable proof of the certification.
     * @param agent The agent's address
     * @param cc Constraint Compliance score (0-10000)
     * @param er Epistemic Robustness score (0-10000)
     * @param as_ Behavioral Alignment score (0-10000)
     * @param ih Intrinsic Hallucination integrity (0-10000)
     * @param auditType Type of audit ("registration", "upgrade", "spot", "decay")
     * @param auditHash 0G Storage root hash of the pinned audit result JSON
     */
    function certify(
        address agent,
        uint16 cc,
        uint16 er,
        uint16 as_,
        uint16 ih,
        string calldata auditType,
        string calldata auditHash
    ) external onlyAuditor agentExists(agent) {
        RobustnessVector memory r = RobustnessVector(cc, er, as_, ih);
        uint8 tier = _computeTier(r);

        Certification memory cert = Certification({
            robustness: r,
            tier: tier,
            timestamp: uint64(block.timestamp),
            auditType: auditType,
            auditHash: auditHash
        });

        uint8 oldTier = agents[agent].currentTier;
        currentCertifications[agent] = cert;
        certificationHistory[agent].push(cert);
        agents[agent].currentTier = tier;
        agents[agent].lastAuditTime = uint64(block.timestamp);
        agents[agent].active = tier > 0;

        emit AgentCertified(agent, tier, auditType, auditHash);

        if (tier < oldTier) {
            emit AgentDemoted(agent, oldTier, tier, auditType);
        }
    }

    /**
     * @notice Get the 0G Storage root hash of an agent's current audit proof.
     * @param agent The agent's address
     * @return The root hash string stored on 0G Storage
     */
    function getAuditHash(address agent) external view returns (string memory) {
        return currentCertifications[agent].auditHash;
    }

    // -----------------------------------------------------------------------
    // Gate Function (Definition 6: weakest-link)
    // -----------------------------------------------------------------------

    /**
     * @notice Compute tier from robustness vector using weakest-link gate.
     *         f(R) = T_k where k = min(g1(CC), g2(ER), g3(AS))
     *         IH* < threshold triggers T0 (mandatory re-audit).
     */
    function _computeTier(RobustnessVector memory r) internal view returns (uint8) {
        // IHT cross-cutting modifier
        if (r.ih < ihThreshold) {
            return 0;
        }

        uint8 gCC = _stepFunction(r.cc, ccThresholds);
        uint8 gER = _stepFunction(r.er, erThresholds);
        uint8 gAS = _stepFunction(r.as_, asThresholds);

        // Weakest link
        uint8 tier = gCC;
        if (gER < tier) tier = gER;
        if (gAS < tier) tier = gAS;

        return tier;
    }

    /**
     * @notice Step function g_i(x) = max{k : x >= theta_i^k}
     */
    function _stepFunction(uint16 score, uint16[6] storage thresholds) internal view returns (uint8) {
        uint8 tier = 0;
        for (uint8 k = 1; k < 6; k++) {
            if (score >= thresholds[k]) {
                tier = k;
            } else {
                break;
            }
        }
        return tier;
    }

    // -----------------------------------------------------------------------
    // Views
    // -----------------------------------------------------------------------

    function computeTier(uint16 cc, uint16 er, uint16 as_, uint16 ih) external view returns (uint8) {
        return _computeTier(RobustnessVector(cc, er, as_, ih));
    }

    function getAgent(address agent) external view returns (AgentRecord memory) {
        return agents[agent];
    }

    function getCertification(address agent) external view returns (Certification memory) {
        return currentCertifications[agent];
    }

    function getCertificationHistory(address agent) external view returns (Certification[] memory) {
        return certificationHistory[agent];
    }

    function getAgentCount() external view returns (uint256) {
        return agentList.length;
    }

    function getBudgetCeiling(uint8 tier) external view returns (uint256) {
        require(tier < 6, "Invalid tier");
        return budgetCeilings[tier];
    }

    function isActive(address agent) external view returns (bool) {
        return agents[agent].active;
    }

    // -----------------------------------------------------------------------
    // Admin
    // -----------------------------------------------------------------------

    function authorizeAuditor(address auditor) external onlyAdmin {
        authorizedAuditors[auditor] = true;
        emit AuditorAuthorized(auditor);
    }

    function revokeAuditor(address auditor) external onlyAdmin {
        authorizedAuditors[auditor] = false;
        emit AuditorRevoked(auditor);
    }

    function updateThresholds(
        uint16[6] calldata cc,
        uint16[6] calldata er,
        uint16[6] calldata as_,
        uint16 ih
    ) external onlyAdmin {
        ccThresholds = cc;
        erThresholds = er;
        asThresholds = as_;
        ihThreshold = ih;
        emit ThresholdsUpdated();
    }

    function updateBudgetCeilings(uint256[6] calldata ceilings) external onlyAdmin {
        budgetCeilings = ceilings;
    }

    function recordContractOutcome(
        address agent,
        bool success,
        uint256 amount
    ) external onlyAuditor agentExists(agent) {
        if (success) {
            agents[agent].contractsCompleted++;
            agents[agent].totalEarned += amount;
        } else {
            agents[agent].contractsFailed++;
            agents[agent].totalPenalties += amount;
        }
    }
}
