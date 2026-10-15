// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

interface IAgentRegistryB {
    function totalSupply() external view returns (uint256);
    function ownerOf(uint256 id) external view returns (address);
}

interface IVRFCoordinatorB {
    function requestRandomWords(uint32 numWords) external returns (uint256 requestId);
}

/// Shared blocklist judged by semantic-blind committees (Part II of docs/theory.md).
///
/// * Any registered agent may report another; a reporter with `maxStrikes` cleared reports is muted.
/// * A report opens a case; a VRF word draws `committeeSize` distinct agent ids uniformly from the registry
///   snapshot (excluding the accused), never reading cards.
/// * The owners of the drawn ids vote during `votingBlocks`; `quorumNum/quorumDen` of the seats voting "block"
///   blocks the accused network-wide; the same quorum voting "keep" clears it and strikes the reporter; otherwise the
///   case is inconclusive and penalises nobody.
/// * An agent is judged at most once per `epochBlocks` (no re-reporting until a committee convicts).
contract Blocklist {
    struct Case {
        uint256 accused;
        uint256 reporter;
        uint256 snapshot;
        uint64 openedAt;
        uint64 seatsDrawnAt;
        uint16 blockVotes;
        uint16 keepVotes;
        bool closed;
    }

    IAgentRegistryB public immutable registry;
    IVRFCoordinatorB public immutable coordinator;
    uint16 public immutable committeeSize;
    uint16 public immutable quorumNum;
    uint16 public immutable quorumDen;
    uint64 public immutable votingBlocks;
    uint64 public immutable epochBlocks;
    uint8 public immutable maxStrikes;

    mapping(uint256 => bool) public blocked;
    mapping(uint256 => uint64) public lastJudged;          // accused -> block of the last case
    mapping(uint256 => uint8) public strikes;              // reporter -> cleared reports
    mapping(uint256 => Case) public cases;                 // requestId -> case
    mapping(uint256 => uint256[]) private _seats;
    mapping(uint256 => mapping(uint256 => uint8)) public seatState;   // requestId -> agent id -> 1 seated, 2 voted

    event Reported(uint256 indexed requestId, uint256 indexed accused, uint256 indexed reporter);
    event CommitteeDrawn(uint256 indexed requestId, uint256[] seats);
    event Voted(uint256 indexed requestId, uint256 indexed seat, bool blockVote);
    event Closed(uint256 indexed requestId, uint256 indexed accused, bool blockedNow);

    constructor(IAgentRegistryB registry_, IVRFCoordinatorB coordinator_, uint16 committeeSize_, uint16 quorumNum_,
                uint16 quorumDen_, uint64 votingBlocks_, uint64 epochBlocks_, uint8 maxStrikes_) {
        registry = registry_;
        coordinator = coordinator_;
        committeeSize = committeeSize_;
        quorumNum = quorumNum_;
        quorumDen = quorumDen_;
        votingBlocks = votingBlocks_;
        epochBlocks = epochBlocks_;
        maxStrikes = maxStrikes_;
    }

    function report(uint256 reporterId, uint256 accused) external returns (uint256 requestId) {
        require(registry.ownerOf(reporterId) == msg.sender, "not owner");
        require(strikes[reporterId] < maxStrikes, "muted");
        require(!blocked[accused] && accused != reporterId, "bad accused");
        require(lastJudged[accused] == 0 || block.number >= lastJudged[accused] + epochBlocks, "judged this epoch");
        uint256 n = registry.totalSupply();
        require(n > committeeSize + 1, "registry too small");
        lastJudged[accused] = uint64(block.number);
        requestId = coordinator.requestRandomWords(1);
        cases[requestId] = Case(accused, reporterId, n, uint64(block.number), 0, 0, 0, false);
        emit Reported(requestId, accused, reporterId);
    }

    function fulfillRandomWords(uint256 requestId, uint256[] calldata words) external {
        require(msg.sender == address(coordinator), "only coordinator");
        Case storage c = cases[requestId];
        require(c.openedAt != 0 && c.seatsDrawnAt == 0, "unknown or drawn");
        c.seatsDrawnAt = uint64(block.number);
        uint256[] storage seats = _seats[requestId];
        uint256 ctr = 0;
        while (seats.length < committeeSize) {
            uint256 id = 1 + (uint256(keccak256(abi.encode(words[0], ctr))) % c.snapshot);
            ctr++;
            if (id == c.accused || seatState[requestId][id] != 0) continue;
            seatState[requestId][id] = 1;
            seats.push(id);
        }
        emit CommitteeDrawn(requestId, seats);
    }

    function vote(uint256 requestId, uint256 seatId, bool blockVote) external {
        Case storage c = cases[requestId];
        require(c.seatsDrawnAt != 0 && !c.closed, "not open");
        require(block.number <= c.seatsDrawnAt + votingBlocks, "voting over");
        require(seatState[requestId][seatId] == 1, "not a seat or voted");
        require(registry.ownerOf(seatId) == msg.sender, "not owner");
        seatState[requestId][seatId] = 2;
        if (blockVote) c.blockVotes++;
        else c.keepVotes++;
        emit Voted(requestId, seatId, blockVote);
    }

    /// Anyone may close a case once a quorum is reached either way or the voting window has passed.
    /// Two-sided verdict: a quorum of "block" votes blocks the accused; a quorum of "keep" votes clears it and
    /// strikes the reporter; anything else is inconclusive and penalises nobody.
    function close(uint256 requestId) external {
        Case storage c = cases[requestId];
        require(c.seatsDrawnAt != 0 && !c.closed, "not open");
        bool convict = uint256(c.blockVotes) * quorumDen >= uint256(quorumNum) * committeeSize;
        bool clear = uint256(c.keepVotes) * quorumDen >= uint256(quorumNum) * committeeSize;
        require(convict || clear || block.number > c.seatsDrawnAt + votingBlocks, "voting open");
        c.closed = true;
        if (convict) {
            blocked[c.accused] = true;
        } else if (clear && strikes[c.reporter] < 255) {
            strikes[c.reporter]++;
        }
        emit Closed(requestId, c.accused, convict);
    }

    function seatsOf(uint256 requestId) external view returns (uint256[] memory) {
        return _seats[requestId];
    }
}
