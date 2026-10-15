// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

/// Minimal view of an ERC-8004 identity registry (ERC-721 with enumerable supply): agent ids are minted
/// sequentially from 1.
interface IAgentRegistry {
    function totalSupply() external view returns (uint256);
    function ownerOf(uint256 id) external view returns (address);
}

/// VRF coordinator interface (request/callback pattern, as in Chainlink VRF).
interface IVRFCoordinator {
    function requestRandomWords(uint32 numWords) external returns (uint256 requestId);
}

/// Semantic-blind anchor sampling on-chain.
///
/// An agent asks for `a` anchors. The contract snapshots the registry size at request time and draws `a` distinct
/// agent ids uniformly from [1, N_snapshot] (excluding the requester) from a VRF output that arrives in a later
/// transaction. Hence:
///   * descriptions (agent cards) are never read: selection is semantic-blind;
///   * identities registered after the request cannot be drawn (no front-running with fresh Sybils);
///   * nobody can predict or choose the draw (VRF uniqueness), which is assumption A1 of Theorem 1;
///   * an agent has at most one pending request, so an oracle that withholds an unwanted output gains no second
///     draw (no re-roll by re-requesting).
contract AnchorSampler {
    struct Request {
        uint256 requesterId;
        uint256 snapshot;     // registry size when the request was made
        uint8 a;              // anchors requested
        bool done;
    }

    IAgentRegistry public immutable registry;
    IVRFCoordinator public immutable coordinator;
    uint8 public constant MAX_ANCHORS = 16;

    mapping(uint256 => Request) public requests;
    mapping(uint256 => uint256) public pendingOf;      // requesterId -> open requestId (0 if none)
    mapping(uint256 => uint256[]) private _anchors;

    event AnchorsRequested(uint256 indexed requestId, uint256 indexed requesterId, uint256 snapshot, uint8 a);
    event AnchorsAssigned(uint256 indexed requestId, uint256 indexed requesterId, uint256[] anchors);

    constructor(IAgentRegistry registry_, IVRFCoordinator coordinator_) {
        registry = registry_;
        coordinator = coordinator_;
    }

    function requestAnchors(uint256 requesterId, uint8 a) external returns (uint256 requestId) {
        require(registry.ownerOf(requesterId) == msg.sender, "not owner");
        require(pendingOf[requesterId] == 0, "request pending");
        uint256 n = registry.totalSupply();
        require(a > 0 && a <= MAX_ANCHORS, "bad a");
        require(n > a, "registry too small");
        requestId = coordinator.requestRandomWords(1);
        requests[requestId] = Request(requesterId, n, a, false);
        pendingOf[requesterId] = requestId;
        emit AnchorsRequested(requestId, requesterId, n, a);
    }

    /// Called by the coordinator with the verified random word.
    function fulfillRandomWords(uint256 requestId, uint256[] calldata words) external {
        require(msg.sender == address(coordinator), "only coordinator");
        Request storage r = requests[requestId];
        require(!r.done && r.a > 0, "unknown or done");
        r.done = true;
        pendingOf[r.requesterId] = 0;
        uint256[] memory out = _draw(words[0], r.snapshot, r.a, r.requesterId);
        _anchors[requestId] = out;
        emit AnchorsAssigned(requestId, r.requesterId, out);
    }

    function anchorsOf(uint256 requestId) external view returns (uint256[] memory) {
        return _anchors[requestId];
    }

    /// `a` distinct ids from [1, n] other than `skip`, by rejection sampling; the modulo bias is at most n / 2^256.
    function _draw(uint256 seed, uint256 n, uint8 a, uint256 skip) internal pure returns (uint256[] memory out) {
        out = new uint256[](a);
        uint256 got = 0;
        uint256 ctr = 0;
        while (got < a) {
            uint256 id = 1 + (uint256(keccak256(abi.encode(seed, ctr))) % n);
            ctr++;
            if (id == skip) continue;
            bool dup = false;
            for (uint256 j = 0; j < got; j++) {
                if (out[j] == id) { dup = true; break; }
            }
            if (dup) continue;
            out[got++] = id;
        }
    }
}

/// Baseline that is NOT safe: seeds the draw with the previous block hash in the same transaction. Whoever
/// produces that block can compute the outcome first and try other block contents until it likes the result.
contract BlockhashAnchorSampler {
    IAgentRegistry public immutable registry;

    event AnchorsAssigned(uint256 indexed requesterId, uint256[] anchors);

    constructor(IAgentRegistry registry_) {
        registry = registry_;
    }

    function sampleAnchors(uint256 requesterId, uint8 a) external returns (uint256[] memory out) {
        require(registry.ownerOf(requesterId) == msg.sender, "not owner");
        uint256 n = registry.totalSupply();
        require(a > 0 && n > a, "bad a");
        uint256 seed = uint256(keccak256(abi.encode(blockhash(block.number - 1), requesterId)));
        out = new uint256[](a);
        uint256 got = 0;
        uint256 ctr = 0;
        while (got < a) {
            uint256 id = 1 + (uint256(keccak256(abi.encode(seed, ctr))) % n);
            ctr++;
            if (id == requesterId) continue;
            bool dup = false;
            for (uint256 j = 0; j < got; j++) {
                if (out[j] == id) { dup = true; break; }
            }
            if (dup) continue;
            out[got++] = id;
        }
        emit AnchorsAssigned(requesterId, out);
    }
}
