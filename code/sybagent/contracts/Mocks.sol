// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

/// Test registry: sequential agent ids, as minted by an ERC-8004 identity registry. `setTotal` creates ids in bulk
/// (owned by nobody) so that gas can be measured at large registry sizes.
contract MockAgentRegistry {
    uint256 public totalSupply;
    mapping(uint256 => address) private _owner;

    function setTotal(uint256 n) external {
        totalSupply = n;
    }

    function register() external returns (uint256 id) {
        id = ++totalSupply;
        _owner[id] = msg.sender;
    }

    function ownerOf(uint256 id) external view returns (address) {
        return _owner[id];
    }
}

interface IVRFConsumer {
    function fulfillRandomWords(uint256 requestId, uint256[] calldata words) external;
}

/// Consumer that does nothing: measures the mock coordinator's own share of a fulfillment's gas.
contract NoopConsumer {
    function fulfillRandomWords(uint256, uint256[] calldata) external {}
}

/// Test coordinator. A production coordinator verifies a VRF proof before calling back; here the test harness
/// supplies the random word, so that gas of the consumer callback can be measured in isolation.
contract MockVRFCoordinator {
    uint256 public nextId = 1;
    mapping(uint256 => address) public consumerOf;

    function requestRandomWords(uint32) external returns (uint256 id) {
        id = nextId++;
        consumerOf[id] = msg.sender;
    }

    function fulfill(uint256 id, uint256 word) external {
        fulfillTo(consumerOf[id], id, word);
    }

    /// Same call path to an arbitrary consumer (used with NoopConsumer to measure the mock's own overhead).
    function fulfillTo(address consumer, uint256 id, uint256 word) public {
        uint256[] memory w = new uint256[](1);
        w[0] = word;
        IVRFConsumer(consumer).fulfillRandomWords(id, w);
    }
}
