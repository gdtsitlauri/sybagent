// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

/// Test registry for the blocklist: ids 1..totalSupply, each owned by `defaultOwner` unless registered otherwise,
/// so that one test account can cast the votes of every drawn seat.
contract MockOwnedRegistry {
    uint256 public totalSupply;
    address public defaultOwner;
    mapping(uint256 => address) private _owner;

    function setTotal(uint256 n, address owner_) external {
        totalSupply = n;
        defaultOwner = owner_;
    }

    function setOwner(uint256 id, address owner_) external {
        _owner[id] = owner_;
    }

    function ownerOf(uint256 id) external view returns (address) {
        address o = _owner[id];
        return o == address(0) ? defaultOwner : o;
    }
}
