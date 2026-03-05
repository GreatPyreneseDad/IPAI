// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/security/ReentrancyGuard.sol";
import "@openzeppelin/contracts/utils/math/SafeMath.sol";

/**
 * @title GCTCoherence
 * @dev Smart contract for storing and managing GCT coherence data on blockchain
 */
contract GCTCoherence is Ownable, ReentrancyGuard {
    using SafeMath for uint256;
    
    // Constants
    uint256 public constant BASIS_POINTS = 10000;
    uint256 public constant MAX_HISTORY = 100;
    uint256 public constant MIN_UPDATE_INTERVAL = 1 hours;
    
    // Coherence data structure
    struct CoherenceData {
        uint256 psi;              // Internal consistency (basis points)
        uint256 rho;              // Accumulated wisdom (basis points)
        uint256 q;                // Moral activation energy (basis points)
        uint256 f;                // Social belonging (basis points)
        uint256 coherenceScore;   // Overall coherence score (basis points)
        uint256 soulEcho;         // Soul echo metric (basis points)
        uint256 timestamp;        // Block timestamp
        uint256 blockNumber;      // Block number
        bytes32 dataHash;         // Hash of the data for integrity
    }
    
    // Individual parameters structure
    struct UserParameters {
        uint256 km;               // Activation threshold (basis points)
        uint256 ki;               // Sustainability threshold (basis points)
        bool calibrated;          // Whether parameters are calibrated
        uint256 calibrationTime;  // Timestamp of calibration
        uint256 calibrationBlock; // Block number of calibration
    }
    
    // Coherence level enumeration
    enum CoherenceLevel { CRITICAL, LOW, MEDIUM, HIGH }
    
    // Mappings
    mapping(address => CoherenceData[]) public userCoherenceHistory;
    mapping(address => UserParameters) public userParameters;
    mapping(address => uint256) public coherenceUpdateCount;
    mapping(address => uint256) public lastUpdateTime;
    mapping(address => bool) public authorizedUpdaters;
    
    // Events
    event CoherenceUpdated(
        address indexed user,
        uint256 coherenceScore,
        uint256 soulEcho,
        CoherenceLevel level,
        uint256 timestamp
    );
    
    event ParametersCalibrated(
        address indexed user,
        uint256 km,
        uint256 ki,
        uint256 timestamp
    );
    
    event AuthorizedUpdaterAdded(address indexed updater);
    event AuthorizedUpdaterRemoved(address indexed updater);
    
    // Modifiers
    modifier onlyAuthorizedUpdater() {
        require(
            authorizedUpdaters[msg.sender] || msg.sender == owner(),
            "Not authorized to update coherence data"
        );
        _;
    }
    
    modifier validCoherenceData(
        uint256 _psi,
        uint256 _rho,
        uint256 _q,
        uint256 _f,
        uint256 _coherenceScore
    ) {
        require(_psi <= BASIS_POINTS, "Invalid psi value");
        require(_rho <= BASIS_POINTS, "Invalid rho value");
        require(_q <= BASIS_POINTS, "Invalid q value");
        require(_f <= BASIS_POINTS, "Invalid f value");
        require(_coherenceScore <= BASIS_POINTS.mul(4), "Invalid coherence score");
        _;
    }
    
    modifier respectsUpdateInterval(address user) {
        require(
            block.timestamp >= lastUpdateTime[user].add(MIN_UPDATE_INTERVAL),
            "Update too frequent"
        );
        _;
    }
    
    constructor() {
        authorizedUpdaters[msg.sender] = true;
    }
    
    /**
     * @dev Update coherence data for a user
     */
    function updateCoherence(
        address _user,
        uint256 _psi,
        uint256 _rho,
        uint256 _q,
        uint256 _f,
        uint256 _coherenceScore
    ) 
        external 
        nonReentrant 
        onlyAuthorizedUpdater 
        validCoherenceData(_psi, _rho, _q, _f, _coherenceScore)
        respectsUpdateInterval(_user)
    {
        // Calculate soul echo on-chain
        uint256 soulEcho = _psi.mul(_rho).mul(_q).mul(_f).div(BASIS_POINTS.mul(BASIS_POINTS).mul(BASIS_POINTS));
        
        // Create data hash for integrity
        bytes32 dataHash = keccak256(abi.encodePacked(
            _user, _psi, _rho, _q, _f, _coherenceScore, block.timestamp, block.number
        ));
        
        // Create new coherence entry
        CoherenceData memory newData = CoherenceData({
            psi: _psi,
            rho: _rho,
            q: _q,
            f: _f,
            coherenceScore: _coherenceScore,
            soulEcho: soulEcho,
            timestamp: block.timestamp,
            blockNumber: block.number,
            dataHash: dataHash
        });
        
        // Add to history
        userCoherenceHistory[_user].push(newData);
        coherenceUpdateCount[_user] = coherenceUpdateCount[_user].add(1);
        lastUpdateTime[_user] = block.timestamp;
        
        // Limit history size for gas efficiency
        if (userCoherenceHistory[_user].length > MAX_HISTORY) {
            // Remove oldest entry
            for (uint i = 0; i < userCoherenceHistory[_user].length - 1; i++) {
                userCoherenceHistory[_user][i] = userCoherenceHistory[_user][i + 1];
            }
            userCoherenceHistory[_user].pop();
        }
        
        // Determine coherence level
        CoherenceLevel level = _getCoherenceLevel(_coherenceScore);
        
        emit CoherenceUpdated(_user, _coherenceScore, soulEcho, level, block.timestamp);
    }
    
    /**
     * @dev Update coherence data for the caller
     */
    function updateMyCoherence(
        uint256 _psi,
        uint256 _rho,
        uint256 _q,
        uint256 _f,
        uint256 _coherenceScore
    ) 
        external 
        validCoherenceData(_psi, _rho, _q, _f, _coherenceScore)
        respectsUpdateInterval(msg.sender)
    {
        this.updateCoherence(msg.sender, _psi, _rho, _q, _f, _coherenceScore);
    }
    
    /**
     * @dev Calibrate individual parameters
     */
    function calibrateParameters(uint256 _km, uint256 _ki) external {
        require(_km >= 1000 && _km <= 5000, "km out of range [0.1, 0.5]");
        require(_ki >= 5000 && _ki <= 20000, "ki out of range [0.5, 2.0]");
        
        userParameters[msg.sender] = UserParameters({
            km: _km,
            ki: _ki,
            calibrated: true,
            calibrationTime: block.timestamp,
            calibrationBlock: block.number
        });
        
        emit ParametersCalibrated(msg.sender, _km, _ki, block.timestamp);
    }
    
    /**
     * @dev Get latest coherence data for a user
     */
    function getLatestCoherence(address user) 
        external 
        view 
        returns (CoherenceData memory) 
    {
        require(userCoherenceHistory[user].length > 0, "No coherence data");
        return userCoherenceHistory[user][userCoherenceHistory[user].length - 1];
    }
    
    /**
     * @dev Get coherence trajectory (last N entries)
     */
    function getCoherenceTrajectory(address user, uint256 count) 
        external 
        view 
        returns (CoherenceData[] memory) 
    {
        uint256 historyLength = userCoherenceHistory[user].length;
        uint256 returnCount = count > historyLength ? historyLength : count;
        
        CoherenceData[] memory trajectory = new CoherenceData[](returnCount);
        
        for (uint256 i = 0; i < returnCount; i++) {
            trajectory[i] = userCoherenceHistory[user][historyLength - returnCount + i];
        }
        
        return trajectory;
    }
    
    /**
     * @dev Calculate coherence velocity (rate of change)
     */
    function getCoherenceVelocity(address user) 
        external 
        view 
        returns (int256 velocity) 
    {
        uint256 historyLength = userCoherenceHistory[user].length;
        require(historyLength >= 2, "Insufficient history");
        
        CoherenceData memory latest = userCoherenceHistory[user][historyLength - 1];
        CoherenceData memory previous = userCoherenceHistory[user][historyLength - 2];
        
        int256 deltaCoherence = int256(latest.coherenceScore) - int256(previous.coherenceScore);
        uint256 deltaTime = latest.timestamp - previous.timestamp;
        
        // Velocity in basis points per day
        velocity = (deltaCoherence * 86400) / int256(deltaTime);
    }
    
    /**
     * @dev Get coherence statistics for a user
     */
    function getCoherenceStatistics(address user) 
        external 
        view 
        returns (
            uint256 count,
            uint256 averageCoherence,
            uint256 maxCoherence,
            uint256 minCoherence,
            uint256 latestCoherence
        ) 
    {
        uint256 historyLength = userCoherenceHistory[user].length;
        require(historyLength > 0, "No coherence data");
        
        count = historyLength;
        
        uint256 sum = 0;
        uint256 max = 0;
        uint256 min = type(uint256).max;
        
        for (uint256 i = 0; i < historyLength; i++) {
            uint256 score = userCoherenceHistory[user][i].coherenceScore;
            sum = sum.add(score);
            
            if (score > max) {
                max = score;
            }
            if (score < min) {
                min = score;
            }
        }
        
        averageCoherence = sum.div(historyLength);
        maxCoherence = max;
        minCoherence = min;
        latestCoherence = userCoherenceHistory[user][historyLength - 1].coherenceScore;
    }
    
    /**
     * @dev Verify data integrity using hash
     */
    function verifyDataIntegrity(address user, uint256 index) 
        external 
        view 
        returns (bool) 
    {
        require(index < userCoherenceHistory[user].length, "Index out of bounds");
        
        CoherenceData memory data = userCoherenceHistory[user][index];
        bytes32 expectedHash = keccak256(abi.encodePacked(
            user, data.psi, data.rho, data.q, data.f, 
            data.coherenceScore, data.timestamp, data.blockNumber
        ));
        
        return data.dataHash == expectedHash;
    }
    
    /**
     * @dev Batch verify data integrity
     */
    function batchVerifyIntegrity(address user, uint256 startIndex, uint256 endIndex) 
        external 
        view 
        returns (bool[] memory) 
    {
        require(startIndex <= endIndex, "Invalid range");
        require(endIndex < userCoherenceHistory[user].length, "End index out of bounds");
        
        uint256 length = endIndex - startIndex + 1;
        bool[] memory results = new bool[](length);
        
        for (uint256 i = 0; i < length; i++) {
            uint256 index = startIndex + i;
            CoherenceData memory data = userCoherenceHistory[user][index];
            bytes32 expectedHash = keccak256(abi.encodePacked(
                user, data.psi, data.rho, data.q, data.f, 
                data.coherenceScore, data.timestamp, data.blockNumber
            ));
            results[i] = (data.dataHash == expectedHash);
        }
        
        return results;
    }
    
    /**
     * @dev Add authorized updater (only owner)
     */
    function addAuthorizedUpdater(address updater) external onlyOwner {
        authorizedUpdaters[updater] = true;
        emit AuthorizedUpdaterAdded(updater);
    }
    
    /**
     * @dev Remove authorized updater (only owner)
     */
    function removeAuthorizedUpdater(address updater) external onlyOwner {
        authorizedUpdaters[updater] = false;
        emit AuthorizedUpdaterRemoved(updater);
    }
    
    /**
     * @dev Get coherence level from score
     */
    function _getCoherenceLevel(uint256 score) internal pure returns (CoherenceLevel) {
        if (score >= 7000) {  // >= 0.7
            return CoherenceLevel.HIGH;
        } else if (score >= 4000) {  // >= 0.4
            return CoherenceLevel.MEDIUM;
        } else if (score >= 2000) {  // >= 0.2
            return CoherenceLevel.LOW;
        } else {
            return CoherenceLevel.CRITICAL;
        }
    }
    
    /**
     * @dev Emergency pause functionality (only owner)
     */
    bool public paused = false;
    
    modifier whenNotPaused() {
        require(!paused, "Contract is paused");
        _;
    }
    
    function pause() external onlyOwner {
        paused = true;
    }
    
    function unpause() external onlyOwner {
        paused = false;
    }
    
    /**
     * @dev Get contract version
     */
    function version() external pure returns (string memory) {
        return "1.0.0";
    }
}