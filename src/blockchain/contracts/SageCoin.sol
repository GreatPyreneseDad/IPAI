// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "@openzeppelin/contracts/token/ERC20/ERC20.sol";
import "@openzeppelin/contracts/token/ERC20/extensions/ERC20Burnable.sol";
import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/security/ReentrancyGuard.sol";
import "./GCTCoherence.sol";
import "./IPAIIdentity.sol";

/**
 * @title SageCoin
 * @dev ERC20 token that rewards coherence and wisdom development
 */
contract SageCoin is ERC20, ERC20Burnable, Ownable, ReentrancyGuard {
    
    // Constants
    uint256 public constant BASIS_POINTS = 10000;
    uint256 public constant MAX_SUPPLY = 1000000000 * 10**18; // 1 billion tokens
    uint256 public constant DAILY_REWARD_CAP = 100 * 10**18;  // 100 SAGE per day max
    
    // Contract references
    GCTCoherence public gctContract;
    IPAIIdentity public identityContract;
    
    // Reward structures
    struct RewardTier {
        uint256 minCoherence;      // Minimum coherence score (basis points)
        uint256 baseReward;        // Base daily reward (tokens)
        uint256 wisdomMultiplier;  // Multiplier for wisdom component (basis points)
        uint256 consistencyBonus; // Bonus for consistency (basis points)
    }
    
    struct UserRewards {
        uint256 totalEarned;       // Total tokens earned
        uint256 lastClaimTime;     // Last reward claim timestamp
        uint256 streakDays;        // Consecutive days of coherence maintenance
        uint256 longestStreak;     // Longest streak achieved
        mapping(uint256 => bool) dailyClaimed; // Daily claim tracking
    }
    
    // Mappings
    mapping(address => UserRewards) public userRewards;
    mapping(address => bool) public rewardMinters;
    mapping(uint256 => RewardTier) public rewardTiers;
    
    // State variables
    uint256 public totalRewardsDistributed;
    uint256 public nextTierId = 0;
    bool public rewardsActive = true;
    uint256 public rewardStartTime;
    
    // Events
    event RewardClaimed(
        address indexed user,
        uint256 amount,
        uint256 coherenceScore,
        uint256 streakDays,
        uint256 timestamp
    );
    
    event StreakAchievement(
        address indexed user,
        uint256 streakDays,
        uint256 bonusReward
    );
    
    event RewardTierAdded(
        uint256 indexed tierId,
        uint256 minCoherence,
        uint256 baseReward
    );
    
    // Modifiers
    modifier onlyRewardMinter() {
        require(rewardMinters[msg.sender] || msg.sender == owner(), "Not authorized minter");
        _;
    }
    
    modifier hasIdentity() {
        require(identityContract.hasIdentity(msg.sender), "No IPAI identity");
        _;
    }
    
    modifier rewardsEnabled() {
        require(rewardsActive, "Rewards are disabled");
        _;
    }
    
    constructor(
        address _gctContract,
        address _identityContract
    ) ERC20("SageCoin", "SAGE") {
        gctContract = GCTCoherence(_gctContract);
        identityContract = IPAIIdentity(_identityContract);
        rewardStartTime = block.timestamp;
        
        // Initialize reward tiers
        _initializeRewardTiers();
        
        // Mint initial supply to contract owner
        _mint(msg.sender, 100000000 * 10**18); // 100 million initial tokens
        
        rewardMinters[msg.sender] = true;
    }
    
    /**
     * @dev Initialize reward tiers
     */
    function _initializeRewardTiers() internal {
        // Critical tier (0-0.2 coherence)
        rewardTiers[0] = RewardTier({
            minCoherence: 0,
            baseReward: 1 * 10**18,      // 1 SAGE
            wisdomMultiplier: 5000,      // 0.5x
            consistencyBonus: 1000       // 0.1x
        });
        
        // Low tier (0.2-0.4 coherence)
        rewardTiers[1] = RewardTier({
            minCoherence: 2000,
            baseReward: 5 * 10**18,      // 5 SAGE
            wisdomMultiplier: 7500,      // 0.75x
            consistencyBonus: 2000       // 0.2x
        });
        
        // Medium tier (0.4-0.7 coherence)
        rewardTiers[2] = RewardTier({
            minCoherence: 4000,
            baseReward: 15 * 10**18,     // 15 SAGE
            wisdomMultiplier: 10000,     // 1.0x
            consistencyBonus: 3000       // 0.3x
        });
        
        // High tier (0.7+ coherence)
        rewardTiers[3] = RewardTier({
            minCoherence: 7000,
            baseReward: 30 * 10**18,     // 30 SAGE
            wisdomMultiplier: 15000,     // 1.5x
            consistencyBonus: 5000       // 0.5x
        });
        
        nextTierId = 4;
    }
    
    /**
     * @dev Claim daily rewards based on coherence
     */
    function claimDailyReward() 
        external 
        nonReentrant 
        hasIdentity 
        rewardsEnabled 
    {
        require(_canClaimToday(msg.sender), "Already claimed today or too early");
        
        // Get latest coherence data
        GCTCoherence.CoherenceData memory coherenceData = gctContract.getLatestCoherence(msg.sender);
        
        // Calculate reward
        uint256 rewardAmount = _calculateReward(msg.sender, coherenceData);
        
        // Update user rewards
        UserRewards storage userReward = userRewards[msg.sender];
        userReward.totalEarned += rewardAmount;
        userReward.lastClaimTime = block.timestamp;
        
        // Update streak
        _updateStreak(msg.sender);
        
        // Mark daily claim
        uint256 dayIndex = (block.timestamp - rewardStartTime) / 86400;
        userReward.dailyClaimed[dayIndex] = true;
        
        // Mint reward tokens
        require(totalSupply() + rewardAmount <= MAX_SUPPLY, "Max supply exceeded");
        _mint(msg.sender, rewardAmount);
        
        totalRewardsDistributed += rewardAmount;
        
        emit RewardClaimed(
            msg.sender,
            rewardAmount,
            coherenceData.coherenceScore,
            userReward.streakDays,
            block.timestamp
        );
        
        // Check for streak achievements
        _checkStreakAchievements(msg.sender);
    }
    
    /**
     * @dev Calculate reward amount based on coherence and user history
     */
    function _calculateReward(
        address user,
        GCTCoherence.CoherenceData memory coherenceData
    ) internal view returns (uint256) {
        uint256 coherenceScore = coherenceData.coherenceScore;
        
        // Find appropriate reward tier
        RewardTier memory tier = _getRewardTier(coherenceScore);
        
        // Base reward
        uint256 reward = tier.baseReward;
        
        // Wisdom bonus (based on rho component)
        uint256 wisdomBonus = (reward * coherenceData.rho * tier.wisdomMultiplier) / (BASIS_POINTS * BASIS_POINTS);
        
        // Consistency bonus (based on psi component)
        uint256 consistencyBonus = (reward * coherenceData.psi * tier.consistencyBonus) / (BASIS_POINTS * BASIS_POINTS);
        
        // Streak multiplier
        uint256 streakMultiplier = _getStreakMultiplier(userRewards[user].streakDays);
        
        // Calculate total reward
        uint256 totalReward = reward + wisdomBonus + consistencyBonus;
        totalReward = (totalReward * streakMultiplier) / BASIS_POINTS;
        
        // Apply daily cap
        return totalReward > DAILY_REWARD_CAP ? DAILY_REWARD_CAP : totalReward;
    }
    
    /**
     * @dev Get reward tier for coherence score
     */
    function _getRewardTier(uint256 coherenceScore) internal view returns (RewardTier memory) {
        // Find highest tier that user qualifies for
        RewardTier memory selectedTier = rewardTiers[0];
        
        for (uint256 i = 0; i < nextTierId; i++) {
            if (coherenceScore >= rewardTiers[i].minCoherence) {
                selectedTier = rewardTiers[i];
            }
        }
        
        return selectedTier;
    }
    
    /**
     * @dev Update user's streak
     */
    function _updateStreak(address user) internal {
        UserRewards storage userReward = userRewards[user];
        uint256 currentDay = (block.timestamp - rewardStartTime) / 86400;
        uint256 previousDay = currentDay - 1;
        
        // Check if claimed yesterday
        if (userReward.dailyClaimed[previousDay] || userReward.streakDays == 0) {
            userReward.streakDays++;
            
            // Update longest streak
            if (userReward.streakDays > userReward.longestStreak) {
                userReward.longestStreak = userReward.streakDays;
            }
        } else {
            // Reset streak
            userReward.streakDays = 1;
        }
    }
    
    /**
     * @dev Get streak multiplier
     */
    function _getStreakMultiplier(uint256 streakDays) internal pure returns (uint256) {
        if (streakDays >= 365) return 20000; // 2.0x for 1 year
        if (streakDays >= 180) return 17500; // 1.75x for 6 months
        if (streakDays >= 90) return 15000;  // 1.5x for 3 months
        if (streakDays >= 30) return 12500;  // 1.25x for 1 month
        if (streakDays >= 7) return 11000;   // 1.1x for 1 week
        return 10000; // 1.0x base
    }
    
    /**
     * @dev Check for streak achievements
     */
    function _checkStreakAchievements(address user) internal {
        uint256 streakDays = userRewards[user].streakDays;
        uint256 bonusReward = 0;
        
        // Award bonus for milestone streaks
        if (streakDays == 7) {
            bonusReward = 10 * 10**18; // 10 SAGE for 1 week
        } else if (streakDays == 30) {
            bonusReward = 50 * 10**18; // 50 SAGE for 1 month
        } else if (streakDays == 90) {
            bonusReward = 200 * 10**18; // 200 SAGE for 3 months
        } else if (streakDays == 365) {
            bonusReward = 1000 * 10**18; // 1000 SAGE for 1 year
        }
        
        if (bonusReward > 0) {
            _mint(user, bonusReward);
            userRewards[user].totalEarned += bonusReward;
            totalRewardsDistributed += bonusReward;
            
            emit StreakAchievement(user, streakDays, bonusReward);
        }
    }
    
    /**
     * @dev Check if user can claim today
     */
    function _canClaimToday(address user) internal view returns (bool) {
        uint256 currentDay = (block.timestamp - rewardStartTime) / 86400;
        return !userRewards[user].dailyClaimed[currentDay];
    }
    
    /**
     * @dev Get user's claimable reward amount
     */
    function getClaimableReward(address user) external view returns (uint256) {
        if (!_canClaimToday(user)) {
            return 0;
        }
        
        try gctContract.getLatestCoherence(user) returns (GCTCoherence.CoherenceData memory coherenceData) {
            return _calculateReward(user, coherenceData);
        } catch {
            return 0;
        }
    }
    
    /**
     * @dev Get user reward information
     */
    function getUserRewardInfo(address user) 
        external 
        view 
        returns (
            uint256 totalEarned,
            uint256 streakDays,
            uint256 longestStreak,
            bool canClaimToday,
            uint256 claimableAmount
        ) 
    {
        UserRewards storage userReward = userRewards[user];
        return (
            userReward.totalEarned,
            userReward.streakDays,
            userReward.longestStreak,
            _canClaimToday(user),
            this.getClaimableReward(user)
        );
    }
    
    /**
     * @dev Add new reward tier (only owner)
     */
    function addRewardTier(
        uint256 _minCoherence,
        uint256 _baseReward,
        uint256 _wisdomMultiplier,
        uint256 _consistencyBonus
    ) external onlyOwner {
        rewardTiers[nextTierId] = RewardTier({
            minCoherence: _minCoherence,
            baseReward: _baseReward,
            wisdomMultiplier: _wisdomMultiplier,
            consistencyBonus: _consistencyBonus
        });
        
        emit RewardTierAdded(nextTierId, _minCoherence, _baseReward);
        nextTierId++;
    }
    
    /**
     * @dev Emergency mint (only owner)
     */
    function emergencyMint(address to, uint256 amount) external onlyOwner {
        require(totalSupply() + amount <= MAX_SUPPLY, "Max supply exceeded");
        _mint(to, amount);
    }
    
    /**
     * @dev Toggle rewards (only owner)
     */
    function toggleRewards() external onlyOwner {
        rewardsActive = !rewardsActive;
    }
    
    /**
     * @dev Add reward minter (only owner)
     */
    function addRewardMinter(address minter) external onlyOwner {
        rewardMinters[minter] = true;
    }
    
    /**
     * @dev Remove reward minter (only owner)
     */
    function removeRewardMinter(address minter) external onlyOwner {
        rewardMinters[minter] = false;
    }
    
    /**
     * @dev Get contract version
     */
    function version() external pure returns (string memory) {
        return "1.0.0";
    }
}