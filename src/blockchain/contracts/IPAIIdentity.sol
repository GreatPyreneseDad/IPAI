// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "@openzeppelin/contracts/token/ERC721/ERC721.sol";
import "@openzeppelin/contracts/token/ERC721/extensions/ERC721Enumerable.sol";
import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/security/ReentrancyGuard.sol";
import "./GCTCoherence.sol";

/**
 * @title IPAIIdentity
 * @dev NFT-based identity system integrated with GCT coherence tracking
 */
contract IPAIIdentity is ERC721, ERC721Enumerable, Ownable, ReentrancyGuard {
    
    // Reference to GCT coherence contract
    GCTCoherence public gctContract;
    
    // Identity structure
    struct Identity {
        bytes32 idHash;                    // Hash of identity data
        uint256 coherenceScore;            // Latest coherence score
        uint256 lastUpdate;                // Last coherence update
        bool isActive;                     // Identity status
        uint256 creationTime;              // Identity creation timestamp
        string ipfsHash;                   // IPFS hash for additional data
        mapping(string => bytes32) attributes;     // Custom attributes
        mapping(string => bool) verifications;     // Verification status
    }
    
    // Verification types
    enum VerificationType {
        EMAIL,
        PHONE,
        GOVERNMENT_ID,
        BIOMETRIC,
        SOCIAL_MEDIA,
        PROFESSIONAL
    }
    
    // Mappings
    mapping(uint256 => Identity) public identities;
    mapping(address => uint256) public userTokenId;
    mapping(bytes32 => bool) public usedHashes;
    mapping(address => bool) public verifiers;
    
    // State variables
    uint256 public nextTokenId = 1;
    uint256 public constant MAX_SUPPLY = 1000000;
    string private _baseTokenURI;
    
    // Events
    event IdentityMinted(
        address indexed owner,
        uint256 indexed tokenId,
        bytes32 idHash,
        uint256 timestamp
    );
    
    event CoherenceUpdated(
        address indexed owner,
        uint256 indexed tokenId,
        uint256 newScore,
        uint256 timestamp
    );
    
    event IdentityVerified(
        address indexed owner,
        uint256 indexed tokenId,
        string verificationType,
        address indexed verifier
    );
    
    event AttributeSet(
        address indexed owner,
        uint256 indexed tokenId,
        string attributeName,
        bytes32 attributeValue
    );
    
    // Modifiers
    modifier onlyVerifier() {
        require(verifiers[msg.sender] || msg.sender == owner(), "Not authorized verifier");
        _;
    }
    
    modifier onlyTokenOwner(uint256 tokenId) {
        require(ownerOf(tokenId) == msg.sender, "Not token owner");
        _;
    }
    
    modifier validTokenId(uint256 tokenId) {
        require(_exists(tokenId), "Token does not exist");
        _;
    }
    
    constructor(
        address _gctContract,
        string memory _baseURI
    ) ERC721("IPAI Identity", "IPAI") {
        gctContract = GCTCoherence(_gctContract);
        _baseTokenURI = _baseURI;
        verifiers[msg.sender] = true;
    }
    
    /**
     * @dev Mint new identity NFT
     */
    function mintIdentity(
        bytes32 _idHash,
        string memory _ipfsHash
    ) external nonReentrant {
        require(userTokenId[msg.sender] == 0, "Identity already exists");
        require(!usedHashes[_idHash], "ID hash already used");
        require(nextTokenId <= MAX_SUPPLY, "Max supply reached");
        
        uint256 tokenId = nextTokenId++;
        usedHashes[_idHash] = true;
        
        _safeMint(msg.sender, tokenId);
        
        Identity storage newIdentity = identities[tokenId];
        newIdentity.idHash = _idHash;
        newIdentity.isActive = true;
        newIdentity.creationTime = block.timestamp;
        newIdentity.lastUpdate = block.timestamp;
        newIdentity.ipfsHash = _ipfsHash;
        
        userTokenId[msg.sender] = tokenId;
        
        emit IdentityMinted(msg.sender, tokenId, _idHash, block.timestamp);
    }
    
    /**
     * @dev Update coherence score from GCT contract
     */
    function updateCoherenceFromGCT() external {
        uint256 tokenId = userTokenId[msg.sender];
        require(tokenId != 0, "No identity");
        require(identities[tokenId].isActive, "Identity not active");
        
        // Get latest coherence data from GCT contract
        GCTCoherence.CoherenceData memory latestData = gctContract.getLatestCoherence(msg.sender);
        
        identities[tokenId].coherenceScore = latestData.coherenceScore;
        identities[tokenId].lastUpdate = block.timestamp;
        
        emit CoherenceUpdated(msg.sender, tokenId, latestData.coherenceScore, block.timestamp);
    }
    
    /**
     * @dev Set identity attribute
     */
    function setAttribute(
        string memory _attributeName,
        bytes32 _attributeValue
    ) external {
        uint256 tokenId = userTokenId[msg.sender];
        require(tokenId != 0, "No identity");
        
        identities[tokenId].attributes[_attributeName] = _attributeValue;
        
        emit AttributeSet(msg.sender, tokenId, _attributeName, _attributeValue);
    }
    
    /**
     * @dev Get identity attribute
     */
    function getAttribute(
        uint256 tokenId,
        string memory _attributeName
    ) external view validTokenId(tokenId) returns (bytes32) {
        return identities[tokenId].attributes[_attributeName];
    }
    
    /**
     * @dev Verify identity aspect
     */
    function verifyIdentity(
        uint256 tokenId,
        string memory _verificationType
    ) external onlyVerifier validTokenId(tokenId) {
        identities[tokenId].verifications[_verificationType] = true;
        
        emit IdentityVerified(
            ownerOf(tokenId),
            tokenId,
            _verificationType,
            msg.sender
        );
    }
    
    /**
     * @dev Check if identity is verified for specific type
     */
    function isVerified(
        uint256 tokenId,
        string memory _verificationType
    ) external view validTokenId(tokenId) returns (bool) {
        return identities[tokenId].verifications[_verificationType];
    }
    
    /**
     * @dev Get identity overview
     */
    function getIdentityOverview(uint256 tokenId) 
        external 
        view 
        validTokenId(tokenId)
        returns (
            bytes32 idHash,
            uint256 coherenceScore,
            uint256 lastUpdate,
            bool isActive,
            uint256 creationTime,
            string memory ipfsHash
        ) 
    {
        Identity storage identity = identities[tokenId];
        return (
            identity.idHash,
            identity.coherenceScore,
            identity.lastUpdate,
            identity.isActive,
            identity.creationTime,
            identity.ipfsHash
        );
    }
    
    /**
     * @dev Get coherence trajectory from GCT contract
     */
    function getCoherenceTrajectory(
        uint256 tokenId,
        uint256 count
    ) 
        external 
        view 
        validTokenId(tokenId)
        returns (GCTCoherence.CoherenceData[] memory) 
    {
        address owner = ownerOf(tokenId);
        return gctContract.getCoherenceTrajectory(owner, count);
    }
    
    /**
     * @dev Get coherence statistics
     */
    function getCoherenceStatistics(uint256 tokenId)
        external
        view
        validTokenId(tokenId)
        returns (
            uint256 count,
            uint256 averageCoherence,
            uint256 maxCoherence,
            uint256 minCoherence,
            uint256 latestCoherence
        )
    {
        address owner = ownerOf(tokenId);
        return gctContract.getCoherenceStatistics(owner);
    }
    
    /**
     * @dev Deactivate identity (only owner)
     */
    function deactivateIdentity() external {
        uint256 tokenId = userTokenId[msg.sender];
        require(tokenId != 0, "No identity");
        
        identities[tokenId].isActive = false;
    }
    
    /**
     * @dev Reactivate identity (only owner)
     */
    function reactivateIdentity() external {
        uint256 tokenId = userTokenId[msg.sender];
        require(tokenId != 0, "No identity");
        
        identities[tokenId].isActive = true;
    }
    
    /**
     * @dev Add verifier (only contract owner)
     */
    function addVerifier(address _verifier) external onlyOwner {
        verifiers[_verifier] = true;
    }
    
    /**
     * @dev Remove verifier (only contract owner)
     */
    function removeVerifier(address _verifier) external onlyOwner {
        verifiers[_verifier] = false;
    }
    
    /**
     * @dev Set base URI (only owner)
     */
    function setBaseURI(string memory _baseURI) external onlyOwner {
        _baseTokenURI = _baseURI;
    }
    
    /**
     * @dev Override tokenURI to include coherence data
     */
    function tokenURI(uint256 tokenId) 
        public 
        view 
        override 
        validTokenId(tokenId) 
        returns (string memory) 
    {
        string memory baseURI = _baseTokenURI;
        return bytes(baseURI).length > 0 
            ? string(abi.encodePacked(baseURI, Strings.toString(tokenId), ".json"))
            : "";
    }
    
    /**
     * @dev Get tokens owned by address
     */
    function getTokensOfOwner(address owner) 
        external 
        view 
        returns (uint256[] memory) 
    {
        uint256 tokenCount = balanceOf(owner);
        uint256[] memory tokens = new uint256[](tokenCount);
        
        for (uint256 i = 0; i < tokenCount; i++) {
            tokens[i] = tokenOfOwnerByIndex(owner, i);
        }
        
        return tokens;
    }
    
    /**
     * @dev Check if address has identity
     */
    function hasIdentity(address user) external view returns (bool) {
        return userTokenId[user] != 0;
    }
    
    /**
     * @dev Get total active identities
     */
    function getTotalActiveIdentities() external view returns (uint256) {
        uint256 activeCount = 0;
        uint256 totalSupply = totalSupply();
        
        for (uint256 i = 1; i <= totalSupply; i++) {
            if (_exists(i) && identities[i].isActive) {
                activeCount++;
            }
        }
        
        return activeCount;
    }
    
    /**
     * @dev Override transfer functions to update mappings
     */
    function _beforeTokenTransfer(
        address from,
        address to,
        uint256 tokenId,
        uint256 batchSize
    ) internal override(ERC721, ERC721Enumerable) {
        super._beforeTokenTransfer(from, to, tokenId, batchSize);
        
        // Update user token mappings on transfer
        if (from != address(0)) {
            userTokenId[from] = 0;
        }
        if (to != address(0)) {
            userTokenId[to] = tokenId;
        }
    }
    
    /**
     * @dev Required override for interface support
     */
    function supportsInterface(bytes4 interfaceId)
        public
        view
        override(ERC721, ERC721Enumerable)
        returns (bool)
    {
        return super.supportsInterface(interfaceId);
    }
    
    /**
     * @dev Get contract version
     */
    function version() external pure returns (string memory) {
        return "1.0.0";
    }
}