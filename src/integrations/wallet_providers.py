"""
Wallet Provider Integration Module

Supports multiple blockchain wallets for IPAI integration.
Handles wallet connections, transactions, and SAGE token management.
"""

from typing import Dict, List, Optional, Any, Union
from enum import Enum
from dataclasses import dataclass
import asyncio
from abc import ABC, abstractmethod
from web3 import Web3
from eth_account import Account
import json


class WalletType(Enum):
    """Supported wallet types"""
    METAMASK = "metamask"
    WALLETCONNECT = "walletconnect"
    COINBASE = "coinbase"
    PHANTOM = "phantom"
    RAINBOW = "rainbow"
    TRUST = "trust"
    LEDGER = "ledger"
    TREZOR = "trezor"
    PRIVATE_KEY = "private_key"
    CUSTOM = "custom"


class BlockchainNetwork(Enum):
    """Supported blockchain networks"""
    ETHEREUM = "ethereum"
    POLYGON = "polygon"
    ARBITRUM = "arbitrum"
    OPTIMISM = "optimism"
    BSC = "bsc"
    AVALANCHE = "avalanche"
    FANTOM = "fantom"
    SOLANA = "solana"
    LOCAL = "local"


@dataclass
class NetworkConfig:
    """Configuration for a blockchain network"""
    name: str
    chain_id: int
    rpc_url: str
    explorer_url: str
    native_token: str
    sage_token_address: Optional[str] = None
    ipai_identity_address: Optional[str] = None
    gct_coherence_address: Optional[str] = None


@dataclass
class WalletConfig:
    """Configuration for a wallet connection"""
    wallet_type: WalletType
    network: BlockchainNetwork
    address: Optional[str] = None
    private_key: Optional[str] = None  # Only for private key wallets
    connection_params: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.connection_params is None:
            self.connection_params = {}


class WalletProviderInfo:
    """Information about wallet providers and networks"""
    
    NETWORKS = {
        BlockchainNetwork.ETHEREUM: NetworkConfig(
            name="Ethereum Mainnet",
            chain_id=1,
            rpc_url="https://mainnet.infura.io/v3/YOUR_INFURA_KEY",
            explorer_url="https://etherscan.io",
            native_token="ETH"
        ),
        BlockchainNetwork.POLYGON: NetworkConfig(
            name="Polygon Mainnet",
            chain_id=137,
            rpc_url="https://polygon-rpc.com",
            explorer_url="https://polygonscan.com",
            native_token="MATIC"
        ),
        BlockchainNetwork.ARBITRUM: NetworkConfig(
            name="Arbitrum One",
            chain_id=42161,
            rpc_url="https://arb1.arbitrum.io/rpc",
            explorer_url="https://arbiscan.io",
            native_token="ETH"
        ),
        BlockchainNetwork.OPTIMISM: NetworkConfig(
            name="Optimism",
            chain_id=10,
            rpc_url="https://mainnet.optimism.io",
            explorer_url="https://optimistic.etherscan.io",
            native_token="ETH"
        ),
        BlockchainNetwork.BSC: NetworkConfig(
            name="BNB Smart Chain",
            chain_id=56,
            rpc_url="https://bsc-dataseed.binance.org",
            explorer_url="https://bscscan.com",
            native_token="BNB"
        ),
        BlockchainNetwork.AVALANCHE: NetworkConfig(
            name="Avalanche C-Chain",
            chain_id=43114,
            rpc_url="https://api.avax.network/ext/bc/C/rpc",
            explorer_url="https://snowtrace.io",
            native_token="AVAX"
        ),
        BlockchainNetwork.FANTOM: NetworkConfig(
            name="Fantom Opera",
            chain_id=250,
            rpc_url="https://rpc.ftm.tools",
            explorer_url="https://ftmscan.com",
            native_token="FTM"
        ),
        BlockchainNetwork.LOCAL: NetworkConfig(
            name="Local Network",
            chain_id=31337,
            rpc_url="http://localhost:8545",
            explorer_url="http://localhost:3000",
            native_token="ETH"
        )
    }
    
    WALLET_PROVIDERS = {
        WalletType.METAMASK: {
            "name": "MetaMask",
            "supports_hardware": True,
            "mobile_support": True,
            "browser_extension": True,
            "supported_networks": ["ethereum", "polygon", "arbitrum", "optimism", "bsc", "avalanche", "fantom"]
        },
        WalletType.WALLETCONNECT: {
            "name": "WalletConnect",
            "supports_hardware": True,
            "mobile_support": True,
            "browser_extension": False,
            "supported_networks": ["ethereum", "polygon", "arbitrum", "optimism", "bsc", "avalanche", "fantom"]
        },
        WalletType.COINBASE: {
            "name": "Coinbase Wallet",
            "supports_hardware": False,
            "mobile_support": True,
            "browser_extension": True,
            "supported_networks": ["ethereum", "polygon", "arbitrum", "optimism"]
        },
        WalletType.PHANTOM: {
            "name": "Phantom",
            "supports_hardware": True,
            "mobile_support": True,
            "browser_extension": True,
            "supported_networks": ["solana", "ethereum", "polygon"]
        },
        WalletType.RAINBOW: {
            "name": "Rainbow",
            "supports_hardware": False,
            "mobile_support": True,
            "browser_extension": True,
            "supported_networks": ["ethereum", "polygon", "arbitrum", "optimism"]
        },
        WalletType.TRUST: {
            "name": "Trust Wallet",
            "supports_hardware": False,
            "mobile_support": True,
            "browser_extension": True,
            "supported_networks": ["ethereum", "polygon", "bsc", "avalanche"]
        },
        WalletType.LEDGER: {
            "name": "Ledger",
            "supports_hardware": True,
            "mobile_support": False,
            "browser_extension": False,
            "supported_networks": ["ethereum", "polygon", "arbitrum", "optimism", "bsc", "avalanche", "fantom"]
        },
        WalletType.TREZOR: {
            "name": "Trezor",
            "supports_hardware": True,
            "mobile_support": False,
            "browser_extension": False,
            "supported_networks": ["ethereum", "polygon", "arbitrum", "optimism"]
        },
        WalletType.PRIVATE_KEY: {
            "name": "Private Key",
            "supports_hardware": False,
            "mobile_support": False,
            "browser_extension": False,
            "supported_networks": ["ethereum", "polygon", "arbitrum", "optimism", "bsc", "avalanche", "fantom", "local"]
        }
    }


class BaseWalletClient(ABC):
    """Base class for wallet clients"""
    
    def __init__(self, config: WalletConfig):
        self.config = config
        self.web3: Optional[Web3] = None
        self.account: Optional[Any] = None
        self.connected = False
    
    @abstractmethod
    async def connect(self) -> bool:
        """Connect to the wallet"""
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from the wallet"""
        pass
    
    @abstractmethod
    async def get_balance(self) -> float:
        """Get native token balance"""
        pass
    
    @abstractmethod
    async def get_token_balance(self, token_address: str) -> float:
        """Get token balance"""
        pass
    
    @abstractmethod
    async def send_transaction(self, to: str, value: float, data: bytes = b'') -> str:
        """Send a transaction"""
        pass
    
    @abstractmethod
    async def sign_message(self, message: str) -> str:
        """Sign a message"""
        pass


class PrivateKeyWallet(BaseWalletClient):
    """Private key based wallet implementation"""
    
    async def connect(self) -> bool:
        """Connect using private key"""
        try:
            network_config = WalletProviderInfo.NETWORKS.get(self.config.network)
            if not network_config:
                raise ValueError(f"Unsupported network: {self.config.network}")
            
            self.web3 = Web3(Web3.HTTPProvider(network_config.rpc_url))
            
            if not self.web3.is_connected():
                return False
            
            # Create account from private key
            if self.config.private_key:
                self.account = Account.from_key(self.config.private_key)
                self.config.address = self.account.address
                self.connected = True
                return True
            
            return False
            
        except Exception as e:
            print(f"Connection error: {e}")
            return False
    
    async def disconnect(self) -> None:
        """Disconnect wallet"""
        self.connected = False
        self.account = None
        self.web3 = None
    
    async def get_balance(self) -> float:
        """Get ETH balance"""
        if not self.connected or not self.web3:
            raise Exception("Wallet not connected")
        
        balance_wei = self.web3.eth.get_balance(self.config.address)
        return self.web3.from_wei(balance_wei, 'ether')
    
    async def get_token_balance(self, token_address: str) -> float:
        """Get ERC20 token balance"""
        if not self.connected or not self.web3:
            raise Exception("Wallet not connected")
        
        # ERC20 ABI for balanceOf
        erc20_abi = [
            {
                "inputs": [{"name": "account", "type": "address"}],
                "name": "balanceOf",
                "outputs": [{"name": "", "type": "uint256"}],
                "stateMutability": "view",
                "type": "function"
            },
            {
                "inputs": [],
                "name": "decimals",
                "outputs": [{"name": "", "type": "uint8"}],
                "stateMutability": "view",
                "type": "function"
            }
        ]
        
        contract = self.web3.eth.contract(address=token_address, abi=erc20_abi)
        balance = contract.functions.balanceOf(self.config.address).call()
        decimals = contract.functions.decimals().call()
        
        return balance / (10 ** decimals)
    
    async def send_transaction(self, to: str, value: float, data: bytes = b'') -> str:
        """Send a transaction"""
        if not self.connected or not self.web3 or not self.account:
            raise Exception("Wallet not connected")
        
        # Build transaction
        nonce = self.web3.eth.get_transaction_count(self.config.address)
        gas_price = self.web3.eth.gas_price
        
        transaction = {
            'nonce': nonce,
            'to': to,
            'value': self.web3.to_wei(value, 'ether'),
            'gas': 21000,  # Basic transfer
            'gasPrice': gas_price,
            'chainId': self.web3.eth.chain_id
        }
        
        if data:
            transaction['data'] = data
            transaction['gas'] = 100000  # Increase for contract interaction
        
        # Sign transaction
        signed_txn = self.account.sign_transaction(transaction)
        
        # Send transaction
        tx_hash = self.web3.eth.send_raw_transaction(signed_txn.rawTransaction)
        
        return tx_hash.hex()
    
    async def sign_message(self, message: str) -> str:
        """Sign a message"""
        if not self.account:
            raise Exception("Wallet not connected")
        
        message_hash = Web3.keccak(text=message)
        signed_message = self.account.signHash(message_hash)
        
        return signed_message.signature.hex()


class MetaMaskWallet(BaseWalletClient):
    """MetaMask wallet implementation (requires browser integration)"""
    
    async def connect(self) -> bool:
        """Connect to MetaMask"""
        # This would be implemented with web3modal or similar in a real browser environment
        # For now, return a placeholder
        return False
    
    async def disconnect(self) -> None:
        """Disconnect from MetaMask"""
        self.connected = False
    
    async def get_balance(self) -> float:
        """Get balance from MetaMask"""
        if not self.connected:
            raise Exception("Wallet not connected")
        return 0.0
    
    async def get_token_balance(self, token_address: str) -> float:
        """Get token balance from MetaMask"""
        if not self.connected:
            raise Exception("Wallet not connected")
        return 0.0
    
    async def send_transaction(self, to: str, value: float, data: bytes = b'') -> str:
        """Send transaction via MetaMask"""
        if not self.connected:
            raise Exception("Wallet not connected")
        return ""
    
    async def sign_message(self, message: str) -> str:
        """Sign message with MetaMask"""
        if not self.connected:
            raise Exception("Wallet not connected")
        return ""


class WalletManager:
    """Manager for wallet connections"""
    
    def __init__(self):
        self.wallets: Dict[str, BaseWalletClient] = {}
        self.active_wallet: Optional[str] = None
        self.sage_token_abi: List[Dict] = self._load_sage_token_abi()
    
    def _load_sage_token_abi(self) -> List[Dict]:
        """Load SAGE token ABI"""
        # This would load from the compiled contract
        # For now, return essential functions
        return [
            {
                "inputs": [{"name": "account", "type": "address"}],
                "name": "balanceOf",
                "outputs": [{"name": "", "type": "uint256"}],
                "stateMutability": "view",
                "type": "function"
            },
            {
                "inputs": [
                    {"name": "recipient", "type": "address"},
                    {"name": "amount", "type": "uint256"}
                ],
                "name": "transfer",
                "outputs": [{"name": "", "type": "bool"}],
                "stateMutability": "nonpayable",
                "type": "function"
            },
            {
                "inputs": [{"name": "user", "type": "address"}],
                "name": "claimDailyReward",
                "outputs": [],
                "stateMutability": "nonpayable",
                "type": "function"
            }
        ]
    
    async def add_wallet(self, name: str, config: WalletConfig) -> bool:
        """Add and connect a wallet"""
        # Create appropriate wallet client
        if config.wallet_type == WalletType.PRIVATE_KEY:
            wallet = PrivateKeyWallet(config)
        elif config.wallet_type == WalletType.METAMASK:
            wallet = MetaMaskWallet(config)
        else:
            raise ValueError(f"Unsupported wallet type: {config.wallet_type}")
        
        # Try to connect
        connected = await wallet.connect()
        if connected:
            self.wallets[name] = wallet
            if not self.active_wallet:
                self.active_wallet = name
        
        return connected
    
    def set_active_wallet(self, name: str) -> None:
        """Set the active wallet"""
        if name not in self.wallets:
            raise ValueError(f"Wallet {name} not found")
        self.active_wallet = name
    
    def get_active_wallet(self) -> BaseWalletClient:
        """Get the active wallet client"""
        if not self.active_wallet:
            raise ValueError("No active wallet")
        return self.wallets[self.active_wallet]
    
    async def get_sage_balance(self, wallet_name: Optional[str] = None) -> float:
        """Get SAGE token balance"""
        wallet_name = wallet_name or self.active_wallet
        if not wallet_name:
            raise ValueError("No wallet specified")
        
        wallet = self.wallets.get(wallet_name)
        if not wallet:
            raise ValueError(f"Wallet {wallet_name} not found")
        
        # Get network config
        network_config = WalletProviderInfo.NETWORKS.get(wallet.config.network)
        if not network_config or not network_config.sage_token_address:
            return 0.0
        
        return await wallet.get_token_balance(network_config.sage_token_address)
    
    async def claim_daily_sage_reward(self, wallet_name: Optional[str] = None) -> str:
        """Claim daily SAGE reward"""
        wallet_name = wallet_name or self.active_wallet
        if not wallet_name:
            raise ValueError("No wallet specified")
        
        wallet = self.wallets.get(wallet_name)
        if not wallet:
            raise ValueError(f"Wallet {wallet_name} not found")
        
        # Get network config
        network_config = WalletProviderInfo.NETWORKS.get(wallet.config.network)
        if not network_config or not network_config.sage_token_address:
            raise ValueError("SAGE token not deployed on this network")
        
        # Encode function call
        web3 = wallet.web3
        contract = web3.eth.contract(
            address=network_config.sage_token_address,
            abi=self.sage_token_abi
        )
        
        # Get function data
        function = contract.functions.claimDailyReward(wallet.config.address)
        data = function.build_transaction({
            'from': wallet.config.address,
            'gas': 100000,
            'gasPrice': web3.eth.gas_price
        })['data']
        
        # Send transaction
        return await wallet.send_transaction(
            to=network_config.sage_token_address,
            value=0,
            data=data
        )
    
    def list_wallets(self) -> List[Dict[str, Any]]:
        """List all connected wallets"""
        wallets = []
        for name, wallet in self.wallets.items():
            wallets.append({
                "name": name,
                "type": wallet.config.wallet_type.value,
                "network": wallet.config.network.value,
                "address": wallet.config.address,
                "connected": wallet.connected,
                "active": name == self.active_wallet
            })
        return wallets
    
    async def disconnect_wallet(self, name: str) -> None:
        """Disconnect a wallet"""
        wallet = self.wallets.get(name)
        if wallet:
            await wallet.disconnect()
            del self.wallets[name]
            if self.active_wallet == name:
                self.active_wallet = None
    
    def save_config(self, path: str) -> None:
        """Save wallet configuration"""
        config_data = {
            "wallets": {
                name: {
                    "wallet_type": wallet.config.wallet_type.value,
                    "network": wallet.config.network.value,
                    "address": wallet.config.address,
                    "connection_params": wallet.config.connection_params
                }
                for name, wallet in self.wallets.items()
            },
            "active_wallet": self.active_wallet
        }
        
        # Don't save private keys!
        with open(path, 'w') as f:
            json.dump(config_data, f, indent=2)
    
    def load_config(self, path: str) -> None:
        """Load wallet configuration"""
        with open(path, 'r') as f:
            config_data = json.load(f)
        
        # Note: This won't restore private key wallets
        # They need to be re-added manually for security