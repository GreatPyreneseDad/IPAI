"""
Configuration Manager for IPAI Integrations

Handles secure storage and management of API keys, wallet configurations,
and integration settings with encryption.
"""

import os
import json
from typing import Dict, Optional, Any
from pathlib import Path
import base64
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import keyring
import getpass


class SecureConfigManager:
    """Secure configuration manager with encryption"""
    
    def __init__(self, config_dir: Optional[str] = None):
        self.config_dir = Path(config_dir or os.path.expanduser("~/.ipai/config"))
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        self.config_file = self.config_dir / "integrations.enc"
        self.salt_file = self.config_dir / ".salt"
        
        self._encryption_key: Optional[bytes] = None
        self._config_cache: Dict[str, Any] = {}
    
    def _get_or_create_salt(self) -> bytes:
        """Get or create encryption salt"""
        if self.salt_file.exists():
            return self.salt_file.read_bytes()
        else:
            salt = os.urandom(16)
            self.salt_file.write_bytes(salt)
            # Make salt file hidden and read-only
            os.chmod(self.salt_file, 0o400)
            return salt
    
    def _derive_key(self, password: str) -> bytes:
        """Derive encryption key from password"""
        salt = self._get_or_create_salt()
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        return key
    
    def _get_system_keyring_password(self) -> Optional[str]:
        """Try to get master password from system keyring"""
        try:
            return keyring.get_password("ipai", "master_password")
        except Exception:
            return None
    
    def _set_system_keyring_password(self, password: str) -> None:
        """Store master password in system keyring"""
        try:
            keyring.set_password("ipai", "master_password", password)
        except Exception:
            pass  # Keyring might not be available
    
    def unlock(self, password: Optional[str] = None) -> bool:
        """Unlock the configuration store"""
        if password is None:
            # Try system keyring first
            password = self._get_system_keyring_password()
            
            if password is None:
                # Prompt for password
                password = getpass.getpass("Enter IPAI master password: ")
        
        try:
            self._encryption_key = self._derive_key(password)
            
            # Test decryption
            if self.config_file.exists():
                self._load_config()
            
            # Store in keyring if successful
            self._set_system_keyring_password(password)
            
            return True
        except Exception:
            self._encryption_key = None
            return False
    
    def is_unlocked(self) -> bool:
        """Check if configuration is unlocked"""
        return self._encryption_key is not None
    
    def _encrypt_data(self, data: Dict[str, Any]) -> bytes:
        """Encrypt configuration data"""
        if not self._encryption_key:
            raise ValueError("Configuration not unlocked")
        
        fernet = Fernet(self._encryption_key)
        json_data = json.dumps(data, indent=2)
        return fernet.encrypt(json_data.encode())
    
    def _decrypt_data(self, encrypted_data: bytes) -> Dict[str, Any]:
        """Decrypt configuration data"""
        if not self._encryption_key:
            raise ValueError("Configuration not unlocked")
        
        fernet = Fernet(self._encryption_key)
        json_data = fernet.decrypt(encrypted_data).decode()
        return json.loads(json_data)
    
    def _load_config(self) -> None:
        """Load configuration from encrypted file"""
        if self.config_file.exists():
            encrypted_data = self.config_file.read_bytes()
            self._config_cache = self._decrypt_data(encrypted_data)
        else:
            self._config_cache = self._get_default_config()
    
    def _save_config(self) -> None:
        """Save configuration to encrypted file"""
        encrypted_data = self._encrypt_data(self._config_cache)
        self.config_file.write_bytes(encrypted_data)
        # Set restrictive permissions
        os.chmod(self.config_file, 0o600)
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration structure"""
        return {
            "llm_providers": {},
            "wallets": {},
            "preferences": {
                "default_llm": None,
                "default_wallet": None,
                "auto_connect_wallet": False,
                "coherence_tracking": True
            },
            "contracts": {
                "sage_token": {},
                "ipai_identity": {},
                "gct_coherence": {}
            }
        }
    
    def add_llm_provider(
        self,
        name: str,
        provider: str,
        api_key: str,
        api_base: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs
    ) -> None:
        """Add LLM provider configuration"""
        if not self.is_unlocked():
            raise ValueError("Configuration not unlocked")
        
        self._config_cache.setdefault("llm_providers", {})[name] = {
            "provider": provider,
            "api_key": api_key,
            "api_base": api_base,
            "model": model,
            **kwargs
        }
        
        self._save_config()
    
    def get_llm_provider(self, name: str) -> Optional[Dict[str, Any]]:
        """Get LLM provider configuration"""
        if not self.is_unlocked():
            raise ValueError("Configuration not unlocked")
        
        return self._config_cache.get("llm_providers", {}).get(name)
    
    def list_llm_providers(self) -> List[str]:
        """List configured LLM providers"""
        if not self.is_unlocked():
            raise ValueError("Configuration not unlocked")
        
        return list(self._config_cache.get("llm_providers", {}).keys())
    
    def remove_llm_provider(self, name: str) -> None:
        """Remove LLM provider configuration"""
        if not self.is_unlocked():
            raise ValueError("Configuration not unlocked")
        
        providers = self._config_cache.get("llm_providers", {})
        if name in providers:
            del providers[name]
            self._save_config()
    
    def add_wallet(
        self,
        name: str,
        wallet_type: str,
        network: str,
        address: Optional[str] = None,
        private_key: Optional[str] = None,
        **kwargs
    ) -> None:
        """Add wallet configuration"""
        if not self.is_unlocked():
            raise ValueError("Configuration not unlocked")
        
        wallet_config = {
            "wallet_type": wallet_type,
            "network": network,
            "address": address,
            **kwargs
        }
        
        # Store private key separately with extra encryption
        if private_key:
            # Double encrypt private keys
            fernet = Fernet(self._encryption_key)
            encrypted_key = fernet.encrypt(private_key.encode()).decode()
            wallet_config["encrypted_private_key"] = encrypted_key
        
        self._config_cache.setdefault("wallets", {})[name] = wallet_config
        self._save_config()
    
    def get_wallet(self, name: str) -> Optional[Dict[str, Any]]:
        """Get wallet configuration"""
        if not self.is_unlocked():
            raise ValueError("Configuration not unlocked")
        
        wallet = self._config_cache.get("wallets", {}).get(name)
        if wallet and "encrypted_private_key" in wallet:
            # Decrypt private key
            fernet = Fernet(self._encryption_key)
            wallet = wallet.copy()
            encrypted_key = wallet.pop("encrypted_private_key")
            wallet["private_key"] = fernet.decrypt(encrypted_key.encode()).decode()
        
        return wallet
    
    def list_wallets(self) -> List[str]:
        """List configured wallets"""
        if not self.is_unlocked():
            raise ValueError("Configuration not unlocked")
        
        return list(self._config_cache.get("wallets", {}).keys())
    
    def remove_wallet(self, name: str) -> None:
        """Remove wallet configuration"""
        if not self.is_unlocked():
            raise ValueError("Configuration not unlocked")
        
        wallets = self._config_cache.get("wallets", {})
        if name in wallets:
            del wallets[name]
            self._save_config()
    
    def set_preference(self, key: str, value: Any) -> None:
        """Set a preference value"""
        if not self.is_unlocked():
            raise ValueError("Configuration not unlocked")
        
        self._config_cache.setdefault("preferences", {})[key] = value
        self._save_config()
    
    def get_preference(self, key: str, default: Any = None) -> Any:
        """Get a preference value"""
        if not self.is_unlocked():
            raise ValueError("Configuration not unlocked")
        
        return self._config_cache.get("preferences", {}).get(key, default)
    
    def set_contract_address(
        self,
        contract_name: str,
        network: str,
        address: str
    ) -> None:
        """Set contract address for a network"""
        if not self.is_unlocked():
            raise ValueError("Configuration not unlocked")
        
        contracts = self._config_cache.setdefault("contracts", {})
        contract_config = contracts.setdefault(contract_name, {})
        contract_config[network] = address
        
        self._save_config()
    
    def get_contract_address(
        self,
        contract_name: str,
        network: str
    ) -> Optional[str]:
        """Get contract address for a network"""
        if not self.is_unlocked():
            raise ValueError("Configuration not unlocked")
        
        return self._config_cache.get("contracts", {}).get(contract_name, {}).get(network)
    
    def export_config(self, exclude_sensitive: bool = True) -> Dict[str, Any]:
        """Export configuration (optionally excluding sensitive data)"""
        if not self.is_unlocked():
            raise ValueError("Configuration not unlocked")
        
        config = self._config_cache.copy()
        
        if exclude_sensitive:
            # Remove API keys and private keys
            for provider in config.get("llm_providers", {}).values():
                provider.pop("api_key", None)
            
            for wallet in config.get("wallets", {}).values():
                wallet.pop("encrypted_private_key", None)
        
        return config
    
    def reset_password(self, old_password: str, new_password: str) -> bool:
        """Reset master password"""
        # Verify old password
        if not self.unlock(old_password):
            return False
        
        # Re-encrypt with new password
        config_data = self._config_cache.copy()
        
        # Derive new key
        self._encryption_key = self._derive_key(new_password)
        
        # Save with new encryption
        self._save_config()
        
        # Update keyring
        self._set_system_keyring_password(new_password)
        
        return True
    
    def clear_all_data(self, password: str) -> bool:
        """Clear all configuration data (requires password confirmation)"""
        if not self.unlock(password):
            return False
        
        # Clear keyring
        try:
            keyring.delete_password("ipai", "master_password")
        except Exception:
            pass
        
        # Delete files
        if self.config_file.exists():
            self.config_file.unlink()
        if self.salt_file.exists():
            self.salt_file.unlink()
        
        # Clear cache
        self._config_cache = {}
        self._encryption_key = None
        
        return True


class ConfigValidation:
    """Validation helpers for configuration values"""
    
    @staticmethod
    def validate_api_key(provider: str, api_key: str) -> bool:
        """Validate API key format"""
        validations = {
            "openai": lambda k: k.startswith("sk-") and len(k) > 20,
            "anthropic": lambda k: k.startswith("sk-ant-") and len(k) > 20,
            "google": lambda k: len(k) == 39,
            "cohere": lambda k: len(k) == 40,
            "together": lambda k: len(k) > 20,
            "replicate": lambda k: k.startswith("r8_") and len(k) == 40,
            "huggingface": lambda k: k.startswith("hf_") and len(k) > 20,
            "groq": lambda k: k.startswith("gsk_") and len(k) > 20,
            "mistral": lambda k: len(k) == 32
        }
        
        validator = validations.get(provider.lower())
        if validator:
            return validator(api_key)
        
        # Default validation - just check it's not empty
        return len(api_key) > 0
    
    @staticmethod
    def validate_ethereum_address(address: str) -> bool:
        """Validate Ethereum address format"""
        if not address.startswith("0x"):
            return False
        if len(address) != 42:
            return False
        
        try:
            int(address, 16)
            return True
        except ValueError:
            return False
    
    @staticmethod
    def validate_private_key(key: str) -> bool:
        """Validate Ethereum private key format"""
        if key.startswith("0x"):
            key = key[2:]
        
        if len(key) != 64:
            return False
        
        try:
            int(key, 16)
            return True
        except ValueError:
            return False