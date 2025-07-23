"""
Configuration API Endpoints

This module provides REST endpoints for managing LLM providers,
wallets, and integration settings.
"""

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, status
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, validator
import asyncio

from ...models.user import User
from ...integrations.llm_providers import (
    LLMProvider, LLMConfig, LLMManager, LLMProviderInfo
)
from ...integrations.wallet_providers import (
    WalletType, WalletConfig, WalletManager, BlockchainNetwork
)
from ...integrations.config_manager import SecureConfigManager, ConfigValidation
from ...core.database import Database
from ..dependencies import get_current_active_user, get_database

router = APIRouter(prefix="/config")

# Global managers (in production, use proper state management)
llm_manager = LLMManager()
wallet_manager = WalletManager()
config_manager = SecureConfigManager()


# Pydantic models

class LLMProviderRequest(BaseModel):
    """Request model for adding LLM provider"""
    name: str = Field(..., min_length=1, max_length=50)
    provider: str = Field(..., description="Provider type")
    api_key: str = Field(..., min_length=1)
    api_base: Optional[str] = None
    model: str = Field(..., min_length=1)
    temperature: float = Field(0.7, ge=0, le=2)
    max_tokens: int = Field(2000, ge=1, le=100000)
    
    @validator('provider')
    def validate_provider(cls, v):
        valid_providers = [p.value for p in LLMProvider]
        if v not in valid_providers:
            raise ValueError(f"Invalid provider. Must be one of: {valid_providers}")
        return v


class WalletRequest(BaseModel):
    """Request model for adding wallet"""
    name: str = Field(..., min_length=1, max_length=50)
    wallet_type: str = Field(..., description="Wallet type")
    network: str = Field(..., description="Blockchain network")
    address: Optional[str] = None
    private_key: Optional[str] = None
    
    @validator('wallet_type')
    def validate_wallet_type(cls, v):
        valid_types = [t.value for t in WalletType]
        if v not in valid_types:
            raise ValueError(f"Invalid wallet type. Must be one of: {valid_types}")
        return v
    
    @validator('network')
    def validate_network(cls, v):
        valid_networks = [n.value for n in BlockchainNetwork]
        if v not in valid_networks:
            raise ValueError(f"Invalid network. Must be one of: {valid_networks}")
        return v


class SettingsUpdate(BaseModel):
    """Request model for updating settings"""
    auto_connect_wallet: bool = False
    coherence_tracking: bool = True
    default_llm: Optional[str] = None
    default_wallet: Optional[str] = None


class PasswordChange(BaseModel):
    """Request model for changing password"""
    old_password: str = Field(..., min_length=8)
    new_password: str = Field(..., min_length=8)


# Helper functions

async def ensure_config_unlocked(user_id: str) -> bool:
    """Ensure configuration is unlocked for user"""
    if not config_manager.is_unlocked():
        # In production, this would use user-specific encryption
        # For now, use a default password
        return config_manager.unlock("default_password")
    return True


# Endpoints

@router.get("/llm-providers")
async def list_llm_providers(
    current_user: User = Depends(get_current_active_user)
):
    """List all configured LLM providers"""
    
    try:
        await ensure_config_unlocked(current_user.id)
        
        # Get providers from manager
        providers = llm_manager.list_providers()
        
        # Add additional info
        for provider in providers:
            info = LLMProviderInfo.get_info(LLMProvider(provider["provider"]))
            provider["provider_name"] = info["name"]
            provider["supports_streaming"] = info["supports_streaming"]
            provider["supports_functions"] = info["supports_functions"]
        
        return providers
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list LLM providers: {str(e)}"
        )


@router.post("/llm-providers")
async def add_llm_provider(
    provider_data: LLMProviderRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_active_user)
):
    """Add a new LLM provider"""
    
    try:
        await ensure_config_unlocked(current_user.id)
        
        # Validate API key format
        if not ConfigValidation.validate_api_key(provider_data.provider, provider_data.api_key):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid API key format for this provider"
            )
        
        # Create LLM config
        config = LLMConfig(
            provider=LLMProvider(provider_data.provider),
            api_key=provider_data.api_key,
            api_base=provider_data.api_base,
            model=provider_data.model,
            temperature=provider_data.temperature,
            max_tokens=provider_data.max_tokens
        )
        
        # Add to manager
        llm_manager.add_provider(provider_data.name, config)
        
        # Save to secure storage
        config_manager.add_llm_provider(
            name=provider_data.name,
            provider=provider_data.provider,
            api_key=provider_data.api_key,
            api_base=provider_data.api_base,
            model=provider_data.model,
            temperature=provider_data.temperature,
            max_tokens=provider_data.max_tokens
        )
        
        # Test connection in background
        background_tasks.add_task(
            test_llm_connection,
            provider_data.name,
            current_user.id
        )
        
        return {
            "status": "success",
            "message": f"LLM provider '{provider_data.name}' added successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to add LLM provider: {str(e)}"
        )


@router.post("/llm-providers/{provider_name}/test")
async def test_llm_provider(
    provider_name: str,
    current_user: User = Depends(get_current_active_user)
):
    """Test LLM provider connection"""
    
    try:
        # Validate provider
        is_valid = await llm_manager.validate_provider(provider_name)
        
        if is_valid:
            # Try a simple completion
            response = await llm_manager.complete(
                "Hello, this is a test. Please respond with 'Test successful'.",
                provider=provider_name,
                max_tokens=20
            )
            
            return {
                "status": "success",
                "valid": True,
                "response": response
            }
        else:
            return {
                "status": "error",
                "valid": False,
                "message": "Failed to connect to provider"
            }
            
    except Exception as e:
        return {
            "status": "error",
            "valid": False,
            "message": str(e)
        }


@router.delete("/llm-providers/{provider_name}")
async def remove_llm_provider(
    provider_name: str,
    current_user: User = Depends(get_current_active_user)
):
    """Remove an LLM provider"""
    
    try:
        await ensure_config_unlocked(current_user.id)
        
        # Remove from manager
        if provider_name in [p["name"] for p in llm_manager.list_providers()]:
            del llm_manager.clients[provider_name]
            del llm_manager.configs[provider_name]
            
            if llm_manager.active_provider == provider_name:
                llm_manager.active_provider = None
        
        # Remove from secure storage
        config_manager.remove_llm_provider(provider_name)
        
        return {
            "status": "success",
            "message": f"LLM provider '{provider_name}' removed"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to remove LLM provider: {str(e)}"
        )


@router.get("/wallets")
async def list_wallets(
    current_user: User = Depends(get_current_active_user)
):
    """List all connected wallets"""
    
    try:
        await ensure_config_unlocked(current_user.id)
        
        # Get wallets from manager
        wallets = wallet_manager.list_wallets()
        
        return wallets
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list wallets: {str(e)}"
        )


@router.post("/wallets")
async def add_wallet(
    wallet_data: WalletRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_active_user)
):
    """Add a new wallet"""
    
    try:
        await ensure_config_unlocked(current_user.id)
        
        # Validate address if provided
        if wallet_data.address and not ConfigValidation.validate_ethereum_address(wallet_data.address):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid Ethereum address format"
            )
        
        # Validate private key if provided
        if wallet_data.private_key and not ConfigValidation.validate_private_key(wallet_data.private_key):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid private key format"
            )
        
        # Create wallet config
        config = WalletConfig(
            wallet_type=WalletType(wallet_data.wallet_type),
            network=BlockchainNetwork(wallet_data.network),
            address=wallet_data.address,
            private_key=wallet_data.private_key
        )
        
        # Add to manager and connect
        connected = await wallet_manager.add_wallet(wallet_data.name, config)
        
        if not connected:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to connect wallet"
            )
        
        # Save to secure storage
        config_manager.add_wallet(
            name=wallet_data.name,
            wallet_type=wallet_data.wallet_type,
            network=wallet_data.network,
            address=config.address,  # Use address from connected wallet
            private_key=wallet_data.private_key
        )
        
        return {
            "status": "success",
            "message": f"Wallet '{wallet_data.name}' added successfully",
            "address": config.address
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to add wallet: {str(e)}"
        )


@router.delete("/wallets/{wallet_name}")
async def remove_wallet(
    wallet_name: str,
    current_user: User = Depends(get_current_active_user)
):
    """Remove a wallet"""
    
    try:
        await ensure_config_unlocked(current_user.id)
        
        # Disconnect and remove from manager
        await wallet_manager.disconnect_wallet(wallet_name)
        
        # Remove from secure storage
        config_manager.remove_wallet(wallet_name)
        
        return {
            "status": "success",
            "message": f"Wallet '{wallet_name}' removed"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to remove wallet: {str(e)}"
        )


@router.get("/settings")
async def get_settings(
    current_user: User = Depends(get_current_active_user)
):
    """Get user settings"""
    
    try:
        await ensure_config_unlocked(current_user.id)
        
        return {
            "auto_connect_wallet": config_manager.get_preference("auto_connect_wallet", False),
            "coherence_tracking": config_manager.get_preference("coherence_tracking", True),
            "default_llm": config_manager.get_preference("default_llm"),
            "default_wallet": config_manager.get_preference("default_wallet")
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get settings: {str(e)}"
        )


@router.put("/settings")
async def update_settings(
    settings: SettingsUpdate,
    current_user: User = Depends(get_current_active_user),
    db: Database = Depends(get_database)
):
    """Update user settings"""
    
    try:
        await ensure_config_unlocked(current_user.id)
        
        # Update preferences
        config_manager.set_preference("auto_connect_wallet", settings.auto_connect_wallet)
        config_manager.set_preference("coherence_tracking", settings.coherence_tracking)
        config_manager.set_preference("default_llm", settings.default_llm)
        config_manager.set_preference("default_wallet", settings.default_wallet)
        
        # Update active providers
        if settings.default_llm:
            llm_manager.set_active_provider(settings.default_llm)
        
        if settings.default_wallet:
            wallet_manager.set_active_wallet(settings.default_wallet)
        
        # Update user preferences in database
        await db.update_user_preferences(
            current_user.id,
            {
                "coherence_tracking": settings.coherence_tracking,
                "default_llm": settings.default_llm,
                "default_wallet": settings.default_wallet
            }
        )
        
        return {
            "status": "success",
            "message": "Settings updated successfully"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update settings: {str(e)}"
        )


@router.post("/change-password")
async def change_master_password(
    password_data: PasswordChange,
    current_user: User = Depends(get_current_active_user)
):
    """Change master password for configuration encryption"""
    
    try:
        success = config_manager.reset_password(
            password_data.old_password,
            password_data.new_password
        )
        
        if success:
            return {
                "status": "success",
                "message": "Password changed successfully"
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid current password"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to change password: {str(e)}"
        )


@router.get("/llm-models/{provider}")
async def get_llm_models(
    provider: str,
    current_user: User = Depends(get_current_active_user)
):
    """Get available models for an LLM provider"""
    
    try:
        # Validate provider
        if provider not in [p.value for p in LLMProvider]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid provider"
            )
        
        provider_enum = LLMProvider(provider)
        models = llm_manager.get_provider_models(provider_enum)
        
        return {
            "provider": provider,
            "models": models
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get models: {str(e)}"
        )


# Background tasks

async def test_llm_connection(provider_name: str, user_id: str):
    """Background task to test LLM connection"""
    try:
        is_valid = await llm_manager.validate_provider(provider_name)
        
        # Log result (in production, update database)
        if is_valid:
            print(f"LLM provider {provider_name} validated successfully for user {user_id}")
        else:
            print(f"LLM provider {provider_name} validation failed for user {user_id}")
            
    except Exception as e:
        print(f"Error testing LLM provider {provider_name}: {e}")


# Initialize default configuration on startup
async def initialize_config():
    """Initialize configuration manager with defaults"""
    try:
        # Unlock with default password (in production, use proper key management)
        config_manager.unlock("default_password")
        
        # Load saved configurations
        for name in config_manager.list_llm_providers():
            provider_config = config_manager.get_llm_provider(name)
            if provider_config:
                config = LLMConfig(
                    provider=LLMProvider(provider_config["provider"]),
                    api_key=provider_config["api_key"],
                    api_base=provider_config.get("api_base"),
                    model=provider_config.get("model", ""),
                    temperature=provider_config.get("temperature", 0.7),
                    max_tokens=provider_config.get("max_tokens", 2000)
                )
                llm_manager.add_provider(name, config)
        
        # Load wallets
        for name in config_manager.list_wallets():
            wallet_config = config_manager.get_wallet(name)
            if wallet_config:
                config = WalletConfig(
                    wallet_type=WalletType(wallet_config["wallet_type"]),
                    network=BlockchainNetwork(wallet_config["network"]),
                    address=wallet_config.get("address"),
                    private_key=wallet_config.get("private_key")
                )
                await wallet_manager.add_wallet(name, config)
        
        # Set defaults
        default_llm = config_manager.get_preference("default_llm")
        if default_llm and default_llm in [p["name"] for p in llm_manager.list_providers()]:
            llm_manager.set_active_provider(default_llm)
        
        default_wallet = config_manager.get_preference("default_wallet")
        if default_wallet and default_wallet in [w["name"] for w in wallet_manager.list_wallets()]:
            wallet_manager.set_active_wallet(default_wallet)
            
    except Exception as e:
        print(f"Failed to initialize configuration: {e}")


# Call initialization on module load
asyncio.create_task(initialize_config())