"""
Configuration Management

This module provides configuration management for the IPAI system
using environment variables and defaults.
"""

import os
from typing import List, Optional, Dict, Any
from pydantic import BaseSettings, Field, validator
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings"""
    
    # Application settings
    APP_NAME: str = Field("IPAI", description="Application name")
    APP_VERSION: str = Field("1.0.0", description="Application version")
    DEBUG: bool = Field(False, description="Debug mode")
    ENVIRONMENT: str = Field("development", description="Environment (development/staging/production)")
    
    # API settings
    API_HOST: str = Field("0.0.0.0", description="API host")
    API_PORT: int = Field(8000, description="API port")
    API_PREFIX: str = Field("/api/v1", description="API prefix")
    
    # Security settings
    SECRET_KEY: str = Field("your-secret-key-change-this", description="Secret key for JWT and encryption")
    JWT_SECRET: str = Field("", description="JWT secret key")
    JWT_ALGORITHM: str = Field("HS256", description="JWT algorithm")
    JWT_EXPIRATION_HOURS: int = Field(24, description="JWT expiration time in hours")
    ENCRYPTION_KEY: Optional[str] = Field(None, description="Encryption key for sensitive data")
    
    # Database settings
    DATABASE_URL: str = Field("postgresql://localhost/ipai", description="Database URL")
    DATABASE_POOL_SIZE: int = Field(10, description="Database connection pool size")
    DATABASE_MAX_OVERFLOW: int = Field(20, description="Database max overflow connections")
    DATABASE_ECHO: bool = Field(False, description="Echo SQL queries")
    
    # Redis settings (for caching and rate limiting)
    REDIS_URL: str = Field("redis://localhost:6379/0", description="Redis URL")
    REDIS_POOL_SIZE: int = Field(10, description="Redis connection pool size")
    
    # LLM settings (Ollama integration)
    OLLAMA_HOST: str = Field("http://localhost:11434", description="Ollama server host")
    OLLAMA_MODEL: str = Field("llama3.2:latest", description="Ollama model name")
    LLM_CONTEXT_LENGTH: int = Field(4096, description="LLM context length")
    LLM_MAX_TOKENS: int = Field(1024, description="LLM max tokens per response")
    LLM_TEMPERATURE: float = Field(0.7, description="LLM temperature")
    LLM_TOP_P: float = Field(0.9, description="LLM top-p sampling")
    LLM_TIMEOUT: float = Field(30.0, description="LLM response timeout in seconds")
    
    # Backward compatibility (deprecated)
    LLM_MODEL_PATH: str = Field("", description="Deprecated: Path to LLM model file")
    LLM_N_THREADS: int = Field(8, description="Deprecated: Number of threads for LLM")
    LLM_N_GPU_LAYERS: int = Field(35, description="Deprecated: Number of GPU layers for LLM")
    
    # Rate limiting settings
    RATE_LIMIT_PER_MINUTE: int = Field(60, description="Rate limit per minute")
    RATE_LIMIT_PER_HOUR: int = Field(1000, description="Rate limit per hour")
    RATE_LIMIT_BURST_SIZE: int = Field(10, description="Rate limit burst size")
    
    # CORS settings
    ALLOWED_ORIGINS: List[str] = Field(
        ["http://localhost:3000", "http://localhost:3001"],
        description="Allowed CORS origins"
    )
    
    # Blockchain settings
    BLOCKCHAIN_NETWORK: str = Field("polygon", description="Blockchain network")
    BLOCKCHAIN_RPC_URL: str = Field("", description="Blockchain RPC URL")
    BLOCKCHAIN_PRIVATE_KEY: Optional[str] = Field(None, description="Blockchain private key")
    CONTRACT_ADDRESS_GCT: Optional[str] = Field(None, description="GCT contract address")
    CONTRACT_ADDRESS_IDENTITY: Optional[str] = Field(None, description="Identity contract address")
    CONTRACT_ADDRESS_SAGE: Optional[str] = Field(None, description="SageCoin contract address")
    
    # Email settings
    EMAIL_SMTP_HOST: str = Field("smtp.gmail.com", description="SMTP host")
    EMAIL_SMTP_PORT: int = Field(587, description="SMTP port")
    EMAIL_SMTP_USERNAME: str = Field("", description="SMTP username")
    EMAIL_SMTP_PASSWORD: str = Field("", description="SMTP password")
    EMAIL_FROM_ADDRESS: str = Field("noreply@ipai.app", description="From email address")
    EMAIL_USE_TLS: bool = Field(True, description="Use TLS for email")
    
    # File storage settings
    UPLOAD_DIR: str = Field("uploads", description="Upload directory")
    MAX_UPLOAD_SIZE: int = Field(10 * 1024 * 1024, description="Max upload size (10MB)")
    ALLOWED_UPLOAD_TYPES: List[str] = Field(
        ["image/jpeg", "image/png", "image/gif", "application/pdf"],
        description="Allowed upload file types"
    )
    
    # Monitoring and logging
    LOG_LEVEL: str = Field("INFO", description="Log level")
    LOG_FORMAT: str = Field(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log format"
    )
    ENABLE_METRICS: bool = Field(True, description="Enable metrics collection")
    METRICS_PORT: int = Field(9090, description="Metrics server port")
    
    # Performance settings
    CACHE_TTL: int = Field(300, description="Default cache TTL in seconds")
    CACHE_MAX_SIZE: int = Field(1000, description="Max cache size")
    ENABLE_COMPRESSION: bool = Field(True, description="Enable response compression")
    
    # Feature flags
    ENABLE_COHERENCE_TRACKING: bool = Field(True, description="Enable coherence tracking")
    ENABLE_LLM_INTEGRATION: bool = Field(True, description="Enable LLM integration")
    ENABLE_BLOCKCHAIN_INTEGRATION: bool = Field(False, description="Enable blockchain integration")
    ENABLE_ASSESSMENT_SYSTEM: bool = Field(True, description="Enable assessment system")
    ENABLE_ANALYTICS: bool = Field(True, description="Enable analytics")
    ENABLE_RESEARCH_MODE: bool = Field(False, description="Enable research mode")
    
    # Development settings
    RELOAD: bool = Field(False, description="Auto-reload on file changes")
    WORKERS: int = Field(1, description="Number of worker processes")
    
    @validator("JWT_SECRET", pre=True)
    def set_jwt_secret(cls, v, values):
        """Set JWT secret from SECRET_KEY if not provided"""
        if not v:
            return values.get("SECRET_KEY", "fallback-secret")
        return v
    
    @validator("ENVIRONMENT")
    def validate_environment(cls, v):
        """Validate environment value"""
        allowed = ["development", "staging", "production"]
        if v not in allowed:
            raise ValueError(f"Environment must be one of: {allowed}")
        return v
    
    @validator("LOG_LEVEL")
    def validate_log_level(cls, v):
        """Validate log level"""
        allowed = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in allowed:
            raise ValueError(f"Log level must be one of: {allowed}")
        return v.upper()
    
    @property
    def is_production(self) -> bool:
        """Check if running in production"""
        return self.ENVIRONMENT == "production"
    
    @property
    def is_development(self) -> bool:
        """Check if running in development"""
        return self.ENVIRONMENT == "development"
    
    @property
    def database_config(self) -> Dict[str, Any]:
        """Get database configuration"""
        return {
            "url": self.DATABASE_URL,
            "pool_size": self.DATABASE_POOL_SIZE,
            "max_overflow": self.DATABASE_MAX_OVERFLOW,
            "echo": self.DATABASE_ECHO and self.DEBUG
        }
    
    @property
    def redis_config(self) -> Dict[str, Any]:
        """Get Redis configuration"""
        return {
            "url": self.REDIS_URL,
            "pool_size": self.REDIS_POOL_SIZE,
            "decode_responses": True
        }
    
    @property
    def llm_config(self) -> Dict[str, Any]:
        """Get LLM configuration"""
        return {
            "ollama_host": self.OLLAMA_HOST,
            "model_name": self.OLLAMA_MODEL,
            "context_length": self.LLM_CONTEXT_LENGTH,
            "max_tokens": self.LLM_MAX_TOKENS,
            "temperature": self.LLM_TEMPERATURE,
            "top_p": self.LLM_TOP_P,
            "response_timeout": self.LLM_TIMEOUT,
            "enable_streaming": True,
            "enable_coherence_checking": True,
            "enable_triadic_processing": True,
            "enable_safety_filtering": True,
            "crisis_intervention_enabled": True,
            "max_response_length": 2000
        }
    
    @property
    def security_config(self) -> Dict[str, Any]:
        """Get security configuration"""
        return {
            "secret_key": self.SECRET_KEY,
            "jwt_secret": self.JWT_SECRET,
            "jwt_algorithm": self.JWT_ALGORITHM,
            "jwt_expiration_hours": self.JWT_EXPIRATION_HOURS,
            "encryption_key": self.ENCRYPTION_KEY
        }
    
    @property
    def cors_config(self) -> Dict[str, Any]:
        """Get CORS configuration"""
        return {
            "allowed_origins": self.ALLOWED_ORIGINS,
            "allow_credentials": True,
            "allow_methods": ["GET", "POST", "PUT", "DELETE", "PATCH"],
            "allow_headers": ["*"]
        }
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


class DatabaseSettings(BaseSettings):
    """Database-specific settings"""
    
    POSTGRES_USER: str = Field("postgres", description="PostgreSQL username")
    POSTGRES_PASSWORD: str = Field("password", description="PostgreSQL password")
    POSTGRES_HOST: str = Field("localhost", description="PostgreSQL host")
    POSTGRES_PORT: int = Field(5432, description="PostgreSQL port")
    POSTGRES_DB: str = Field("ipai", description="PostgreSQL database name")
    
    @property
    def url(self) -> str:
        """Get database URL"""
        return f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
    
    class Config:
        env_file = ".env"
        env_prefix = "DATABASE_"


class RedisSettings(BaseSettings):
    """Redis-specific settings"""
    
    REDIS_HOST: str = Field("localhost", description="Redis host")
    REDIS_PORT: int = Field(6379, description="Redis port")
    REDIS_DB: int = Field(0, description="Redis database number")
    REDIS_PASSWORD: Optional[str] = Field(None, description="Redis password")
    
    @property
    def url(self) -> str:
        """Get Redis URL"""
        auth = f":{self.REDIS_PASSWORD}@" if self.REDIS_PASSWORD else ""
        return f"redis://{auth}{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"
    
    class Config:
        env_file = ".env"
        env_prefix = "REDIS_"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()


@lru_cache()
def get_database_settings() -> DatabaseSettings:
    """Get cached database settings instance"""
    return DatabaseSettings()


@lru_cache()
def get_redis_settings() -> RedisSettings:
    """Get cached Redis settings instance"""
    return RedisSettings()


# Environment-specific configurations
def get_development_settings() -> Settings:
    """Get development-specific settings"""
    settings = Settings()
    settings.DEBUG = True
    settings.ENVIRONMENT = "development"
    settings.LOG_LEVEL = "DEBUG"
    settings.DATABASE_ECHO = True
    settings.RELOAD = True
    return settings


def get_production_settings() -> Settings:
    """Get production-specific settings"""
    settings = Settings()
    settings.DEBUG = False
    settings.ENVIRONMENT = "production"
    settings.LOG_LEVEL = "INFO"
    settings.DATABASE_ECHO = False
    settings.RELOAD = False
    settings.WORKERS = 4
    return settings


def get_testing_settings() -> Settings:
    """Get testing-specific settings"""
    settings = Settings()
    settings.DEBUG = True
    settings.ENVIRONMENT = "testing"
    settings.LOG_LEVEL = "DEBUG"
    settings.DATABASE_URL = "postgresql://localhost/ipai_test"
    settings.REDIS_URL = "redis://localhost:6379/1"
    return settings


# Utility functions
def load_config_from_file(file_path: str) -> Dict[str, Any]:
    """Load configuration from JSON or YAML file"""
    import json
    
    try:
        with open(file_path, 'r') as f:
            if file_path.endswith('.json'):
                return json.load(f)
            elif file_path.endswith(('.yml', '.yaml')):
                import yaml
                return yaml.safe_load(f)
            else:
                raise ValueError("Unsupported config file format")
    except Exception as e:
        raise ValueError(f"Failed to load config from {file_path}: {e}")


def validate_settings(settings: Settings) -> List[str]:
    """Validate settings and return list of issues"""
    issues = []
    
    # Check required settings for production
    if settings.is_production:
        if settings.SECRET_KEY == "your-secret-key-change-this":
            issues.append("SECRET_KEY must be changed in production")
        
        if not settings.DATABASE_URL.startswith("postgresql://"):
            issues.append("DATABASE_URL must be PostgreSQL in production")
        
        if settings.DEBUG:
            issues.append("DEBUG should be False in production")
    
    # Check LLM availability
    if settings.ENABLE_LLM_INTEGRATION:
        # For Ollama integration, we'll check connectivity at runtime
        # For backwards compatibility, check model file if specified
        if settings.LLM_MODEL_PATH and not os.path.exists(settings.LLM_MODEL_PATH):
            issues.append(f"LLM model file not found: {settings.LLM_MODEL_PATH}")
        
        # Validate Ollama configuration
        if not settings.OLLAMA_HOST.startswith(('http://', 'https://')):
            issues.append(f"OLLAMA_HOST must be a valid HTTP(S) URL: {settings.OLLAMA_HOST}")
        
        if not settings.OLLAMA_MODEL:
            issues.append("OLLAMA_MODEL must be specified for LLM integration")
    
    # Check upload directory
    if not os.path.exists(settings.UPLOAD_DIR):
        try:
            os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
        except Exception as e:
            issues.append(f"Cannot create upload directory: {e}")
    
    return issues