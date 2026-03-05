"""
Security Manager

Enhanced security management for the IPAI system with
defensive security practices and comprehensive protection.
"""

import os
import secrets
import hashlib
import hmac
import base64
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt
import jwt
from passlib.context import CryptContext
from passlib.hash import bcrypt
import logging
import redis
import json

logger = logging.getLogger(__name__)


class SecurityManager:
    """Centralized security management with defensive practices"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        self.pwd_context = CryptContext(
            schemes=["bcrypt"],
            deprecated="auto",
            bcrypt__rounds=12  # Increased rounds for better security
        )
        self._init_encryption()
        self._init_rate_limiting()
        self._init_redis()
        
        # Security event logging
        self.security_events = []
        
    def _default_config(self) -> Dict:
        """Default security configuration with defensive settings"""
        return {
            'jwt_secret': os.getenv('JWT_SECRET', self._generate_secure_key()),
            'jwt_algorithm': 'HS256',
            'jwt_expiration_hours': 24,
            'jwt_refresh_expiration_days': 7,
            'encryption_key': os.getenv('ENCRYPTION_KEY'),
            'password_min_length': 12,  # Increased minimum length
            'password_require_uppercase': True,
            'password_require_lowercase': True,
            'password_require_digits': True,
            'password_require_special': True,
            'max_login_attempts': 5,
            'lockout_duration_minutes': 30,
            'session_timeout_hours': 8,
            'enable_mfa': False,  # Future enhancement
            'enable_audit_logging': True,
            'password_history_length': 10,
            'force_password_change_days': 90
        }
    
    def _generate_secure_key(self) -> str:
        """Generate cryptographically secure key"""
        return base64.urlsafe_b64encode(secrets.token_bytes(32)).decode()
    
    def _init_encryption(self):
        """Initialize encryption components with secure defaults"""
        if not self.config['encryption_key']:
            # Generate new key and warn user to save it
            self.config['encryption_key'] = Fernet.generate_key()
            logger.warning("Generated new encryption key. Please save it securely in your environment variables as ENCRYPTION_KEY. The key has been generated but NOT logged for security reasons.")
        
        if isinstance(self.config['encryption_key'], str):
            self.config['encryption_key'] = self.config['encryption_key'].encode()
        
        self.cipher = Fernet(self.config['encryption_key'])
        
        # Initialize key derivation for additional security
        self.salt = os.getenv('SALT', secrets.token_bytes(16))
        if isinstance(self.salt, str):
            self.salt = self.salt.encode()
    
    def _init_rate_limiting(self):
        """Initialize rate limiting for security events"""
        self.login_attempts = {}
        self.failed_attempts = {}
        self.locked_accounts = {}
    
    def _init_redis(self):
        """Initialize Redis connection for token blacklist"""
        try:
            redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
            self.redis_client = redis.from_url(redis_url, decode_responses=True)
            # Test connection
            self.redis_client.ping()
            logger.info("Redis connection established for token blacklist")
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}. Token revocation will use in-memory storage.")
            self.redis_client = None
            # Fallback to in-memory storage
            self.token_blacklist = {}
    
    # Password Security
    
    def hash_password(self, password: str) -> str:
        """Hash password with enhanced security"""
        self._validate_password_strength(password)
        
        # Use bcrypt with high cost factor
        hashed = self.pwd_context.hash(password)
        
        # Log security event
        self._log_security_event('password_hashed', {'method': 'bcrypt'})
        
        return hashed
    
    def verify_password(self, plain_password: str, hashed_password: str, user_id: str = None) -> bool:
        """Verify password with rate limiting and security logging"""
        
        # Check if account is locked
        if user_id and self._is_account_locked(user_id):
            self._log_security_event('login_attempt_locked_account', {'user_id': user_id})
            return False
        
        # Verify password
        is_valid = self.pwd_context.verify(plain_password, hashed_password)
        
        if user_id:
            if is_valid:
                self._reset_login_attempts(user_id)
                self._log_security_event('login_success', {'user_id': user_id})
            else:
                self._record_failed_attempt(user_id)
                self._log_security_event('login_failed', {'user_id': user_id})
        
        return is_valid
    
    def _validate_password_strength(self, password: str) -> bool:
        """Validate password meets security requirements"""
        errors = []
        
        if len(password) < self.config['password_min_length']:
            errors.append(f"Password must be at least {self.config['password_min_length']} characters")
        
        if self.config['password_require_uppercase'] and not any(c.isupper() for c in password):
            errors.append("Password must contain at least one uppercase letter")
        
        if self.config['password_require_lowercase'] and not any(c.islower() for c in password):
            errors.append("Password must contain at least one lowercase letter")
        
        if self.config['password_require_digits'] and not any(c.isdigit() for c in password):
            errors.append("Password must contain at least one digit")
        
        if self.config['password_require_special'] and not any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password):
            errors.append("Password must contain at least one special character")
        
        # Check for common weak passwords
        weak_patterns = ['password', '123456', 'qwerty', 'admin', 'user']
        if any(pattern in password.lower() for pattern in weak_patterns):
            errors.append("Password contains common weak patterns")
        
        if errors:
            raise ValueError("; ".join(errors))
        
        return True
    
    def generate_secure_password(self, length: int = 16) -> str:
        """Generate cryptographically secure password"""
        import string
        
        # Ensure minimum requirements
        chars = []
        chars.append(secrets.choice(string.ascii_uppercase))
        chars.append(secrets.choice(string.ascii_lowercase))
        chars.append(secrets.choice(string.digits))
        chars.append(secrets.choice("!@#$%^&*()_+-="))
        
        # Fill remaining length
        all_chars = string.ascii_letters + string.digits + "!@#$%^&*()_+-="
        for _ in range(length - 4):
            chars.append(secrets.choice(all_chars))
        
        # Shuffle to avoid predictable patterns
        secrets.SystemRandom().shuffle(chars)
        
        return ''.join(chars)
    
    # JWT Token Security
    
    def create_access_token(self, user_id: str, additional_claims: Optional[Dict] = None) -> str:
        """Create secure JWT access token"""
        now = datetime.utcnow()
        
        claims = {
            'sub': user_id,
            'iat': now,
            'exp': now + timedelta(hours=self.config['jwt_expiration_hours']),
            'type': 'access',
            'jti': secrets.token_urlsafe(16),  # JWT ID for token revocation
            'aud': 'ipai-api',  # Audience
            'iss': 'ipai-system'  # Issuer
        }
        
        if additional_claims:
            # Validate additional claims don't override security claims
            forbidden_claims = {'sub', 'iat', 'exp', 'jti', 'aud', 'iss'}
            for claim in additional_claims:
                if claim not in forbidden_claims:
                    claims[claim] = additional_claims[claim]
        
        token = jwt.encode(claims, self.config['jwt_secret'], algorithm=self.config['jwt_algorithm'])
        
        self._log_security_event('token_created', {
            'user_id': user_id,
            'token_type': 'access',
            'expires_at': claims['exp'].isoformat()
        })
        
        return token
    
    def create_refresh_token(self, user_id: str) -> str:
        """Create secure refresh token"""
        now = datetime.utcnow()
        
        claims = {
            'sub': user_id,
            'iat': now,
            'exp': now + timedelta(days=self.config['jwt_refresh_expiration_days']),
            'type': 'refresh',
            'jti': secrets.token_urlsafe(16),
            'aud': 'ipai-api',
            'iss': 'ipai-system'
        }
        
        token = jwt.encode(claims, self.config['jwt_secret'], algorithm=self.config['jwt_algorithm'])
        
        self._log_security_event('token_created', {
            'user_id': user_id,
            'token_type': 'refresh',
            'expires_at': claims['exp'].isoformat()
        })
        
        return token
    
    def verify_token(self, token: str, token_type: str = 'access') -> Optional[Dict]:
        """Verify JWT token with enhanced security checks"""
        try:
            payload = jwt.decode(
                token,
                self.config['jwt_secret'],
                algorithms=[self.config['jwt_algorithm']],
                audience='ipai-api',
                issuer='ipai-system'
            )
            
            # Verify token type
            if payload.get('type') != token_type:
                self._log_security_event('token_type_mismatch', {
                    'expected': token_type,
                    'actual': payload.get('type')
                })
                return None
            
            # Check if token is revoked (would need to implement token blacklist)
            jti = payload.get('jti')
            if jti and self._is_token_revoked(jti):
                self._log_security_event('revoked_token_used', {'jti': jti})
                return None
            
            self._log_security_event('token_verified', {
                'user_id': payload.get('sub'),
                'token_type': token_type
            })
            
            return payload
            
        except jwt.ExpiredSignatureError:
            self._log_security_event('token_expired', {'token_type': token_type})
            return None
        except jwt.InvalidTokenError as e:
            self._log_security_event('token_invalid', {
                'error': str(e),
                'token_type': token_type
            })
            return None
    
    def revoke_token(self, token: str) -> bool:
        """Revoke a JWT token (add to blacklist)"""
        try:
            payload = jwt.decode(
                token,
                self.config['jwt_secret'],
                algorithms=[self.config['jwt_algorithm']],
                options={"verify_exp": False}  # Allow expired tokens for revocation
            )
            
            jti = payload.get('jti')
            if jti:
                # Add to revocation list (implement persistent storage)
                self._add_to_token_blacklist(jti, payload.get('exp'))
                
                self._log_security_event('token_revoked', {
                    'jti': jti,
                    'user_id': payload.get('sub')
                })
                
                return True
                
        except jwt.InvalidTokenError:
            pass
        
        return False
    
    # Data Encryption
    
    def encrypt_sensitive_data(self, data: str) -> str:
        """Encrypt sensitive data with additional metadata"""
        if not isinstance(data, str):
            data = str(data)
        
        # Add timestamp and checksum for integrity
        timestamp = datetime.utcnow().isoformat()
        checksum = hashlib.sha256(data.encode()).hexdigest()[:16]
        
        payload = f"{timestamp}|{checksum}|{data}"
        
        encrypted = self.cipher.encrypt(payload.encode())
        
        self._log_security_event('data_encrypted', {'data_length': len(data)})
        
        return base64.urlsafe_b64encode(encrypted).decode()
    
    def decrypt_sensitive_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data with integrity verification"""
        try:
            encrypted_bytes = base64.urlsafe_b64decode(encrypted_data.encode())
            decrypted_payload = self.cipher.decrypt(encrypted_bytes).decode()
            
            # Parse payload
            parts = decrypted_payload.split('|', 2)
            if len(parts) != 3:
                raise ValueError("Invalid encrypted data format")
            
            timestamp_str, stored_checksum, data = parts
            
            # Verify checksum
            calculated_checksum = hashlib.sha256(data.encode()).hexdigest()[:16]
            if not hmac.compare_digest(stored_checksum, calculated_checksum):
                raise ValueError("Data integrity check failed")
            
            # Check age (optional - implement data expiration)
            timestamp = datetime.fromisoformat(timestamp_str)
            age_days = (datetime.utcnow() - timestamp).days
            
            self._log_security_event('data_decrypted', {
                'data_length': len(data),
                'age_days': age_days
            })
            
            return data
            
        except Exception as e:
            self._log_security_event('decryption_failed', {'error': str(e)})
            raise ValueError("Failed to decrypt data")
    
    # Account Security
    
    def _is_account_locked(self, user_id: str) -> bool:
        """Check if account is locked due to failed attempts"""
        if user_id not in self.locked_accounts:
            return False
        
        lock_time = self.locked_accounts[user_id]
        lockout_duration = timedelta(minutes=self.config['lockout_duration_minutes'])
        
        if datetime.utcnow() - lock_time > lockout_duration:
            # Unlock account
            del self.locked_accounts[user_id]
            self._log_security_event('account_unlocked', {'user_id': user_id})
            return False
        
        return True
    
    def _record_failed_attempt(self, user_id: str):
        """Record failed login attempt"""
        if user_id not in self.failed_attempts:
            self.failed_attempts[user_id] = []
        
        self.failed_attempts[user_id].append(datetime.utcnow())
        
        # Remove old attempts (older than lockout duration)
        cutoff = datetime.utcnow() - timedelta(minutes=self.config['lockout_duration_minutes'])
        self.failed_attempts[user_id] = [
            attempt for attempt in self.failed_attempts[user_id]
            if attempt > cutoff
        ]
        
        # Check if account should be locked
        if len(self.failed_attempts[user_id]) >= self.config['max_login_attempts']:
            self.locked_accounts[user_id] = datetime.utcnow()
            self._log_security_event('account_locked', {
                'user_id': user_id,
                'failed_attempts': len(self.failed_attempts[user_id])
            })
    
    def _reset_login_attempts(self, user_id: str):
        """Reset failed login attempts for user"""
        if user_id in self.failed_attempts:
            del self.failed_attempts[user_id]
        if user_id in self.locked_accounts:
            del self.locked_accounts[user_id]
    
    # Token Management
    
    def _is_token_revoked(self, jti: str) -> bool:
        """Check if token is revoked"""
        if self.redis_client:
            try:
                # Check Redis blacklist
                key = f"token_blacklist:{jti}"
                return self.redis_client.exists(key) > 0
            except Exception as e:
                logger.error(f"Redis error checking token revocation: {e}")
                # Fallback to in-memory check
                return jti in self.token_blacklist
        else:
            # Use in-memory blacklist
            return jti in self.token_blacklist
    
    def _add_to_token_blacklist(self, jti: str, exp_timestamp: int):
        """Add token to blacklist with expiration"""
        if self.redis_client:
            try:
                # Add to Redis with expiration
                key = f"token_blacklist:{jti}"
                # Calculate TTL based on token expiration
                ttl = max(exp_timestamp - int(datetime.utcnow().timestamp()), 0)
                if ttl > 0:
                    # Store token data with expiration
                    token_data = {
                        'jti': jti,
                        'revoked_at': datetime.utcnow().isoformat(),
                        'exp': exp_timestamp
                    }
                    self.redis_client.setex(key, ttl, json.dumps(token_data))
                    logger.info(f"Token {jti} added to Redis blacklist with TTL {ttl}s")
            except Exception as e:
                logger.error(f"Redis error adding token to blacklist: {e}")
                # Fallback to in-memory storage
                self.token_blacklist[jti] = {
                    'exp': exp_timestamp,
                    'revoked_at': datetime.utcnow()
                }
        else:
            # Use in-memory blacklist
            self.token_blacklist[jti] = {
                'exp': exp_timestamp,
                'revoked_at': datetime.utcnow()
            }
        
        self._log_security_event('token_blacklisted', {
            'jti': jti,
            'exp_timestamp': exp_timestamp
        })
    
    def _cleanup_expired_tokens(self):
        """Clean up expired tokens from in-memory blacklist"""
        if not self.redis_client and hasattr(self, 'token_blacklist'):
            current_time = datetime.utcnow()
            expired_tokens = []
            
            for jti, token_info in self.token_blacklist.items():
                if 'exp' in token_info:
                    exp_time = datetime.utcfromtimestamp(token_info['exp'])
                    if current_time > exp_time:
                        expired_tokens.append(jti)
            
            for jti in expired_tokens:
                del self.token_blacklist[jti]
                
            if expired_tokens:
                logger.info(f"Cleaned up {len(expired_tokens)} expired tokens from blacklist")
    
    def get_revoked_tokens_count(self) -> int:
        """Get count of revoked tokens"""
        if self.redis_client:
            try:
                # Count Redis keys with pattern
                keys = self.redis_client.keys("token_blacklist:*")
                return len(keys)
            except Exception as e:
                logger.error(f"Redis error counting revoked tokens: {e}")
                return len(self.token_blacklist) if hasattr(self, 'token_blacklist') else 0
        else:
            return len(self.token_blacklist) if hasattr(self, 'token_blacklist') else 0
    
    # Security Event Logging
    
    def _log_security_event(self, event_type: str, details: Dict[str, Any]):
        """Log security events for monitoring and auditing"""
        if not self.config['enable_audit_logging']:
            return
        
        event = {
            'timestamp': datetime.utcnow().isoformat(),
            'event_type': event_type,
            'details': details,
            'source': 'security_manager'
        }
        
        # Store in memory (implement persistent storage for production)
        self.security_events.append(event)
        
        # Keep only recent events in memory
        if len(self.security_events) > 1000:
            self.security_events = self.security_events[-1000:]
        
        # Log to application logger
        logger.info(f"Security event: {event_type}", extra={'security_event': event})
    
    def get_security_events(self, limit: int = 100, event_type: str = None) -> List[Dict]:
        """Get recent security events"""
        events = self.security_events[-limit:]
        
        if event_type:
            events = [e for e in events if e['event_type'] == event_type]
        
        return events
    
    # Utility Methods
    
    def generate_csrf_token(self, session_id: str) -> str:
        """Generate CSRF token for session"""
        timestamp = str(int(datetime.utcnow().timestamp()))
        data = f"{session_id}:{timestamp}"
        
        signature = hmac.new(
            self.config['jwt_secret'].encode(),
            data.encode(),
            hashlib.sha256
        ).hexdigest()
        
        token = base64.urlsafe_b64encode(f"{data}:{signature}".encode()).decode()
        
        self._log_security_event('csrf_token_generated', {'session_id': session_id})
        
        return token
    
    def validate_csrf_token(self, token: str, session_id: str, max_age: int = 3600) -> bool:
        """Validate CSRF token"""
        try:
            decoded = base64.urlsafe_b64decode(token.encode()).decode()
            parts = decoded.split(':')
            
            if len(parts) != 3:
                return False
            
            token_session_id, timestamp, signature = parts
            
            # Verify session ID
            if token_session_id != session_id:
                return False
            
            # Check timestamp
            if int(datetime.utcnow().timestamp()) - int(timestamp) > max_age:
                return False
            
            # Verify signature
            data = f"{token_session_id}:{timestamp}"
            expected_signature = hmac.new(
                self.config['jwt_secret'].encode(),
                data.encode(),
                hashlib.sha256
            ).hexdigest()
            
            is_valid = hmac.compare_digest(signature, expected_signature)
            
            self._log_security_event('csrf_token_validated', {
                'session_id': session_id,
                'valid': is_valid
            })
            
            return is_valid
            
        except Exception as e:
            self._log_security_event('csrf_validation_error', {
                'error': str(e),
                'session_id': session_id
            })
            return False
    
    def hash_api_key(self, api_key: str) -> str:
        """Hash API key for secure storage"""
        # Use scrypt for API key hashing (more secure than bcrypt for this use case)
        salt = secrets.token_bytes(32)
        kdf = Scrypt(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            n=2**14,
            r=8,
            p=1,
        )
        key = kdf.derive(api_key.encode())
        
        # Combine salt and key for storage
        return base64.urlsafe_b64encode(salt + key).decode()
    
    def verify_api_key(self, api_key: str, hashed_key: str) -> bool:
        """Verify API key against hash"""
        try:
            decoded = base64.urlsafe_b64decode(hashed_key.encode())
            salt = decoded[:32]
            stored_key = decoded[32:]
            
            kdf = Scrypt(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                n=2**14,
                r=8,
                p=1,
            )
            
            try:
                kdf.verify(api_key.encode(), stored_key)
                return True
            except:
                return False
                
        except Exception:
            return False
    
    def get_security_summary(self) -> Dict[str, Any]:
        """Get security configuration summary"""
        # Clean up expired tokens periodically
        self._cleanup_expired_tokens()
        
        return {
            'password_requirements': {
                'min_length': self.config['password_min_length'],
                'require_uppercase': self.config['password_require_uppercase'],
                'require_lowercase': self.config['password_require_lowercase'],
                'require_digits': self.config['password_require_digits'],
                'require_special': self.config['password_require_special']
            },
            'authentication': {
                'max_login_attempts': self.config['max_login_attempts'],
                'lockout_duration_minutes': self.config['lockout_duration_minutes'],
                'jwt_expiration_hours': self.config['jwt_expiration_hours'],
                'session_timeout_hours': self.config['session_timeout_hours']
            },
            'security_features': {
                'encryption_enabled': bool(self.config['encryption_key']),
                'audit_logging_enabled': self.config['enable_audit_logging'],
                'mfa_enabled': self.config['enable_mfa'],
                'token_revocation_enabled': self.redis_client is not None or hasattr(self, 'token_blacklist'),
                'token_revocation_backend': 'redis' if self.redis_client else 'memory'
            },
            'recent_events_count': len(self.security_events),
            'locked_accounts_count': len(self.locked_accounts),
            'revoked_tokens_count': self.get_revoked_tokens_count()
        }