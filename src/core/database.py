"""
Database Interface and Management

This module provides database connectivity and management
for the IPAI system with connection pooling and security.
"""

import logging
import asyncio
from typing import Dict, Any, Optional, List, AsyncGenerator
from contextlib import asynccontextmanager
from sqlalchemy import create_engine, MetaData, Table, Column, String, Integer, Float, DateTime, Boolean, Text, JSON
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, sessionmaker
from sqlalchemy.pool import QueuePool
from sqlalchemy.exc import SQLAlchemyError
from datetime import datetime
import json

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    """Base class for SQLAlchemy models"""
    pass


class Database:
    """Database connection and management"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        self.engine = None
        self.async_engine = None
        self.session_maker = None
        self.async_session_maker = None
        self.metadata = MetaData()
        self._connection_pool = None
        self._is_connected = False
        
    def _default_config(self) -> Dict:
        """Default database configuration"""
        return {
            'url': 'postgresql://localhost/ipai',
            'async_url': 'postgresql+asyncpg://localhost/ipai',
            'echo': False,
            'pool_size': 10,
            'max_overflow': 20,
            'pool_timeout': 30,
            'pool_recycle': 3600,
            'pool_pre_ping': True,
            'connect_args': {
                'connect_timeout': 10,
                'command_timeout': 30,
                'server_settings': {
                    'jit': 'off',  # Disable JIT for better performance consistency
                    'application_name': 'ipai-system'
                }
            }
        }
    
    async def connect(self):
        """Initialize database connections"""
        try:
            # Create async engine
            self.async_engine = create_async_engine(
                self.config['async_url'],
                echo=self.config['echo'],
                pool_size=self.config['pool_size'],
                max_overflow=self.config['max_overflow'],
                pool_timeout=self.config['pool_timeout'],
                pool_recycle=self.config['pool_recycle'],
                pool_pre_ping=self.config['pool_pre_ping'],
                connect_args=self.config['connect_args']
            )
            
            # Create sync engine for migrations and setup
            sync_url = self.config['url']
            self.engine = create_engine(
                sync_url,
                echo=self.config['echo'],
                pool_size=self.config['pool_size'],
                max_overflow=self.config['max_overflow'],
                pool_timeout=self.config['pool_timeout'],
                pool_recycle=self.config['pool_recycle'],
                pool_pre_ping=self.config['pool_pre_ping'],
                connect_args=self.config['connect_args']
            )
            
            # Create session makers
            self.async_session_maker = async_sessionmaker(
                self.async_engine,
                class_=AsyncSession,
                expire_on_commit=False
            )
            
            self.session_maker = sessionmaker(
                self.engine,
                expire_on_commit=False
            )
            
            # Test connection
            await self._test_connection()
            
            self._is_connected = True
            logger.info("Database connection established successfully")
            
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise
    
    async def _test_connection(self):
        """Test database connection"""
        async with self.async_session_maker() as session:
            await session.execute("SELECT 1")
    
    async def disconnect(self):
        """Close database connections"""
        if self.async_engine:
            await self.async_engine.dispose()
            self.async_engine = None
        
        if self.engine:
            self.engine.dispose()
            self.engine = None
        
        self._is_connected = False
        logger.info("Database connections closed")
    
    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get async database session"""
        if not self._is_connected:
            await self.connect()
        
        async with self.async_session_maker() as session:
            try:
                yield session
                await session.commit()
            except Exception as e:
                await session.rollback()
                logger.error(f"Database session error: {e}")
                raise
            finally:
                await session.close()
    
    async def execute_query(self, query: str, params: Optional[Dict] = None) -> List[Dict]:
        """Execute raw SQL query"""
        async with self.get_session() as session:
            result = await session.execute(query, params or {})
            return [dict(row) for row in result.fetchall()]
    
    async def execute_scalar(self, query: str, params: Optional[Dict] = None) -> Any:
        """Execute query and return scalar result"""
        async with self.get_session() as session:
            result = await session.execute(query, params or {})
            return result.scalar()
    
    async def create_tables(self):
        """Create database tables"""
        if not self.engine:
            await self.connect()
        
        try:
            # Create tables synchronously
            Base.metadata.create_all(self.engine)
            logger.info("Database tables created successfully")
        except Exception as e:
            logger.error(f"Failed to create tables: {e}")
            raise
    
    async def drop_tables(self):
        """Drop all database tables"""
        if not self.engine:
            await self.connect()
        
        try:
            Base.metadata.drop_all(self.engine)
            logger.info("Database tables dropped successfully")
        except Exception as e:
            logger.error(f"Failed to drop tables: {e}")
            raise
    
    async def get_table_info(self, table_name: str) -> Dict:
        """Get information about a table"""
        query = """
        SELECT column_name, data_type, is_nullable, column_default
        FROM information_schema.columns
        WHERE table_name = :table_name
        ORDER BY ordinal_position
        """
        
        columns = await self.execute_query(query, {'table_name': table_name})
        
        # Get row count
        count_query = f"SELECT COUNT(*) FROM {table_name}"
        try:
            row_count = await self.execute_scalar(count_query)
        except:
            row_count = None
        
        return {
            'table_name': table_name,
            'columns': columns,
            'row_count': row_count
        }
    
    async def get_database_stats(self) -> Dict:
        """Get database statistics"""
        stats = {}
        
        try:
            # Get connection stats
            if self.async_engine:
                pool = self.async_engine.pool
                stats['connection_pool'] = {
                    'size': pool.size(),
                    'checked_in': pool.checkedin(),
                    'checked_out': pool.checkedout(),
                    'overflow': pool.overflow(),
                    'invalid': pool.invalid()
                }
            
            # Get table stats
            query = """
            SELECT 
                schemaname,
                tablename,
                n_tup_ins as inserts,
                n_tup_upd as updates,
                n_tup_del as deletes,
                n_live_tup as live_tuples,
                n_dead_tup as dead_tuples
            FROM pg_stat_user_tables
            """
            
            table_stats = await self.execute_query(query)
            stats['tables'] = table_stats
            
            # Get database size
            size_query = "SELECT pg_size_pretty(pg_database_size(current_database()))"
            stats['database_size'] = await self.execute_scalar(size_query)
            
        except Exception as e:
            logger.error(f"Failed to get database stats: {e}")
            stats['error'] = str(e)
        
        return stats
    
    async def health_check(self) -> Dict:
        """Perform database health check"""
        try:
            start_time = datetime.utcnow()
            
            # Test connection
            await self._test_connection()
            
            # Test write operation
            async with self.get_session() as session:
                await session.execute("CREATE TEMP TABLE health_check (id INTEGER)")
                await session.execute("INSERT INTO health_check (id) VALUES (1)")
                result = await session.execute("SELECT id FROM health_check")
                assert result.scalar() == 1
                await session.execute("DROP TABLE health_check")
            
            response_time = (datetime.utcnow() - start_time).total_seconds()
            
            return {
                'status': 'healthy',
                'response_time': response_time,
                'timestamp': datetime.utcnow().isoformat(),
                'connected': self._is_connected
            }
            
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return {
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat(),
                'connected': False
            }


class CoherenceProfileDB:
    """Database operations for coherence profiles"""
    
    def __init__(self, db: Database):
        self.db = db
    
    async def create_profile(self, user_id: str, profile_data: Dict) -> str:
        """Create a new coherence profile"""
        query = """
        INSERT INTO coherence_profiles (user_id, profile_data, created_at, updated_at)
        VALUES (:user_id, :profile_data, :created_at, :updated_at)
        RETURNING id
        """
        
        now = datetime.utcnow()
        params = {
            'user_id': user_id,
            'profile_data': json.dumps(profile_data),
            'created_at': now,
            'updated_at': now
        }
        
        result = await self.db.execute_scalar(query, params)
        return str(result)
    
    async def get_profile(self, user_id: str) -> Optional[Dict]:
        """Get coherence profile for user"""
        query = """
        SELECT id, user_id, profile_data, created_at, updated_at
        FROM coherence_profiles
        WHERE user_id = :user_id
        ORDER BY updated_at DESC
        LIMIT 1
        """
        
        result = await self.db.execute_query(query, {'user_id': user_id})
        
        if result:
            profile = result[0]
            profile['profile_data'] = json.loads(profile['profile_data'])
            return profile
        
        return None
    
    async def update_profile(self, user_id: str, profile_data: Dict):
        """Update coherence profile"""
        query = """
        UPDATE coherence_profiles
        SET profile_data = :profile_data, updated_at = :updated_at
        WHERE user_id = :user_id
        """
        
        params = {
            'user_id': user_id,
            'profile_data': json.dumps(profile_data),
            'updated_at': datetime.utcnow()
        }
        
        await self.db.execute_query(query, params)
    
    async def get_trajectory(self, user_id: str, limit: int = 100) -> List[Dict]:
        """Get coherence trajectory for user"""
        query = """
        SELECT profile_data, created_at
        FROM coherence_profiles
        WHERE user_id = :user_id
        ORDER BY created_at DESC
        LIMIT :limit
        """
        
        results = await self.db.execute_query(query, {'user_id': user_id, 'limit': limit})
        
        trajectory = []
        for result in results:
            data = json.loads(result['profile_data'])
            trajectory.append({
                'timestamp': result['created_at'].isoformat(),
                'coherence_score': data.get('coherence_score', 0),
                'components': data.get('components', {}),
                'parameters': data.get('parameters', {})
            })
        
        return trajectory


class AssessmentDB:
    """Database operations for assessments"""
    
    def __init__(self, db: Database):
        self.db = db
    
    async def create_assessment(self, user_id: str, assessment_data: Dict) -> str:
        """Create a new assessment"""
        query = """
        INSERT INTO assessments (user_id, assessment_type, assessment_data, created_at)
        VALUES (:user_id, :assessment_type, :assessment_data, :created_at)
        RETURNING id
        """
        
        params = {
            'user_id': user_id,
            'assessment_type': assessment_data.get('type', 'unknown'),
            'assessment_data': json.dumps(assessment_data),
            'created_at': datetime.utcnow()
        }
        
        result = await self.db.execute_scalar(query, params)
        return str(result)
    
    async def get_assessments(self, user_id: str, assessment_type: str = None) -> List[Dict]:
        """Get assessments for user"""
        query = """
        SELECT id, assessment_type, assessment_data, created_at
        FROM assessments
        WHERE user_id = :user_id
        """
        
        params = {'user_id': user_id}
        
        if assessment_type:
            query += " AND assessment_type = :assessment_type"
            params['assessment_type'] = assessment_type
        
        query += " ORDER BY created_at DESC"
        
        results = await self.db.execute_query(query, params)
        
        assessments = []
        for result in results:
            assessment = {
                'id': result['id'],
                'type': result['assessment_type'],
                'data': json.loads(result['assessment_data']),
                'created_at': result['created_at'].isoformat()
            }
            assessments.append(assessment)
        
        return assessments


class AnalyticsDB:
    """Database operations for analytics"""
    
    def __init__(self, db: Database):
        self.db = db
    
    async def record_interaction(self, user_id: str, interaction_data: Dict):
        """Record user interaction for analytics"""
        query = """
        INSERT INTO user_interactions (user_id, interaction_type, interaction_data, timestamp)
        VALUES (:user_id, :interaction_type, :interaction_data, :timestamp)
        """
        
        params = {
            'user_id': user_id,
            'interaction_type': interaction_data.get('type', 'unknown'),
            'interaction_data': json.dumps(interaction_data),
            'timestamp': datetime.utcnow()
        }
        
        await self.db.execute_query(query, params)
    
    async def get_user_analytics(self, user_id: str, days: int = 30) -> Dict:
        """Get user analytics data"""
        query = """
        SELECT 
            DATE(timestamp) as date,
            interaction_type,
            COUNT(*) as count
        FROM user_interactions
        WHERE user_id = :user_id
        AND timestamp >= NOW() - INTERVAL ':days days'
        GROUP BY DATE(timestamp), interaction_type
        ORDER BY date DESC
        """
        
        results = await self.db.execute_query(query, {'user_id': user_id, 'days': days})
        
        # Process results into analytics format
        analytics = {
            'daily_interactions': {},
            'interaction_types': {},
            'total_interactions': 0
        }
        
        for result in results:
            date_str = result['date'].isoformat()
            interaction_type = result['interaction_type']
            count = result['count']
            
            if date_str not in analytics['daily_interactions']:
                analytics['daily_interactions'][date_str] = 0
            analytics['daily_interactions'][date_str] += count
            
            if interaction_type not in analytics['interaction_types']:
                analytics['interaction_types'][interaction_type] = 0
            analytics['interaction_types'][interaction_type] += count
            
            analytics['total_interactions'] += count
        
        return analytics


# Database table definitions
def create_tables_sql() -> List[str]:
    """SQL statements to create database tables"""
    return [
        """
        CREATE TABLE IF NOT EXISTS users (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            email VARCHAR(255) UNIQUE NOT NULL,
            password_hash VARCHAR(255) NOT NULL,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            is_active BOOLEAN DEFAULT TRUE,
            profile_data JSONB DEFAULT '{}'::jsonb
        )
        """,
        
        """
        CREATE TABLE IF NOT EXISTS coherence_profiles (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            profile_data JSONB NOT NULL,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        )
        """,
        
        """
        CREATE TABLE IF NOT EXISTS assessments (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            assessment_type VARCHAR(100) NOT NULL,
            assessment_data JSONB NOT NULL,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        )
        """,
        
        """
        CREATE TABLE IF NOT EXISTS user_interactions (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            interaction_type VARCHAR(100) NOT NULL,
            interaction_data JSONB NOT NULL,
            timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        )
        """,
        
        """
        CREATE TABLE IF NOT EXISTS coherence_calculations (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            calculation_data JSONB NOT NULL,
            coherence_score FLOAT NOT NULL,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        )
        """,
        
        """
        CREATE INDEX IF NOT EXISTS idx_coherence_profiles_user_id ON coherence_profiles(user_id);
        CREATE INDEX IF NOT EXISTS idx_coherence_profiles_updated_at ON coherence_profiles(updated_at);
        CREATE INDEX IF NOT EXISTS idx_assessments_user_id ON assessments(user_id);
        CREATE INDEX IF NOT EXISTS idx_assessments_type ON assessments(assessment_type);
        CREATE INDEX IF NOT EXISTS idx_user_interactions_user_id ON user_interactions(user_id);
        CREATE INDEX IF NOT EXISTS idx_user_interactions_timestamp ON user_interactions(timestamp);
        CREATE INDEX IF NOT EXISTS idx_coherence_calculations_user_id ON coherence_calculations(user_id);
        """
    ]


# Global database instance
database = Database()