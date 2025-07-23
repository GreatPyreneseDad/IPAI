#!/usr/bin/env python3
"""
Database setup script for IPAI

This script:
1. Creates the database if it doesn't exist
2. Runs all migrations
3. Seeds initial data (optional)
"""

import os
import sys
import asyncio
from pathlib import Path
from sqlalchemy import create_engine, text
from sqlalchemy.exc import ProgrammingError
from alembic.config import Config
from alembic import command

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.core.config import get_settings
from src.models.database_models import Base


def create_database():
    """Create the database if it doesn't exist"""
    settings = get_settings()
    
    # Parse database URL
    db_url = settings.DATABASE_URL
    if db_url.startswith("postgresql://"):
        db_url = db_url.replace("postgresql://", "postgresql+psycopg2://")
    
    # Extract database name
    parts = db_url.split("/")
    db_name = parts[-1].split("?")[0]
    base_url = "/".join(parts[:-1])
    
    # Connect to postgres database to create our database
    engine = create_engine(f"{base_url}/postgres")
    
    try:
        with engine.connect() as conn:
            conn.execute(text("COMMIT"))  # Exit any transaction
            exists = conn.execute(
                text(f"SELECT 1 FROM pg_database WHERE datname = '{db_name}'")
            ).fetchone()
            
            if not exists:
                conn.execute(text(f"CREATE DATABASE {db_name}"))
                print(f"✓ Created database: {db_name}")
            else:
                print(f"✓ Database already exists: {db_name}")
                
    except Exception as e:
        print(f"✗ Error creating database: {e}")
        raise
    finally:
        engine.dispose()


def run_migrations():
    """Run Alembic migrations"""
    alembic_cfg = Config("alembic.ini")
    
    try:
        # Upgrade to latest revision
        command.upgrade(alembic_cfg, "head")
        print("✓ Migrations completed successfully")
    except Exception as e:
        print(f"✗ Error running migrations: {e}")
        raise


async def seed_initial_data():
    """Seed initial data (optional)"""
    from src.core.database import Database
    
    db = Database()
    await db.connect()
    
    try:
        # Add any initial data here
        # For example, default achievements, system configurations, etc.
        
        print("✓ Initial data seeded (if any)")
    except Exception as e:
        print(f"✗ Error seeding data: {e}")
        raise
    finally:
        await db.disconnect()


def main():
    """Main setup function"""
    print("IPAI Database Setup")
    print("=" * 50)
    
    try:
        # Step 1: Create database
        print("\n1. Creating database...")
        create_database()
        
        # Step 2: Run migrations
        print("\n2. Running migrations...")
        run_migrations()
        
        # Step 3: Seed data (optional)
        print("\n3. Seeding initial data...")
        asyncio.run(seed_initial_data())
        
        print("\n" + "=" * 50)
        print("✓ Database setup completed successfully!")
        
    except Exception as e:
        print("\n" + "=" * 50)
        print(f"✗ Database setup failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()