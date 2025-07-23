"""Initial migration

Revision ID: 001
Revises: 
Create Date: 2025-07-23

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
import uuid

# revision identifiers, used by Alembic.
revision = '001'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create enum types
    op.execute("CREATE TYPE userrole AS ENUM ('user', 'premium', 'admin', 'moderator')")
    op.execute("CREATE TYPE coherencelevel AS ENUM ('critical', 'low', 'moderate', 'high', 'optimal')")
    op.execute("CREATE TYPE interactiontype AS ENUM ('chat', 'assessment', 'meditation', 'journal', 'analysis')")
    op.execute("CREATE TYPE inferencestatus AS ENUM ('pending', 'verified', 'rejected', 'consensus')")
    
    # Create users table
    op.create_table('users',
        sa.Column('id', sa.String(), nullable=False),
        sa.Column('email', sa.String(length=255), nullable=False),
        sa.Column('username', sa.String(length=50), nullable=False),
        sa.Column('hashed_password', sa.String(length=255), nullable=False),
        sa.Column('full_name', sa.String(length=100), nullable=True),
        sa.Column('bio', sa.Text(), nullable=True),
        sa.Column('avatar_url', sa.String(length=500), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=True),
        sa.Column('is_verified', sa.Boolean(), nullable=True),
        sa.Column('role', sa.Enum('USER', 'PREMIUM', 'ADMIN', 'MODERATOR', name='userrole'), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.Column('last_login', sa.DateTime(), nullable=True),
        sa.Column('current_coherence_score', sa.Float(), nullable=True),
        sa.Column('coherence_level', sa.Enum('CRITICAL', 'LOW', 'MODERATE', 'HIGH', 'OPTIMAL', name='coherencelevel'), nullable=True),
        sa.Column('preferences', sa.JSON(), nullable=True),
        sa.Column('notification_settings', sa.JSON(), nullable=True),
        sa.Column('wallet_address', sa.String(length=42), nullable=True),
        sa.Column('ipai_identity_token_id', sa.Integer(), nullable=True),
        sa.Column('sage_balance', sa.Float(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('email'),
        sa.UniqueConstraint('username'),
        sa.UniqueConstraint('wallet_address')
    )
    op.create_index('idx_user_coherence', 'users', ['current_coherence_score', 'coherence_level'], unique=False)
    op.create_index('idx_user_email_active', 'users', ['email', 'is_active'], unique=False)
    op.create_index(op.f('ix_users_email'), 'users', ['email'], unique=True)
    op.create_index(op.f('ix_users_username'), 'users', ['username'], unique=True)
    
    # Create coherence_profiles table
    op.create_table('coherence_profiles',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.String(), nullable=False),
        sa.Column('psi', sa.Float(), nullable=False),
        sa.Column('rho', sa.Float(), nullable=False),
        sa.Column('q', sa.Float(), nullable=False),
        sa.Column('f', sa.Float(), nullable=False),
        sa.Column('coherence_score', sa.Float(), nullable=False),
        sa.Column('soul_echo', sa.Float(), nullable=False),
        sa.Column('level', sa.Enum('CRITICAL', 'LOW', 'MODERATE', 'HIGH', 'OPTIMAL', name='coherencelevel'), nullable=False),
        sa.Column('k_m', sa.Float(), nullable=True),
        sa.Column('k_i', sa.Float(), nullable=True),
        sa.Column('calculated_at', sa.DateTime(), nullable=True),
        sa.Column('calculation_method', sa.String(length=50), nullable=True),
        sa.Column('confidence_score', sa.Float(), nullable=True),
        sa.Column('risk_factors', sa.JSON(), nullable=True),
        sa.Column('growth_potential', sa.Float(), nullable=True),
        sa.Column('stability_index', sa.Float(), nullable=True),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_coherence_score', 'coherence_profiles', ['coherence_score', 'level'], unique=False)
    op.create_index('idx_coherence_user_time', 'coherence_profiles', ['user_id', 'calculated_at'], unique=False)
    
    # Create achievements table
    op.create_table('achievements',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('name', sa.String(length=100), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('icon_url', sa.String(length=500), nullable=True),
        sa.Column('requirement_type', sa.String(length=50), nullable=True),
        sa.Column('requirement_value', sa.JSON(), nullable=True),
        sa.Column('sage_reward', sa.Float(), nullable=True),
        sa.Column('badge_color', sa.String(length=7), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('name')
    )
    
    # Create api_keys table
    op.create_table('api_keys',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.String(), nullable=False),
        sa.Column('key_hash', sa.String(length=255), nullable=False),
        sa.Column('name', sa.String(length=100), nullable=False),
        sa.Column('scopes', sa.JSON(), nullable=True),
        sa.Column('rate_limit', sa.Integer(), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=True),
        sa.Column('last_used', sa.DateTime(), nullable=True),
        sa.Column('usage_count', sa.Integer(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('expires_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('key_hash')
    )
    op.create_index('idx_api_key_hash', 'api_keys', ['key_hash'], unique=False)
    op.create_index('idx_api_key_user', 'api_keys', ['user_id', 'is_active'], unique=False)
    
    # Create assessments table
    op.create_table('assessments',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.String(), nullable=False),
        sa.Column('assessment_type', sa.String(length=50), nullable=False),
        sa.Column('version', sa.String(length=10), nullable=True),
        sa.Column('questions', sa.JSON(), nullable=False),
        sa.Column('responses', sa.JSON(), nullable=False),
        sa.Column('raw_scores', sa.JSON(), nullable=False),
        sa.Column('normalized_scores', sa.JSON(), nullable=False),
        sa.Column('psi_score', sa.Float(), nullable=True),
        sa.Column('rho_score', sa.Float(), nullable=True),
        sa.Column('q_score', sa.Float(), nullable=True),
        sa.Column('f_score', sa.Float(), nullable=True),
        sa.Column('suggested_k_m', sa.Float(), nullable=True),
        sa.Column('suggested_k_i', sa.Float(), nullable=True),
        sa.Column('completed_at', sa.DateTime(), nullable=True),
        sa.Column('time_taken_seconds', sa.Integer(), nullable=True),
        sa.Column('completion_rate', sa.Float(), nullable=True),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_assessment_user_time', 'assessments', ['user_id', 'completed_at'], unique=False)
    
    # Create blockchain_contracts table
    op.create_table('blockchain_contracts',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('name', sa.String(length=50), nullable=False),
        sa.Column('network', sa.String(length=50), nullable=False),
        sa.Column('address', sa.String(length=42), nullable=False),
        sa.Column('abi', sa.JSON(), nullable=False),
        sa.Column('is_active', sa.Boolean(), nullable=True),
        sa.Column('deployed_at', sa.DateTime(), nullable=True),
        sa.Column('deployed_by', sa.String(length=42), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('name', 'network', name='_name_network_uc')
    )
    op.create_index('idx_contract_network', 'blockchain_contracts', ['network', 'is_active'], unique=False)
    
    # Create notifications table
    op.create_table('notifications',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.String(), nullable=False),
        sa.Column('type', sa.String(length=50), nullable=False),
        sa.Column('title', sa.String(length=200), nullable=False),
        sa.Column('message', sa.Text(), nullable=False),
        sa.Column('data', sa.JSON(), nullable=True),
        sa.Column('is_read', sa.Boolean(), nullable=True),
        sa.Column('read_at', sa.DateTime(), nullable=True),
        sa.Column('channels', sa.JSON(), nullable=True),
        sa.Column('delivered_channels', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('expires_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_notification_created', 'notifications', ['created_at'], unique=False)
    op.create_index('idx_notification_user_unread', 'notifications', ['user_id', 'is_read'], unique=False)
    
    # Create system_metrics table
    op.create_table('system_metrics',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('metric_type', sa.String(length=50), nullable=False),
        sa.Column('metric_name', sa.String(length=100), nullable=False),
        sa.Column('value', sa.Float(), nullable=False),
        sa.Column('unit', sa.String(length=20), nullable=True),
        sa.Column('tags', sa.JSON(), nullable=True),
        sa.Column('recorded_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_metric_name', 'system_metrics', ['metric_name'], unique=False)
    op.create_index('idx_metric_type_time', 'system_metrics', ['metric_type', 'recorded_at'], unique=False)
    
    # Create user_interactions table
    op.create_table('user_interactions',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.String(), nullable=False),
        sa.Column('interaction_type', sa.Enum('CHAT', 'ASSESSMENT', 'MEDITATION', 'JOURNAL', 'ANALYSIS', name='interactiontype'), nullable=False),
        sa.Column('input_text', sa.Text(), nullable=True),
        sa.Column('output_text', sa.Text(), nullable=True),
        sa.Column('coherence_before', sa.Float(), nullable=True),
        sa.Column('coherence_after', sa.Float(), nullable=True),
        sa.Column('coherence_delta', sa.Float(), nullable=True),
        sa.Column('safety_score', sa.Float(), nullable=True),
        sa.Column('howlround_risk', sa.Float(), nullable=True),
        sa.Column('pressure_score', sa.Float(), nullable=True),
        sa.Column('intervention_triggered', sa.Boolean(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('duration_seconds', sa.Integer(), nullable=True),
        sa.Column('llm_provider', sa.String(length=50), nullable=True),
        sa.Column('llm_model', sa.String(length=100), nullable=True),
        sa.Column('block_hash', sa.String(length=66), nullable=True),
        sa.Column('transaction_hash', sa.String(length=66), nullable=True),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_interaction_type', 'user_interactions', ['interaction_type'], unique=False)
    op.create_index('idx_interaction_user_time', 'user_interactions', ['user_id', 'created_at'], unique=False)
    
    # Create inferences table
    op.create_table('inferences',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('interaction_id', sa.Integer(), nullable=False),
        sa.Column('inference_type', sa.String(length=50), nullable=False),
        sa.Column('content', sa.JSON(), nullable=False),
        sa.Column('confidence_score', sa.Float(), nullable=False),
        sa.Column('coherence_context', sa.JSON(), nullable=False),
        sa.Column('verification_hash', sa.String(length=66), nullable=False),
        sa.Column('status', sa.Enum('PENDING', 'VERIFIED', 'REJECTED', 'CONSENSUS', name='inferencestatus'), nullable=True),
        sa.Column('verified_at', sa.DateTime(), nullable=True),
        sa.Column('verified_by', sa.String(length=100), nullable=True),
        sa.Column('public_ledger_hash', sa.String(length=66), nullable=True),
        sa.Column('sage_reward', sa.Float(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['interaction_id'], ['user_interactions.id'], ),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('verification_hash')
    )
    op.create_index('idx_inference_hash', 'inferences', ['verification_hash'], unique=False)
    op.create_index('idx_inference_status', 'inferences', ['status'], unique=False)
    
    # Create user_achievements association table
    op.create_table('user_achievements',
        sa.Column('user_id', sa.String(), nullable=True),
        sa.Column('achievement_id', sa.Integer(), nullable=True),
        sa.Column('earned_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['achievement_id'], ['achievements.id'], ),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], )
    )


def downgrade() -> None:
    # Drop tables in reverse order
    op.drop_table('user_achievements')
    op.drop_table('inferences')
    op.drop_table('user_interactions')
    op.drop_table('system_metrics')
    op.drop_table('notifications')
    op.drop_table('blockchain_contracts')
    op.drop_table('assessments')
    op.drop_table('api_keys')
    op.drop_table('achievements')
    op.drop_table('coherence_profiles')
    op.drop_table('users')
    
    # Drop enum types
    op.execute('DROP TYPE IF EXISTS inferencestatus')
    op.execute('DROP TYPE IF EXISTS interactiontype')
    op.execute('DROP TYPE IF EXISTS coherencelevel')
    op.execute('DROP TYPE IF EXISTS userrole')