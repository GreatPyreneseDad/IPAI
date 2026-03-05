"""
User Management API Endpoints

User profile management and preferences for the IPAI system
with privacy protection and coherence integration.
"""

import logging
from datetime import datetime
from typing import Dict, Any, Optional, List

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, EmailStr, Field, validator

from ...core.database import database, CoherenceProfileDB, AnalyticsDB
from ...models.user import User, UserPreferences
from .auth import get_current_user, get_optional_user

logger = logging.getLogger(__name__)

router = APIRouter()


# Request/Response Models
class UserProfileUpdate(BaseModel):
    """User profile update request"""
    first_name: Optional[str] = Field(None, min_length=1, max_length=50)
    last_name: Optional[str] = Field(None, min_length=1, max_length=50)
    bio: Optional[str] = Field(None, max_length=500)
    avatar_url: Optional[str] = None
    preferences: Optional[Dict[str, Any]] = None
    
    class Config:
        schema_extra = {
            "example": {
                "first_name": "John",
                "last_name": "Doe",
                "bio": "AI researcher interested in coherence theory",
                "preferences": {
                    "language": "en",
                    "timezone": "UTC",
                    "theme": "dark",
                    "notifications_enabled": True,
                    "coherence_tracking": True
                }
            }
        }


class UserPreferencesUpdate(BaseModel):
    """User preferences update request"""
    language: Optional[str] = Field(None, regex="^[a-z]{2}$")
    timezone: Optional[str] = None
    theme: Optional[str] = Field(None, regex="^(light|dark|auto)$")
    notifications_enabled: Optional[bool] = None
    coherence_tracking: Optional[bool] = None
    privacy_level: Optional[str] = Field(None, regex="^(public|private|restricted)$")
    data_sharing: Optional[bool] = None
    research_participation: Optional[bool] = None
    
    class Config:
        schema_extra = {
            "example": {
                "language": "en",
                "timezone": "America/New_York",
                "theme": "dark",
                "notifications_enabled": True,
                "coherence_tracking": True,
                "privacy_level": "private",
                "data_sharing": False,
                "research_participation": False
            }
        }


class UserProfile(BaseModel):
    """User profile response"""
    id: str
    email: str
    first_name: Optional[str]
    last_name: Optional[str]
    bio: Optional[str]
    avatar_url: Optional[str]
    created_at: datetime
    updated_at: datetime
    is_active: bool
    preferences: Dict[str, Any]
    coherence_summary: Optional[Dict[str, Any]] = None
    statistics: Optional[Dict[str, Any]] = None


class UserActivitySummary(BaseModel):
    """User activity summary"""
    user_id: str
    total_sessions: int
    total_interactions: int
    avg_coherence_score: float
    coherence_trend: str
    last_active: datetime
    achievements: List[str]


# Endpoints
@router.get("/profile", response_model=UserProfile)
async def get_user_profile(
    current_user: Dict[str, Any] = Depends(get_current_user),
    include_coherence: bool = Query(True, description="Include coherence summary"),
    include_stats: bool = Query(True, description="Include usage statistics")
):
    """Get current user's complete profile"""
    
    try:
        # Get user data
        async with database.get_session() as session:
            user_query = """
            SELECT id, email, first_name, last_name, created_at, updated_at,
                   is_active, profile_data
            FROM users
            WHERE id = :user_id
            """
            user_result = await database.execute_query(user_query, {"user_id": current_user["id"]})
            
            if not user_result:
                raise HTTPException(status_code=404, detail="User not found")
            
            user_data = user_result[0]
            profile_data = user_data.get("profile_data", {})
        
        # Get coherence summary if requested
        coherence_summary = None
        if include_coherence:
            coherence_db = CoherenceProfileDB(database)
            latest_profile = await coherence_db.get_profile(current_user["id"])
            
            if latest_profile:
                coherence_summary = {
                    "current_score": latest_profile["profile_data"].get("coherence_score"),
                    "components": latest_profile["profile_data"].get("components"),
                    "last_updated": latest_profile["updated_at"]
                }
        
        # Get usage statistics if requested
        statistics = None
        if include_stats:
            analytics_db = AnalyticsDB(database)
            user_analytics = await analytics_db.get_user_analytics(current_user["id"])
            
            statistics = {
                "total_interactions": user_analytics.get("total_interactions", 0),
                "daily_average": user_analytics.get("total_interactions", 0) / 30 if user_analytics.get("total_interactions") else 0,
                "interaction_types": user_analytics.get("interaction_types", {}),
                "most_active_day": max(user_analytics.get("daily_interactions", {}).items(), default=(None, 0), key=lambda x: x[1])[0]
            }
        
        return UserProfile(
            id=user_data["id"],
            email=user_data["email"],
            first_name=user_data.get("first_name"),
            last_name=user_data.get("last_name"),
            bio=profile_data.get("bio"),
            avatar_url=profile_data.get("avatar_url"),
            created_at=user_data["created_at"],
            updated_at=user_data["updated_at"],
            is_active=user_data["is_active"],
            preferences=profile_data.get("preferences", {}),
            coherence_summary=coherence_summary,
            statistics=statistics
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get profile error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve user profile"
        )


@router.put("/profile", response_model=UserProfile)
async def update_user_profile(
    profile_update: UserProfileUpdate,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Update user profile information"""
    
    try:
        # Get current profile data
        async with database.get_session() as session:
            user_query = "SELECT profile_data FROM users WHERE id = :user_id"
            user_result = await database.execute_query(user_query, {"user_id": current_user["id"]})
            
            if not user_result:
                raise HTTPException(status_code=404, detail="User not found")
            
            current_profile_data = user_result[0].get("profile_data", {})
        
        # Build update data
        update_data = {}
        profile_data_updates = {}
        
        # Update basic fields
        if profile_update.first_name is not None:
            update_data["first_name"] = profile_update.first_name
        if profile_update.last_name is not None:
            update_data["last_name"] = profile_update.last_name
        
        # Update profile data fields
        if profile_update.bio is not None:
            profile_data_updates["bio"] = profile_update.bio
        if profile_update.avatar_url is not None:
            profile_data_updates["avatar_url"] = profile_update.avatar_url
        if profile_update.preferences is not None:
            # Merge with existing preferences
            current_preferences = current_profile_data.get("preferences", {})
            current_preferences.update(profile_update.preferences)
            profile_data_updates["preferences"] = current_preferences
        
        # Update profile data
        if profile_data_updates:
            current_profile_data.update(profile_data_updates)
            update_data["profile_data"] = current_profile_data
        
        # Add updated timestamp
        update_data["updated_at"] = datetime.utcnow()
        
        # Build update query
        set_clauses = []
        params = {"user_id": current_user["id"]}
        
        for field, value in update_data.items():
            set_clauses.append(f"{field} = :{field}")
            params[field] = value
        
        if set_clauses:
            update_query = f"""
            UPDATE users
            SET {', '.join(set_clauses)}
            WHERE id = :user_id
            """
            
            await database.execute_query(update_query, params)
        
        # Return updated profile
        return await get_user_profile(current_user, include_coherence=False, include_stats=False)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Update profile error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update user profile"
        )


@router.get("/preferences")
async def get_user_preferences(
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Get user preferences"""
    
    try:
        async with database.get_session() as session:
            user_query = "SELECT profile_data FROM users WHERE id = :user_id"
            user_result = await database.execute_query(user_query, {"user_id": current_user["id"]})
            
            if not user_result:
                raise HTTPException(status_code=404, detail="User not found")
            
            profile_data = user_result[0].get("profile_data", {})
            preferences = profile_data.get("preferences", {})
            
            return {
                "preferences": preferences,
                "last_updated": profile_data.get("preferences_updated_at")
            }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get preferences error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve user preferences"
        )


@router.put("/preferences")
async def update_user_preferences(
    preferences_update: UserPreferencesUpdate,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Update user preferences"""
    
    try:
        # Get current profile data
        async with database.get_session() as session:
            user_query = "SELECT profile_data FROM users WHERE id = :user_id"
            user_result = await database.execute_query(user_query, {"user_id": current_user["id"]})
            
            if not user_result:
                raise HTTPException(status_code=404, detail="User not found")
            
            profile_data = user_result[0].get("profile_data", {})
            current_preferences = profile_data.get("preferences", {})
        
        # Update preferences
        update_dict = preferences_update.dict(exclude_unset=True)
        current_preferences.update(update_dict)
        
        # Update profile data
        profile_data["preferences"] = current_preferences
        profile_data["preferences_updated_at"] = datetime.utcnow().isoformat()
        
        # Update in database
        update_query = """
        UPDATE users
        SET profile_data = :profile_data, updated_at = :updated_at
        WHERE id = :user_id
        """
        
        await database.execute_query(update_query, {
            "profile_data": profile_data,
            "updated_at": datetime.utcnow(),
            "user_id": current_user["id"]
        })
        
        logger.info(f"Preferences updated for user: {current_user['id']}")
        
        return {
            "preferences": current_preferences,
            "last_updated": profile_data["preferences_updated_at"]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Update preferences error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update user preferences"
        )


@router.get("/activity", response_model=UserActivitySummary)
async def get_user_activity(
    current_user: Dict[str, Any] = Depends(get_current_user),
    days: int = Query(30, ge=1, le=365, description="Number of days to analyze")
):
    """Get user activity summary"""
    
    try:
        # Get activity data
        analytics_db = AnalyticsDB(database)
        activity_data = await analytics_db.get_user_analytics(current_user["id"], days)
        
        # Get coherence trajectory
        coherence_db = CoherenceProfileDB(database)
        coherence_trajectory = await coherence_db.get_trajectory(current_user["id"], limit=days)
        
        # Calculate coherence trend
        if len(coherence_trajectory) >= 2:
            recent_scores = [entry["coherence_score"] for entry in coherence_trajectory[:7]]  # Last week
            older_scores = [entry["coherence_score"] for entry in coherence_trajectory[7:14]]  # Previous week
            
            if older_scores:
                recent_avg = sum(recent_scores) / len(recent_scores)
                older_avg = sum(older_scores) / len(older_scores)
                
                if recent_avg > older_avg + 0.05:
                    trend = "improving"
                elif recent_avg < older_avg - 0.05:
                    trend = "declining"
                else:
                    trend = "stable"
            else:
                trend = "insufficient_data"
        else:
            trend = "insufficient_data"
        
        # Calculate average coherence score
        avg_coherence = 0.0
        if coherence_trajectory:
            avg_coherence = sum(entry["coherence_score"] for entry in coherence_trajectory) / len(coherence_trajectory)
        
        # Get last activity timestamp
        last_active = datetime.utcnow()
        if activity_data.get("daily_interactions"):
            last_active_date = max(activity_data["daily_interactions"].keys())
            last_active = datetime.fromisoformat(last_active_date)
        
        # Generate achievements (simple implementation)
        achievements = []
        if activity_data.get("total_interactions", 0) >= 100:
            achievements.append("Active User")
        if avg_coherence >= 0.8:
            achievements.append("High Coherence")
        if len(coherence_trajectory) >= 30:
            achievements.append("Consistent Tracker")
        if trend == "improving":
            achievements.append("Growing Coherence")
        
        return UserActivitySummary(
            user_id=current_user["id"],
            total_sessions=len(activity_data.get("daily_interactions", {})),
            total_interactions=activity_data.get("total_interactions", 0),
            avg_coherence_score=round(avg_coherence, 3),
            coherence_trend=trend,
            last_active=last_active,
            achievements=achievements
        )
        
    except Exception as e:
        logger.error(f"Get activity error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve user activity"
        )


@router.delete("/account", status_code=status.HTTP_204_NO_CONTENT)
async def delete_user_account(
    current_user: Dict[str, Any] = Depends(get_current_user),
    confirm: bool = Query(False, description="Confirm account deletion")
):
    """Delete user account and all associated data"""
    
    if not confirm:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Account deletion must be confirmed"
        )
    
    try:
        # Soft delete - mark as inactive instead of hard delete
        update_query = """
        UPDATE users
        SET is_active = FALSE, 
            email = CONCAT('deleted_', id, '@deleted.local'),
            updated_at = :updated_at,
            profile_data = :profile_data
        WHERE id = :user_id
        """
        
        await database.execute_query(update_query, {
            "updated_at": datetime.utcnow(),
            "profile_data": {"deleted_at": datetime.utcnow().isoformat()},
            "user_id": current_user["id"]
        })
        
        logger.info(f"Account deleted for user: {current_user['id']}")
        
    except Exception as e:
        logger.error(f"Delete account error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete account"
        )


@router.get("/export")
async def export_user_data(
    current_user: Dict[str, Any] = Depends(get_current_user),
    format: str = Query("json", regex="^(json|csv)$", description="Export format")
):
    """Export user data for GDPR compliance"""
    
    try:
        # Get user data
        async with database.get_session() as session:
            user_query = """
            SELECT id, email, first_name, last_name, created_at, updated_at, profile_data
            FROM users
            WHERE id = :user_id
            """
            user_result = await database.execute_query(user_query, {"user_id": current_user["id"]})
            
            if not user_result:
                raise HTTPException(status_code=404, detail="User not found")
        
        # Get coherence data
        coherence_db = CoherenceProfileDB(database)
        coherence_trajectory = await coherence_db.get_trajectory(current_user["id"], limit=1000)
        
        # Get analytics data
        analytics_db = AnalyticsDB(database)
        analytics_data = await analytics_db.get_user_analytics(current_user["id"], days=365)
        
        # Compile export data
        export_data = {
            "user_profile": user_result[0],
            "coherence_data": coherence_trajectory,
            "analytics_data": analytics_data,
            "export_date": datetime.utcnow().isoformat(),
            "data_retention_note": "This export contains all personal data stored in the IPAI system"
        }
        
        if format == "json":
            from fastapi.responses import JSONResponse
            return JSONResponse(
                content=export_data,
                headers={
                    "Content-Disposition": f"attachment; filename=ipai_export_{current_user['id']}.json"
                }
            )
        else:  # CSV format
            import csv
            import io
            from fastapi.responses import StreamingResponse
            
            output = io.StringIO()
            
            # Create CSV with flattened data (simplified)
            writer = csv.writer(output)
            writer.writerow(["Type", "Date", "Data"])
            
            # Add user data
            user_data = user_result[0]
            writer.writerow(["profile", user_data["created_at"], str(user_data)])
            
            # Add coherence data
            for entry in coherence_trajectory:
                writer.writerow(["coherence", entry["timestamp"], str(entry)])
            
            output.seek(0)
            
            return StreamingResponse(
                io.BytesIO(output.getvalue().encode()),
                media_type="text/csv",
                headers={
                    "Content-Disposition": f"attachment; filename=ipai_export_{current_user['id']}.csv"
                }
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Export data error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to export user data"
        )