-- IPAI SQL Optimization Examples
-- Optimized queries for analytics and data processing

-- =====================================================
-- 1. User Coherence Analytics Dashboard Query
-- =====================================================
-- Original: Multiple separate queries
-- Optimized: Single query with CTEs and window functions

WITH user_coherence_metrics AS (
    -- Get latest coherence scores and calculate trends
    SELECT 
        u.id AS user_id,
        u.email,
        u.created_at AS user_created_at,
        cp.coherence_score,
        cp.profile_data->>'psi' AS psi,
        cp.profile_data->>'rho' AS rho,
        cp.profile_data->>'q' AS q,
        cp.profile_data->>'f' AS f,
        cp.created_at AS profile_created_at,
        ROW_NUMBER() OVER (PARTITION BY u.id ORDER BY cp.created_at DESC) AS rn,
        LAG(cp.coherence_score) OVER (PARTITION BY u.id ORDER BY cp.created_at) AS prev_score,
        FIRST_VALUE(cp.coherence_score) OVER (
            PARTITION BY u.id 
            ORDER BY cp.created_at 
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) AS first_score
    FROM users u
    LEFT JOIN coherence_profiles cp ON u.id = cp.user_id
    WHERE cp.created_at >= NOW() - INTERVAL '30 days'
),
user_summary AS (
    -- Calculate per-user metrics
    SELECT 
        user_id,
        email,
        user_created_at,
        MAX(CASE WHEN rn = 1 THEN coherence_score END) AS current_score,
        MAX(CASE WHEN rn = 1 THEN psi::FLOAT END) AS current_psi,
        MAX(CASE WHEN rn = 1 THEN rho::FLOAT END) AS current_rho,
        MAX(CASE WHEN rn = 1 THEN q::FLOAT END) AS current_q,
        MAX(CASE WHEN rn = 1 THEN f::FLOAT END) AS current_f,
        AVG(coherence_score) AS avg_score_30d,
        STDDEV(coherence_score) AS score_volatility,
        COUNT(*) AS profile_count,
        MAX(CASE WHEN rn = 1 THEN coherence_score - prev_score END) AS recent_change,
        MAX(CASE WHEN rn = 1 THEN coherence_score - first_score END) AS total_improvement
    FROM user_coherence_metrics
    GROUP BY user_id, email, user_created_at
),
cohort_analysis AS (
    -- Group users into cohorts based on coherence levels
    SELECT 
        CASE 
            WHEN current_score >= 0.8 THEN 'High'
            WHEN current_score >= 0.6 THEN 'Medium-High'
            WHEN current_score >= 0.4 THEN 'Medium'
            WHEN current_score >= 0.2 THEN 'Low'
            ELSE 'Critical'
        END AS coherence_level,
        COUNT(*) AS user_count,
        AVG(current_score) AS avg_score,
        AVG(score_volatility) AS avg_volatility,
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY current_score) AS median_score
    FROM user_summary
    WHERE current_score IS NOT NULL
    GROUP BY coherence_level
)
-- Final dashboard query
SELECT 
    'summary' AS metric_type,
    json_build_object(
        'total_active_users', (SELECT COUNT(DISTINCT user_id) FROM user_summary WHERE current_score IS NOT NULL),
        'avg_coherence', (SELECT AVG(current_score) FROM user_summary WHERE current_score IS NOT NULL),
        'improving_users', (SELECT COUNT(*) FROM user_summary WHERE recent_change > 0.05),
        'at_risk_users', (SELECT COUNT(*) FROM user_summary WHERE current_score < 0.3),
        'cohort_distribution', (SELECT json_agg(row_to_json(cohort_analysis)) FROM cohort_analysis)
    ) AS metrics
UNION ALL
SELECT 
    'top_performers' AS metric_type,
    json_agg(
        json_build_object(
            'user_id', user_id,
            'email', email,
            'score', current_score,
            'improvement', total_improvement
        ) ORDER BY current_score DESC
    ) AS metrics
FROM user_summary
WHERE current_score >= 0.8
LIMIT 10;

-- =====================================================
-- 2. Time Series Analysis with Window Functions
-- =====================================================
-- Efficient trajectory analysis using window functions

WITH coherence_time_series AS (
    SELECT 
        user_id,
        DATE_TRUNC('day', created_at) AS day,
        AVG(coherence_score) AS daily_avg_score,
        COUNT(*) AS measurements,
        STDDEV(coherence_score) AS daily_volatility
    FROM coherence_profiles
    WHERE created_at >= NOW() - INTERVAL '90 days'
    GROUP BY user_id, DATE_TRUNC('day', created_at)
),
trajectory_analysis AS (
    SELECT 
        user_id,
        day,
        daily_avg_score,
        measurements,
        daily_volatility,
        -- 7-day moving average
        AVG(daily_avg_score) OVER (
            PARTITION BY user_id 
            ORDER BY day 
            ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
        ) AS ma7_score,
        -- 30-day moving average
        AVG(daily_avg_score) OVER (
            PARTITION BY user_id 
            ORDER BY day 
            ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
        ) AS ma30_score,
        -- Trend calculation
        REGR_SLOPE(daily_avg_score, EXTRACT(EPOCH FROM day)) OVER (
            PARTITION BY user_id 
            ORDER BY day 
            ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
        ) AS trend_slope,
        -- Detect anomalies (scores > 2 std dev from 30-day mean)
        CASE 
            WHEN ABS(daily_avg_score - AVG(daily_avg_score) OVER (
                PARTITION BY user_id 
                ORDER BY day 
                ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
            )) > 2 * STDDEV(daily_avg_score) OVER (
                PARTITION BY user_id 
                ORDER BY day 
                ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
            ) THEN 1
            ELSE 0
        END AS is_anomaly
    FROM coherence_time_series
)
SELECT 
    user_id,
    COUNT(DISTINCT day) AS active_days,
    AVG(daily_avg_score) AS avg_coherence,
    AVG(ma7_score) AS current_ma7,
    AVG(ma30_score) AS current_ma30,
    AVG(trend_slope) AS avg_trend,
    SUM(is_anomaly) AS anomaly_count,
    CASE 
        WHEN AVG(trend_slope) > 0.001 THEN 'Improving'
        WHEN AVG(trend_slope) < -0.001 THEN 'Declining'
        ELSE 'Stable'
    END AS trend_direction
FROM trajectory_analysis
WHERE day >= NOW() - INTERVAL '30 days'
GROUP BY user_id
HAVING COUNT(DISTINCT day) >= 7  -- Only users with sufficient data
ORDER BY avg_trend DESC;

-- =====================================================
-- 3. Component Correlation Analysis
-- =====================================================
-- Analyze relationships between GCT components

WITH component_data AS (
    SELECT 
        user_id,
        (profile_data->>'psi')::FLOAT AS psi,
        (profile_data->>'rho')::FLOAT AS rho,
        (profile_data->>'q')::FLOAT AS q,
        (profile_data->>'f')::FLOAT AS f,
        coherence_score
    FROM coherence_profiles
    WHERE created_at >= NOW() - INTERVAL '30 days'
        AND profile_data IS NOT NULL
)
SELECT 
    -- Correlation matrix
    CORR(psi, rho) AS psi_rho_correlation,
    CORR(psi, q) AS psi_q_correlation,
    CORR(psi, f) AS psi_f_correlation,
    CORR(rho, q) AS rho_q_correlation,
    CORR(rho, f) AS rho_f_correlation,
    CORR(q, f) AS q_f_correlation,
    -- Component impact on coherence
    CORR(psi, coherence_score) AS psi_coherence_impact,
    CORR(rho, coherence_score) AS rho_coherence_impact,
    CORR(q, coherence_score) AS q_coherence_impact,
    CORR(f, coherence_score) AS f_coherence_impact,
    -- Statistical summary
    COUNT(*) AS sample_size,
    AVG(coherence_score) AS avg_coherence,
    STDDEV(coherence_score) AS coherence_std_dev
FROM component_data;

-- =====================================================
-- 4. Intervention Effectiveness Analysis
-- =====================================================
-- Track effectiveness of interventions/recommendations

WITH intervention_windows AS (
    SELECT 
        cp1.user_id,
        cp1.created_at AS pre_intervention_time,
        cp1.coherence_score AS pre_score,
        cp2.created_at AS post_intervention_time,
        cp2.coherence_score AS post_score,
        cp2.coherence_score - cp1.coherence_score AS score_change,
        EXTRACT(DAYS FROM cp2.created_at - cp1.created_at) AS days_elapsed
    FROM coherence_profiles cp1
    JOIN coherence_profiles cp2 
        ON cp1.user_id = cp2.user_id 
        AND cp2.created_at > cp1.created_at
        AND cp2.created_at <= cp1.created_at + INTERVAL '14 days'
    WHERE EXISTS (
        -- Users who received interventions
        SELECT 1 FROM user_interactions ui
        WHERE ui.user_id = cp1.user_id
            AND ui.interaction_type = 'intervention_applied'
            AND ui.timestamp BETWEEN cp1.created_at AND cp2.created_at
    )
),
intervention_impact AS (
    SELECT 
        user_id,
        AVG(score_change) AS avg_improvement,
        MAX(score_change) AS max_improvement,
        MIN(score_change) AS min_improvement,
        COUNT(*) AS intervention_count,
        AVG(days_elapsed) AS avg_days_to_impact
    FROM intervention_windows
    GROUP BY user_id
)
SELECT 
    COUNT(DISTINCT user_id) AS users_with_interventions,
    AVG(avg_improvement) AS overall_avg_improvement,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY avg_improvement) AS median_improvement,
    SUM(CASE WHEN avg_improvement > 0 THEN 1 ELSE 0 END)::FLOAT / COUNT(*) AS success_rate,
    AVG(avg_days_to_impact) AS avg_response_time
FROM intervention_impact;

-- =====================================================
-- 5. Real-time Monitoring Query
-- =====================================================
-- For operational dashboards and alerting

WITH recent_activity AS (
    SELECT 
        DATE_TRUNC('hour', created_at) AS hour,
        COUNT(*) AS profile_count,
        AVG(coherence_score) AS avg_score,
        COUNT(DISTINCT user_id) AS unique_users,
        PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY coherence_score) AS p95_score,
        PERCENTILE_CONT(0.05) WITHIN GROUP (ORDER BY coherence_score) AS p05_score
    FROM coherence_profiles
    WHERE created_at >= NOW() - INTERVAL '24 hours'
    GROUP BY DATE_TRUNC('hour', created_at)
),
system_health AS (
    SELECT 
        ra.*,
        -- Calculate hourly rate vs. 7-day average
        ra.profile_count::FLOAT / NULLIF(
            (SELECT AVG(profile_count) 
             FROM (
                SELECT COUNT(*) AS profile_count
                FROM coherence_profiles
                WHERE created_at >= NOW() - INTERVAL '7 days'
                GROUP BY DATE_TRUNC('hour', created_at)
             ) AS weekly_avg
            ), 0
        ) AS activity_ratio,
        -- Detect anomalous coherence distributions
        CASE 
            WHEN ra.p95_score - ra.p05_score > 0.8 THEN 'High Variance Alert'
            WHEN ra.avg_score < 0.3 THEN 'Low Coherence Alert'
            WHEN ra.profile_count = 0 THEN 'No Activity Alert'
            ELSE 'Normal'
        END AS health_status
    FROM recent_activity ra
)
SELECT 
    hour,
    profile_count,
    unique_users,
    ROUND(avg_score::NUMERIC, 3) AS avg_coherence_score,
    ROUND(activity_ratio::NUMERIC, 2) AS activity_vs_baseline,
    health_status,
    json_build_object(
        'p95', ROUND(p95_score::NUMERIC, 3),
        'p05', ROUND(p05_score::NUMERIC, 3),
        'spread', ROUND((p95_score - p05_score)::NUMERIC, 3)
    ) AS distribution_metrics
FROM system_health
ORDER BY hour DESC;

-- =====================================================
-- 6. Materialized View for Performance
-- =====================================================
-- Create materialized views for expensive aggregations

DROP MATERIALIZED VIEW IF EXISTS mv_user_coherence_summary CASCADE;

CREATE MATERIALIZED VIEW mv_user_coherence_summary AS
WITH latest_profiles AS (
    SELECT DISTINCT ON (user_id) 
        user_id,
        coherence_score,
        profile_data,
        created_at
    FROM coherence_profiles
    ORDER BY user_id, created_at DESC
),
historical_stats AS (
    SELECT 
        user_id,
        COUNT(*) AS total_profiles,
        AVG(coherence_score) AS avg_coherence_all_time,
        STDDEV(coherence_score) AS coherence_volatility,
        MIN(created_at) AS first_profile_date,
        MAX(created_at) AS last_profile_date
    FROM coherence_profiles
    GROUP BY user_id
)
SELECT 
    u.id AS user_id,
    u.email,
    u.created_at AS user_created_at,
    lp.coherence_score AS current_coherence_score,
    (lp.profile_data->>'psi')::FLOAT AS current_psi,
    (lp.profile_data->>'rho')::FLOAT AS current_rho,
    (lp.profile_data->>'q')::FLOAT AS current_q,
    (lp.profile_data->>'f')::FLOAT AS current_f,
    hs.total_profiles,
    hs.avg_coherence_all_time,
    hs.coherence_volatility,
    hs.first_profile_date,
    hs.last_profile_date,
    EXTRACT(DAYS FROM NOW() - hs.last_profile_date) AS days_since_last_profile,
    CASE 
        WHEN lp.coherence_score >= 0.7 THEN 'High'
        WHEN lp.coherence_score >= 0.4 THEN 'Medium'
        WHEN lp.coherence_score >= 0.2 THEN 'Low'
        ELSE 'Critical'
    END AS coherence_level,
    NOW() AS refreshed_at
FROM users u
LEFT JOIN latest_profiles lp ON u.id = lp.user_id
LEFT JOIN historical_stats hs ON u.id = hs.user_id;

-- Create indexes on materialized view
CREATE INDEX idx_mv_user_coherence_user_id ON mv_user_coherence_summary(user_id);
CREATE INDEX idx_mv_user_coherence_level ON mv_user_coherence_summary(coherence_level);
CREATE INDEX idx_mv_user_coherence_score ON mv_user_coherence_summary(current_coherence_score);

-- Refresh function (call periodically)
CREATE OR REPLACE FUNCTION refresh_coherence_summary()
RETURNS void AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY mv_user_coherence_summary;
END;
$$ LANGUAGE plpgsql;

-- =====================================================
-- 7. Partitioned Table for Scale
-- =====================================================
-- Example of partitioning for better performance at scale

-- Create partitioned table
CREATE TABLE coherence_profiles_partitioned (
    LIKE coherence_profiles INCLUDING ALL
) PARTITION BY RANGE (created_at);

-- Create monthly partitions
CREATE TABLE coherence_profiles_2024_01 PARTITION OF coherence_profiles_partitioned
    FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');

CREATE TABLE coherence_profiles_2024_02 PARTITION OF coherence_profiles_partitioned
    FOR VALUES FROM ('2024-02-01') TO ('2024-03-01');

-- Add constraint to enable partition pruning
ALTER TABLE coherence_profiles_partitioned 
ADD CONSTRAINT coherence_score_check 
CHECK (coherence_score >= 0 AND coherence_score <= 1);

-- =====================================================
-- 8. Query Performance Analysis
-- =====================================================
-- Analyze query performance and identify slow queries

SELECT 
    query,
    calls,
    total_time,
    mean_time,
    stddev_time,
    min_time,
    max_time,
    rows
FROM pg_stat_statements
WHERE query LIKE '%coherence%'
    AND total_time > 1000  -- Queries taking more than 1 second total
ORDER BY mean_time DESC
LIMIT 20;