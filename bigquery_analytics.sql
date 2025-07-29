-- IPAI BigQuery Analytics Queries
-- Advanced analytics patterns optimized for BigQuery

-- =====================================================
-- 1. Cohort Analysis with ARRAY_AGG
-- =====================================================
-- Analyze user cohorts by signup date and coherence progression

WITH user_cohorts AS (
  SELECT 
    user_id,
    DATE_TRUNC(user_created_at, MONTH) AS cohort_month,
    DATE_DIFF(CURRENT_DATE(), DATE(user_created_at), DAY) AS days_since_signup
  FROM `project.dataset.users`
),
coherence_progression AS (
  SELECT 
    c.user_id,
    c.cohort_month,
    p.coherence_score,
    p.created_at,
    DATE_DIFF(DATE(p.created_at), DATE(c.cohort_month), DAY) AS days_from_signup,
    CASE 
      WHEN DATE_DIFF(DATE(p.created_at), DATE(c.cohort_month), DAY) <= 7 THEN 'Week 1'
      WHEN DATE_DIFF(DATE(p.created_at), DATE(c.cohort_month), DAY) <= 30 THEN 'Month 1'
      WHEN DATE_DIFF(DATE(p.created_at), DATE(c.cohort_month), DAY) <= 90 THEN 'Quarter 1'
      ELSE 'After Quarter 1'
    END AS retention_period
  FROM user_cohorts c
  JOIN `project.dataset.coherence_profiles` p ON c.user_id = p.user_id
),
cohort_metrics AS (
  SELECT 
    cohort_month,
    retention_period,
    COUNT(DISTINCT user_id) AS active_users,
    AVG(coherence_score) AS avg_coherence,
    APPROX_QUANTILES(coherence_score, 100)[OFFSET(50)] AS median_coherence,
    STDDEV(coherence_score) AS coherence_stddev,
    -- Create array of all scores for further analysis
    ARRAY_AGG(coherence_score IGNORE NULLS) AS score_distribution
  FROM coherence_progression
  GROUP BY cohort_month, retention_period
)
SELECT 
  cohort_month,
  retention_period,
  active_users,
  ROUND(avg_coherence, 3) AS avg_coherence,
  ROUND(median_coherence, 3) AS median_coherence,
  ROUND(coherence_stddev, 3) AS coherence_stddev,
  -- Calculate percentiles from the distribution
  ROUND(APPROX_QUANTILES(ARRAY_CONCAT_AGG(score_distribution), 100)[OFFSET(25)], 3) AS p25,
  ROUND(APPROX_QUANTILES(ARRAY_CONCAT_AGG(score_distribution), 100)[OFFSET(75)], 3) AS p75
FROM cohort_metrics
GROUP BY cohort_month, retention_period, active_users, avg_coherence, median_coherence, coherence_stddev
ORDER BY cohort_month DESC, retention_period;

-- =====================================================
-- 2. User Journey Analysis with WINDOW Functions
-- =====================================================
-- Track individual user journeys and identify patterns

WITH user_events AS (
  -- Combine different event types into unified timeline
  SELECT 
    user_id,
    'profile_created' AS event_type,
    created_at AS event_time,
    coherence_score AS event_value,
    STRUCT(
      JSON_EXTRACT_SCALAR(profile_data, '$.psi') AS psi,
      JSON_EXTRACT_SCALAR(profile_data, '$.rho') AS rho,
      JSON_EXTRACT_SCALAR(profile_data, '$.q') AS q,
      JSON_EXTRACT_SCALAR(profile_data, '$.f') AS f
    ) AS components
  FROM `project.dataset.coherence_profiles`
  
  UNION ALL
  
  SELECT 
    user_id,
    interaction_type AS event_type,
    timestamp AS event_time,
    NULL AS event_value,
    NULL AS components
  FROM `project.dataset.user_interactions`
),
user_journey_enriched AS (
  SELECT 
    user_id,
    event_type,
    event_time,
    event_value,
    components,
    -- Calculate time since last event
    TIMESTAMP_DIFF(event_time, 
      LAG(event_time) OVER (PARTITION BY user_id ORDER BY event_time), 
      HOUR
    ) AS hours_since_last_event,
    -- Running coherence average
    AVG(event_value) OVER (
      PARTITION BY user_id 
      ORDER BY event_time 
      ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS running_avg_coherence,
    -- Event sequence number
    ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY event_time) AS event_sequence,
    -- Days since first event
    DATE_DIFF(
      DATE(event_time), 
      DATE(FIRST_VALUE(event_time) OVER (PARTITION BY user_id ORDER BY event_time)),
      DAY
    ) AS days_since_start
  FROM user_events
),
journey_patterns AS (
  SELECT 
    user_id,
    COUNT(*) AS total_events,
    COUNT(DISTINCT DATE(event_time)) AS active_days,
    MAX(days_since_start) AS journey_length_days,
    AVG(hours_since_last_event) AS avg_hours_between_events,
    -- Create journey summary
    STRING_AGG(
      event_type, 
      ' -> ' 
      ORDER BY event_time
      LIMIT 10
    ) AS journey_path,
    -- Final vs initial coherence
    MAX(CASE WHEN event_value IS NOT NULL THEN event_value END) AS latest_coherence,
    MIN(CASE WHEN event_value IS NOT NULL THEN event_value END) AS initial_coherence,
    -- Engagement pattern
    CASE 
      WHEN AVG(hours_since_last_event) < 24 THEN 'Daily Active'
      WHEN AVG(hours_since_last_event) < 168 THEN 'Weekly Active'
      WHEN AVG(hours_since_last_event) < 720 THEN 'Monthly Active'
      ELSE 'Sporadic'
    END AS engagement_pattern
  FROM user_journey_enriched
  GROUP BY user_id
)
SELECT 
  engagement_pattern,
  COUNT(*) AS user_count,
  AVG(journey_length_days) AS avg_journey_days,
  AVG(total_events) AS avg_events_per_user,
  AVG(latest_coherence - initial_coherence) AS avg_coherence_improvement,
  -- Sample journey paths
  ARRAY_AGG(
    STRUCT(user_id, journey_path) 
    ORDER BY latest_coherence - initial_coherence DESC 
    LIMIT 5
  ) AS top_improving_journeys
FROM journey_patterns
GROUP BY engagement_pattern;

-- =====================================================
-- 3. ML-Ready Feature Engineering
-- =====================================================
-- Prepare features for machine learning models

WITH user_features AS (
  SELECT 
    u.user_id,
    -- User demographics
    DATE_DIFF(CURRENT_DATE(), DATE(u.created_at), DAY) AS account_age_days,
    u.profile_data.age AS user_age,
    u.profile_data.gender AS user_gender,
    u.profile_data.timezone AS user_timezone,
    
    -- Coherence statistics
    AVG(cp.coherence_score) AS avg_coherence_30d,
    STDDEV(cp.coherence_score) AS coherence_volatility_30d,
    MAX(cp.coherence_score) AS max_coherence_30d,
    MIN(cp.coherence_score) AS min_coherence_30d,
    COUNT(cp.id) AS profile_count_30d,
    
    -- Component averages
    AVG(CAST(JSON_EXTRACT_SCALAR(cp.profile_data, '$.psi') AS FLOAT64)) AS avg_psi_30d,
    AVG(CAST(JSON_EXTRACT_SCALAR(cp.profile_data, '$.rho') AS FLOAT64)) AS avg_rho_30d,
    AVG(CAST(JSON_EXTRACT_SCALAR(cp.profile_data, '$.q') AS FLOAT64)) AS avg_q_30d,
    AVG(CAST(JSON_EXTRACT_SCALAR(cp.profile_data, '$.f') AS FLOAT64)) AS avg_f_30d,
    
    -- Trend features
    ARRAY_AGG(
      STRUCT(
        cp.created_at,
        cp.coherence_score
      )
      ORDER BY cp.created_at
    ) AS coherence_history
    
  FROM `project.dataset.users` u
  LEFT JOIN `project.dataset.coherence_profiles` cp 
    ON u.user_id = cp.user_id
    AND cp.created_at >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 30 DAY)
  GROUP BY u.user_id, u.created_at, u.profile_data.age, u.profile_data.gender, u.profile_data.timezone
),
engineered_features AS (
  SELECT 
    *,
    -- Calculate trend using linear regression approximation
    (SELECT 
      CORR(
        UNIX_SECONDS(h.created_at), 
        h.coherence_score
      ) * STDDEV(h.coherence_score) / STDDEV(UNIX_SECONDS(h.created_at))
    FROM UNNEST(coherence_history) h
    ) AS coherence_trend_slope,
    
    -- Time-based features
    EXTRACT(DAYOFWEEK FROM CURRENT_TIMESTAMP()) AS current_day_of_week,
    EXTRACT(HOUR FROM CURRENT_TIMESTAMP()) AS current_hour,
    
    -- Engagement features
    profile_count_30d / 30.0 AS daily_profile_rate,
    
    -- Imbalance features
    GREATEST(
      ABS(avg_psi_30d - avg_rho_30d),
      ABS(avg_psi_30d - avg_q_30d),
      ABS(avg_psi_30d - avg_f_30d),
      ABS(avg_rho_30d - avg_q_30d),
      ABS(avg_rho_30d - avg_f_30d),
      ABS(avg_q_30d - avg_f_30d)
    ) AS max_component_imbalance,
    
    -- Risk indicators
    CASE 
      WHEN avg_coherence_30d < 0.3 THEN 1 
      ELSE 0 
    END AS is_at_risk,
    
    CASE 
      WHEN coherence_volatility_30d > 0.2 THEN 1 
      ELSE 0 
    END AS is_volatile
    
  FROM user_features
)
-- Export features for ML training
SELECT 
  * EXCEPT(coherence_history)
FROM engineered_features
WHERE profile_count_30d >= 5;  -- Only users with sufficient data

-- =====================================================
-- 4. Network Effects Analysis
-- =====================================================
-- Analyze how users influence each other

WITH user_interactions_network AS (
  -- Extract user-to-user interactions
  SELECT 
    ui.user_id AS source_user,
    JSON_EXTRACT_SCALAR(ui.interaction_data, '$.target_user_id') AS target_user,
    ui.interaction_type,
    ui.timestamp
  FROM `project.dataset.user_interactions` ui
  WHERE JSON_EXTRACT_SCALAR(ui.interaction_data, '$.target_user_id') IS NOT NULL
),
user_coherence_snapshot AS (
  -- Get latest coherence for each user
  SELECT 
    user_id,
    coherence_score,
    created_at
  FROM (
    SELECT 
      user_id,
      coherence_score,
      created_at,
      ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY created_at DESC) AS rn
    FROM `project.dataset.coherence_profiles`
  )
  WHERE rn = 1
),
network_metrics AS (
  SELECT 
    n.source_user,
    COUNT(DISTINCT n.target_user) AS connections_count,
    AVG(t.coherence_score) AS avg_connection_coherence,
    s.coherence_score AS user_coherence,
    -- Calculate influence score
    CORR(s.coherence_score, t.coherence_score) AS influence_correlation,
    -- Network position
    COUNT(DISTINCT n2.source_user) AS inbound_connections
  FROM user_interactions_network n
  JOIN user_coherence_snapshot s ON n.source_user = s.user_id
  JOIN user_coherence_snapshot t ON n.target_user = t.user_id
  LEFT JOIN user_interactions_network n2 ON n.source_user = n2.target_user
  GROUP BY n.source_user, s.coherence_score
)
SELECT 
  CASE 
    WHEN user_coherence >= 0.7 AND avg_connection_coherence >= 0.6 THEN 'High Coherence Cluster'
    WHEN user_coherence >= 0.7 AND avg_connection_coherence < 0.6 THEN 'Positive Influencer'
    WHEN user_coherence < 0.4 AND avg_connection_coherence >= 0.6 THEN 'Needs Support'
    ELSE 'Mixed Network'
  END AS network_category,
  COUNT(*) AS user_count,
  AVG(connections_count) AS avg_connections,
  AVG(user_coherence) AS avg_user_coherence,
  AVG(avg_connection_coherence) AS avg_network_coherence,
  AVG(influence_correlation) AS avg_influence_score
FROM network_metrics
GROUP BY network_category;

-- =====================================================
-- 5. Intervention Recommendation Engine
-- =====================================================
-- Generate personalized intervention recommendations

WITH user_risk_profiles AS (
  SELECT 
    user_id,
    -- Current state
    current_coherence,
    current_psi,
    current_rho,
    current_q,
    current_f,
    -- Historical patterns
    coherence_trend,
    volatility,
    days_since_last_profile,
    -- Risk scores
    CASE 
      WHEN current_coherence < 0.2 THEN 'critical'
      WHEN current_coherence < 0.4 THEN 'high'
      WHEN coherence_trend < -0.1 THEN 'moderate'
      ELSE 'low'
    END AS risk_level,
    -- Component analysis
    LEAST(current_psi, current_rho, current_q, current_f) AS weakest_component_value,
    CASE 
      WHEN current_psi = LEAST(current_psi, current_rho, current_q, current_f) THEN 'psi'
      WHEN current_rho = LEAST(current_psi, current_rho, current_q, current_f) THEN 'rho'
      WHEN current_q = LEAST(current_psi, current_rho, current_q, current_f) THEN 'q'
      ELSE 'f'
    END AS weakest_component
  FROM `project.dataset.mv_user_coherence_summary`
  WHERE days_since_last_profile <= 7  -- Active users only
),
intervention_recommendations AS (
  SELECT 
    user_id,
    risk_level,
    current_coherence,
    weakest_component,
    -- Generate recommendations array
    ARRAY_CONCAT(
      -- Risk-based interventions
      CASE risk_level
        WHEN 'critical' THEN ['immediate_support', 'crisis_resources', 'daily_check_in']
        WHEN 'high' THEN ['intensive_coaching', 'peer_support_group']
        WHEN 'moderate' THEN ['weekly_coaching', 'self_guided_exercises']
        ELSE ['monthly_check_in']
      END,
      -- Component-specific interventions
      CASE weakest_component
        WHEN 'psi' THEN ['consistency_exercises', 'value_alignment_workshop']
        WHEN 'rho' THEN ['reflection_practices', 'wisdom_building_content']
        WHEN 'q' THEN ['moral_activation_challenges', 'purpose_finding_session']
        WHEN 'f' THEN ['community_engagement', 'social_connection_activities']
      END,
      -- Trend-based interventions
      IF(coherence_trend < -0.05, ['trend_reversal_program'], [])
    ) AS recommended_interventions,
    -- Priority score
    CASE risk_level
      WHEN 'critical' THEN 1
      WHEN 'high' THEN 2
      WHEN 'moderate' THEN 3
      ELSE 4
    END AS priority_score
  FROM user_risk_profiles
)
-- Output recommendations with estimated impact
SELECT 
  user_id,
  risk_level,
  ROUND(current_coherence, 3) AS current_coherence,
  weakest_component,
  recommended_interventions,
  priority_score,
  -- Estimate potential improvement
  CASE 
    WHEN risk_level = 'critical' THEN 0.15
    WHEN risk_level = 'high' THEN 0.10
    WHEN risk_level = 'moderate' THEN 0.05
    ELSE 0.02
  END AS estimated_coherence_improvement
FROM intervention_recommendations
ORDER BY priority_score, current_coherence;

-- =====================================================
-- 6. A/B Test Analysis Framework
-- =====================================================
-- Analyze effectiveness of different interventions

WITH experiment_groups AS (
  SELECT 
    user_id,
    experiment_id,
    variant,
    DATE(enrollment_time) AS enrollment_date
  FROM `project.dataset.experiments`
  WHERE experiment_id = 'coherence_intervention_v2'
),
pre_post_metrics AS (
  SELECT 
    e.user_id,
    e.variant,
    e.enrollment_date,
    -- Pre-experiment metrics (30 days before)
    AVG(CASE 
      WHEN cp.created_at < e.enrollment_time 
      AND cp.created_at >= TIMESTAMP_SUB(e.enrollment_time, INTERVAL 30 DAY)
      THEN cp.coherence_score 
    END) AS pre_coherence,
    -- Post-experiment metrics (30 days after)
    AVG(CASE 
      WHEN cp.created_at >= e.enrollment_time 
      AND cp.created_at < TIMESTAMP_ADD(e.enrollment_time, INTERVAL 30 DAY)
      THEN cp.coherence_score 
    END) AS post_coherence,
    -- Engagement metrics
    COUNT(CASE 
      WHEN cp.created_at >= e.enrollment_time 
      AND cp.created_at < TIMESTAMP_ADD(e.enrollment_time, INTERVAL 30 DAY)
      THEN 1 
    END) AS post_profile_count
  FROM experiment_groups e
  LEFT JOIN `project.dataset.coherence_profiles` cp ON e.user_id = cp.user_id
  GROUP BY e.user_id, e.variant, e.enrollment_date
),
experiment_results AS (
  SELECT 
    variant,
    COUNT(*) AS sample_size,
    AVG(post_coherence - pre_coherence) AS avg_improvement,
    STDDEV(post_coherence - pre_coherence) AS improvement_stddev,
    -- Statistical significance using t-test approximation
    AVG(post_coherence - pre_coherence) / 
      (STDDEV(post_coherence - pre_coherence) / SQRT(COUNT(*))) AS t_statistic,
    -- Effect size (Cohen's d)
    AVG(post_coherence - pre_coherence) / 
      STDDEV(pre_coherence) AS effect_size,
    -- Engagement metrics
    AVG(post_profile_count) AS avg_post_profiles,
    SUM(CASE WHEN post_profile_count = 0 THEN 1 ELSE 0 END) / COUNT(*) AS dropout_rate
  FROM pre_post_metrics
  WHERE pre_coherence IS NOT NULL AND post_coherence IS NOT NULL
  GROUP BY variant
)
SELECT 
  variant,
  sample_size,
  ROUND(avg_improvement, 4) AS avg_improvement,
  ROUND(improvement_stddev, 4) AS improvement_stddev,
  ROUND(t_statistic, 2) AS t_statistic,
  CASE 
    WHEN ABS(t_statistic) > 2.576 THEN 'p < 0.01'
    WHEN ABS(t_statistic) > 1.96 THEN 'p < 0.05'
    WHEN ABS(t_statistic) > 1.645 THEN 'p < 0.10'
    ELSE 'not significant'
  END AS significance,
  ROUND(effect_size, 3) AS cohens_d,
  ROUND(avg_post_profiles, 1) AS avg_engagement,
  ROUND(dropout_rate, 3) AS dropout_rate
FROM experiment_results
ORDER BY avg_improvement DESC;