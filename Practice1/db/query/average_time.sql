SELECT
    m.version,
    ROUND(AVG(m.duration) / 60, 2) AS avg_duration_minutes,
    COUNT(*) AS matches
FROM matches m
GROUP BY m.version
ORDER BY avg_duration_minutes DESC;
