SELECT
    c.name AS champion,
    COUNT(*) AS games,
    ROUND(AVG((s.kills + s.assists)::NUMERIC / NULLIF(s.deaths, 0)), 2) AS avg_kda
FROM participants p
JOIN stats1 s ON p.id = s.id
JOIN champs c ON p.championid = c.id
GROUP BY c.name
HAVING COUNT(*) > 100
ORDER BY avg_kda DESC
LIMIT 10;
