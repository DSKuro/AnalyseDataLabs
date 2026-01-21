SELECT
    c.name AS champion,
    p.position,
    COUNT(*) AS games,
    ROUND(AVG(CASE WHEN s.win THEN 1 ELSE 0 END)::NUMERIC * 100, 2) AS winrate
FROM participants p
JOIN stats1 s ON p.id = s.id
JOIN champs c ON p.championid = c.id
GROUP BY c.name, p.position
HAVING COUNT(*) > 50
ORDER BY winrate DESC;
