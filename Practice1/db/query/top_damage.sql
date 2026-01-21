SELECT
    c.name,
    ROUND(AVG(s.totdmgtochamp), 0) AS avg_damage,
    COUNT(*) AS games
FROM participants p
JOIN stats1 s ON p.id = s.id
JOIN champs c ON p.championid = c.id
GROUP BY c.name
HAVING COUNT(*) > 100
ORDER BY avg_damage DESC
LIMIT 10;
