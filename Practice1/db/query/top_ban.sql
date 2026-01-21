SELECT
    c.name AS champion,
    COUNT(*) AS bans
FROM teambans tb
JOIN champs c ON tb.championid = c.id
GROUP BY c.name
ORDER BY bans DESC
LIMIT 10;
