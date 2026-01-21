SELECT
    c.name AS champion,
    COUNT(*) AS games,
    ROUND(AVG(CASE WHEN s.win THEN 1 ELSE 0 END)::NUMERIC * 100, 2) AS winrate,
    ROUND(AVG(s.kills), 2) AS avg_kills,
    ROUND(AVG(s.deaths), 2) AS avg_deaths,
    ROUND(AVG(s.assists), 2) AS avg_assists,
    ROUND(AVG(s.goldearned), 0) AS avg_gold,
    ROUND(AVG(s.totdmgtochamp), 0) AS avg_damage
FROM champs c
JOIN participants p ON c.id = p.championid
JOIN stats1 s ON p.id = s.id
WHERE c.name = 'Jax'
GROUP BY c.name;
