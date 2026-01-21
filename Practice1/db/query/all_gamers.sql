SELECT COUNT(*) AS total_participants
FROM participants p
JOIN matches m ON p.matchid = m.id;
