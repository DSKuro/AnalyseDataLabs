\copy champs FROM 'D:/Repos/PyCharm/AnalyseDataLabs/data/champs.csv' DELIMITER ',' CSV HEADER QUOTE '"';
\copy matches FROM 'D:/Repos/PyCharm/AnalyseDataLabs/data/matches.csv' DELIMITER ',' CSV HEADER QUOTE '"';
\copy participants FROM 'D:/Repos/PyCharm/AnalyseDataLabs/data/participants.csv' DELIMITER ',' CSV HEADER QUOTE '"';
\copy stats1 FROM 'D:/Repos/PyCharm/AnalyseDataLabs/data/stats1.csv' DELIMITER ',' CSV HEADER QUOTE '"' NULL '\N';
\copy stats2 FROM 'D:/Repos/PyCharm/AnalyseDataLabs/data/stats2.csv' DELIMITER ',' CSV HEADER QUOTE '"' NULL '\N';
\copy teamstats FROM 'D:/Repos/PyCharm/AnalyseDataLabs/data/teamstats.csv' DELIMITER ',' CSV HEADER QUOTE '"' NULL '\N';
\copy teambans FROM 'D:/Repos/PyCharm/AnalyseDataLabs/data/teambans.csv' DELIMITER ',' CSV HEADER QUOTE '"';
