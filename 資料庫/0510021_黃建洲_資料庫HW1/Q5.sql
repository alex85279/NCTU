SELECT matchType, AVG(matchDuration) as avgDuration FROM `match` GROUP BY matchType ORDER BY avgDuration;
