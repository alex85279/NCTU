SELECT teamRoadKills, AVG(AVGwin) as avgWingPlacePerc 
FROM (SELECT SUM(roadKills) as teamRoadKills, AVG(winPlacePerc) as AVGwin, matchID FROM player_statistic GROUP BY groupId, matchId) as vehicle_focus 
JOIN(SELECT matchId FROM `match` WHERE matchType = 'squad' OR matchType = 'squad-fpp') as squad_game_focus 
USING(matchId) 
GROUP BY teamRoadKills 
ORDER BY teamRoadKills desc;
