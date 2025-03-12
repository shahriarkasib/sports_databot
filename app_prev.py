from flask import Flask, request, jsonify, make_response, redirect
from flask_cors import CORS
from openai import OpenAI
import openai
import os
import psycopg2
from urllib.parse import urlparse
import re
import pandas as pd
from flask_talisman import Talisman



app = Flask(__name__)
CORS(app)
Talisman(app)

mlb_pitching_dataframe = pd.read_csv('mlb_pitching_gpt.csv')
mlb_batting_dataframe = pd.read_csv('mlb_batting_gpt.csv')
paDF = pd.read_csv('paDF_gpt.csv')
pitchDF = pd.read_csv('pitchDF_gpt.csv')
nfl_main_df = pd.read_csv('nfl_main_gpt.csv')
play_by_play_nfl = pd.read_csv('play_by_play_gpt.csv')
nba_main_dataframe = pd.read_csv('nba_main_dataframe_gpt.csv')
play_by_play_nba = pd.read_csv('nba_play_by_play_gpt.csv')

table = mlb_batting_dataframe.to_csv()
table2 = mlb_pitching_dataframe.to_csv()
table3 = paDF.to_csv()
table4 = pitchDF.to_csv()
table5 = nfl_main_df.to_csv()
table6 = play_by_play_nfl.to_csv()
table7 = nba_main_dataframe.to_csv()
table8 = play_by_play_nba.to_csv()

baseball_stats = {
    "Hits (H)": "A hit is a single, double, triple or home_run. Use column h  or h_allowed (only in mlb_pitching_dataframe)",
    "Errors (E)": "An error is charged to a fielder when they mishandle a ball in a way that allows a batter or baserunner to advance one or more bases or reach base when they otherwise would have been out.",
    "At Bats (AB)": "At bat is every time the ball is hit_into_play or when a strikeouts happens",
    "Total Bases (TB)": "singles*1 + doubles*2 + triples*3 + hr*4",
    "Runs (R)": "A run is scored when a player successfully advances around the bases and reaches home plate.",
    "Runs Batted In (RBI)": "Runs batted in are credited to a batter when the result of their at bat results in a run being scored, except in the case of an error or a ground into double play.",
    "Batting Average (AVG)": "Batting average is calculated by dividing a player's hits (h) by their at bats (AVG = h/ab).",
    "On-Base Percentage (OBP)": "On-base percentage is a measure of how often a batter reaches base. It is calculated by dividing the sum of hits, walks, and hit-by-pitches by the sum of at bats, walks, hit-by-pitches, and sacrifice flies (OBP = (H + BB + HBP) / (AB + BB + HBP)).",
    "Slugging Percentage (SLG)": "Slugging percentage measures the power of a hitter. It is calculated by dividing total bases by at bats (SLG = TB/AB).",
    "Earned Run Average (ERA)": "ERA is a statistic used to evaluate pitchers. It is the average number of earned runs a pitcher allows per nine innings pitched (ERA = (ER/IP) * 9).",
    "Strikeouts (K)": "A strikeout occurs when a batter accumulates three strikes during their at bat.",
    "Walks (BB)": "A walk occurs when a batter receives four balls during their at bat and is awarded first base.",
    "Plate Appearances (PA)": "Plate appearances is the number of times a player completes a turn batting, including all at bats plus walks, hit by pitches, sacrifices, and times reaching on defensive interference or obstruction.",
    "WHIP (Walks and Hits per Innings Pitched)": "WHIP is a measure of the number of baserunners a pitcher has allowed per inning pitched, calculated as (bb + h) / ip.",
    "K/9 (Strikeout rate)": "K/9 is the number of strikeouts a pitcher averages per nine innings pitched, calculated as (k * 9) / ip.",
    "BB/9 (Walk rate)": "BB/9 is the number of walks a pitcher averages per nine innings pitched, calculated as (bb * 9) / ip.",
    "K% (Strikeout percentage)": "K% is the percentage of plate appearances that result in a strikeout for a pitcher, calculated as (k / pa) * 100.",
    "BB% (Walk percentage)": "BB% is the percentage of plate appearances that result in a walk for a pitcher, calculated as (bb / pa) * 100.",
    "K/BB (Strikeout-to-Walk ratio)": "K/BB is the ratio of strikeouts to walks for a pitcher, calculated as k / bb.",
    "Line Drive Percentage (LD%)": "LD% is the percentage of batted balls that are line drives, calculated as (Line Drives / Balls in Play * 100.",
    "Groundball Percentage (GB%)": "GB% is the percentage of batted balls that are groundballs, calculated as (Groundballs / Balls in Play * 100.",
    "Popup Percentage (IFFB%)": "IFFB% is the percentage of fly balls that are infield pop-ups, calculated as (Infield Fly Balls / Fly Balls) * 100.",
    "Flyball Percentage (FB%)": "FB% is the percentage of batted balls that are flyballs, calculated as (Flyballs / Balls in Play) * 100.",
    "No Hitter": "A No Hitter is when a pitcher pitches 9 innings and gives up 0 hits",
     "Whiff Rate": "Whiff Rate is the percentage of swings and misses a batter generates on the swings they take, calculated as (swinging_strike / (swings)) * 100.",
    "Strike Ratio":"strike/strike+ball"

}


batting_columns = list(mlb_batting_dataframe.columns)
pitching_columns = list(mlb_pitching_dataframe.columns)

pa_columns = list(paDF.columns)

pitch_df_columns = list(pitchDF.columns)
nfl_cols = list(nfl_main_df.columns)
nfl_pbp_cols = list(play_by_play_nfl.columns)
nba_main_cols = list(nba_main_dataframe.columns)
nba_pbp_cols = list(play_by_play_nba.columns)


content = '''
You are an expert on sports data.

Based on the question you have to decide which table to use. There are 8 options:

The first 4 tables are to get baseball data
The first table is called mlb_batting_dataframe and it is here {}.
mlb_batting_dataframe columns are {}
The second table is called mlb_pitching_dataframe and it is here {}.
mlb_pitching_dataframe columns are {}
mlb_pitching_dataframe has per game information for pitchers (each row represents one game pitched by a pitcher). Use this table for pitcher related questions. 
mlb_batting_dataframe has per game information for batters (each row represents one game of stats for a single batter). Use this table for batter related questions. 
The third table is called plate_appearance_dataframe and it is here {}.
plate_appearance_dataframe columns are {}
The fourth table is called pitch_dataframe and it is here {}
pitch_dataframe columns are {}
plate_appearance_dataframe has per plate appearance information for both batter and pitchers. Each row represents a single plate appearance between a batter and a pitcher.
pitch_dataframe has per pitch information for both batter and pitchers. Each row represents a single pitch thrown by a pitcher against a certain batter.
BE CAREFUL TO USE MLB_BATTING_DATAFRAME FOR HITTERS AND MLB_PITCHING_DATAFRAME FOR PITCHERS
note that plate_appearance_dataframe doesn't have a player_name column, but rather has a pitcher_name AND a batter_name column since for each at bat there is a batter and a pitcher.
Try to see if it's possible to answer the question asked first by using mlb_batting_dataframe or mlb_pitching_dataframe. These contain game level stats.
If the question asked requires information that can only be extracted from a per at bat table , then always use plate_appearance_dataframe.
Always use plate_appearance_dataframe when the question is related to count
Always use plate_appearance_dataframe when the question is related to runners on base
Always use plate_appearance_dataframe when the question is related to number of outs
Always use plate_appearance_dataframe when the question is related to handness of batter/pitcher 
Or if the question asks about a matchup between 2 players, use plate_appearance_dataframe.
Use batter_team_name if the question asks about "teams against ... pitching/pitchers" 
Use pitcher_team_name if the question asks about "teams against ... batting/batters" 
Use batter_name if the question asks about "players/batters against ... pitching/pitchers" 
Use pitcher_name if the question asks about "playes/pitchers against ... batting/batters" 
to filter batters handness always use plate_appearance_dataframe 
to filter pitchers handness always use plate_appearance_dataframe 
Never use mlb_pitching_dataframe if the question asks about a specific inning (1st, 2nd, 3rd inning etc...)
For whiff rate, contact rate use pitch_dataframe

There are 2 table to get data regarding NFL players

The first table is called nfl_main_df and it is here {}.
nfl_main_df is on a per game level. Each row ahve stats of a player on a certain game
Always use opponent_team_name in nfl_main_df when you need to get any allowed stat (or when the question asks about defense stats)
nfl_main_df columns are {}

The second table is called nfl_play_by_play and it is here {}
nfl_play_by_play is on a per play level. It has down-by-down play data. Each row have stats of a certain play
nfl_play_by_play columns are {}. 
Always use nfl_play_by_play if the question is related to time remaining, yards to endzone, yards to 1st down etc
Always use nfl_play_by_play if the question is related quarter (even first/1st quarter)
IF the question contains "down" or "downs" (in any context), you MUST use the nfl_play_by_play table. Even if the question is about 1st down, 2nd down, first down, second down, third down, 3rd down or any down-related context, select nfl_play_by_play. Ignore other tables if this keyword is detected.
Return nfl_play_by_play if "down" or "downs" is in the question

There are 2 tables to get data regarding NBA players
The first table is called nba_main_dataframe and it is here {}.
nba_main_dataframe is on a per game level. Each row have stats of a player on a certain game
nba_main_dataframe columns are {}. 
The second table is called nba_play_by_play and it is here {}
nba_play_by_play columns are {}. 
nba_play_by_play is on a per play level. Each row have stats of a player on an individual play
Use nba_play_by_play if the question is regarding interactions between players (ex: which player assists the most to each other, which player blocked the most certain other player)
Use nba_play_by_play if the question is regarding a player stat in a certain quarter
Always use one of the NBA tables if the question asks about assists

Return ONLY the name of the table you would use. And choose only 1

'''.format(table,batting_columns,table2,pitching_columns,
           table3,pa_columns,table4,pitch_df_columns,
          table5,nfl_cols,table6,nfl_pbp_cols,table7,nba_main_cols,
		table8,nba_pbp_cols)

content1 = """
    
You are an expert on baseball data.
You have to use mlb_batting_dataframe to answer a question
mlb_batting_dataframe is here {}  
mlb_batting_dataframe has per game information for batters (each row represents one game of stats for a single batter). Use this table for batter related questions. 

Also, use common sense when someone asks subjective questions like "best", "worst","highest","lowest". There's a minimum threshold of ab or pa that should be considered before annointing someone as the "best" or "worst" or "highest" or "lowest".
Any time you're answering a question that asks the "best","worst","highest" or "lowest", YOU MUST ALWAYS use an appropriate minimum threshold of ab or pa for both pitchers or batters (both allowed or hitted stats)
As long as the word "career" isn't used in the question, include a filter for data only in 2024.
For certain stats like ERA, OPS, Batting Average, and others, you cannot just average the rows. You have to sum the cumulative components of the stats first.
There is a DIFFERENCE between called strikes and called strikeouts. Similarly there is a difference between swinging strikes and swinging strikeouts.
For called strikes and swinging strikes, use mlb_pitching_dataframe. For called strikeouts and swinging strikeouts, you have to use plate_appearance_dataframe.
Every time it asks for some average (like batting average, earned runs average...), percentage (like on base percentage...), or rate (like flyball rate, strikeout rate...), you have to look for the formula and do the division
When asked regarding pitchers (if the question has "pitcher" or "allowed" in it, you should consider only starting pitchers (starter=1). Unless the question specifically asks consider relievers

ON EVERY QUERY, MAKE SURE TO INCLUDES CHECKS TO NOT DIVIDE BY ZERO. The numerator can be zero at times but if the denominator is zero that's a problem. 
Ensure all non-aggregated columns in the SELECT statement are included in the GROUP BY clause. In nested queries, match the GROUP BY in the outer query with any non-aggregated columns from the subquery to avoid GroupingError.
There are no NaNs in the tables, so never use "NOT NULL" in any queries
When grouping by to calculate a percentage or a rate, always use a minimum threshold
When generating SQL queries involving mathematical operations, especially division, always ensure that all operands are explicitly cast to the same type (e.g., FLOAT) to avoid type mismatch errors. Use CAST as FLOAT on both
Any time the question is asking for "best," "worst," "highest," or "lowest,", return the top 10 results, not just 1.

Use this glossary for baseball formulas: {}
mlb_batting_dataframe columns are {}

Using these tables, return a SQL query that will answer the question.

RETURN ONLY the SQL query, NOTHING else.


""".format(table,baseball_stats,batting_columns,
          )

content2 = """
    
You are an expert on baseball data.
You have to use mlb_pitching_dataframe to answer a question
mlb_pitching_dataframe is here {}  
mlb_pitching_dataframe has per game information for pitchers (each row represents one game pitched by a pitcher). Use this table for pitcher related questions. 

Also, use common sense when someone asks subjective questions like "best", "worst","highest","lowest". There's a minimum threshold of ab or pa that should be considered before annointing someone as the "best" or "worst" or "highest" or "lowest".
Any time you're answering a question that asks the "best","worst","highest" or "lowest", YOU MUST ALWAYS use an appropriate minimum threshold of ab or pa for both pitchers or batters (both allowed or hitted stats)
You should also ALWAYS add a minimum threshold when the question is regarding pitchers.
As long as the word "career" isn't used in the question, include a filter for data only in 2024.
For certain stats like ERA, OPS, Batting Average, and others, you cannot just average the rows. You have to sum the cumulative components of the stats first.
There is a DIFFERENCE between called strikes and called strikeouts. Similarly there is a difference between swinging strikes and swinging strikeouts.
Every time it asks for some average (like batting average, earned runs average...), percentage (like on base percentage...), or rate (like flyball rate, strikeout rate...), you have to look for the formula and do the division
When asked regarding pitchers (if the question has "pitcher" or "allowed" in it, you should consider only starting pitchers (starter=1). Unless the question specifically asks consider relievers

ON EVERY QUERY, MAKE SURE TO INCLUDES CHECKS TO NOT DIVIDE BY ZERO. The numerator can be zero at times but if the denominator is zero that's a problem. 
Ensure all non-aggregated columns in the SELECT statement are included in the GROUP BY clause. In nested queries, match the GROUP BY in the outer query with any non-aggregated columns from the subquery to avoid GroupingError.
There are no NaNs in the tables, so never use "NOT NULL" in any queries
When grouping by to calculate a percentage or a rate, always use a minimum threshold
When generating SQL queries involving mathematical operations, especially division, always ensure that all operands are explicitly cast to the same type (e.g., FLOAT) to avoid type mismatch errors. Use CAST as FLOAT on both
Any time the question is asking for "best," "worst," "highest," or "lowest,", return the top 10 results, not just 1.

Use this glossary for baseball formulas: {}
mlb_pitching_dataframe columns are {}

Using these tables, return a SQL query that will answer the question.

RETURN ONLY the SQL query, NOTHING else.


""".format(table2,baseball_stats,pitching_columns,
          )


content3 = """
    
You are an expert on baseball data.
You have to use plate_appearance_dataframe to answer a question
plate_appearance_dataframe is here {}  
plate_appearance_dataframe has per plate appearance information for both batter and pitchers. Each row represents a single plate appearance between a batter and a pitcher.

Also, use common sense when someone asks subjective questions like "best", "worst","highest","lowest". There's a minimum threshold of ab or pa that should be considered before annointing someone as the "best" or "worst" or "highest" or "lowest".
Any time you're answering a question that asks the "best","worst","highest" or "lowest", YOU MUST ALWAYS use an appropriate minimum threshold of ab or pa for both pitchers or batters (both allowed or hitted stats)
You should also ALWAYS add a minimum threshold when the question is regarding pitchers.
As long as the word "career" isn't used in the question, include a filter for data only in 2024.
For certain stats like ERA, OPS, Batting Average, and others, you cannot just average the rows. You have to sum the cumulative components of the stats first.
There is a DIFFERENCE between called strikes and called strikeouts. Similarly there is a difference between swinging strikes and swinging strikeouts.
Every time it asks for some average (like batting average, earned runs average...), percentage (like on base percentage...), or rate (like flyball rate, strikeout rate...), you have to look for the formula and do the division
When asked regarding pitchers (if the question has "pitcher" or "allowed" in it, you should consider only starting pitchers (starter=1). Unless the question specifically asks consider relievers

ON EVERY QUERY, MAKE SURE TO INCLUDES CHECKS TO NOT DIVIDE BY ZERO. The numerator can be zero at times but if the denominator is zero that's a problem. 
Ensure all non-aggregated columns in the SELECT statement are included in the GROUP BY clause. In nested queries, match the GROUP BY in the outer query with any non-aggregated columns from the subquery to avoid GroupingError.
There are no NaNs in the tables, so never use "NOT NULL" in any queries
When grouping by to calculate a percentage or a rate, always use a minimum threshold
When generating SQL queries involving mathematical operations, especially division, always ensure that all operands are explicitly cast to the same type (e.g., FLOAT) to avoid type mismatch errors. Use CAST as FLOAT on both
Any time the question is asking for "best," "worst," "highest," or "lowest,", return the top 10 results, not just 1.
Use p_throws column to get pitcher handness
Use batter_team_name if the question asks about "teams against ... pitching/pitchers" 
Use pitcher_team_name if the question asks about "teams against ... batting/batters" 
Use batter_name if the question asks about "players/batters against ... pitching/pitchers" 
Use pitcher_name if the question asks about "playes/pitchers against ... batting/batters" 
When asked about number of hits, home runs, doubles, walks, you always have to sum and not count

Use this glossary for baseball formulas: {}
plate_appearance_dataframe columns are {}

Using that table, return a SQL query that will answer the question.

RETURN ONLY the SQL query, NOTHING else.

""".format(table3,baseball_stats,pa_columns,
          )

content4 = """
    
You are an expert on baseball data.
You have to use pitch_dataframe to answer a question
pitch_dataframe is here {}  
pitch_dataframe has per pitch information for both batter and pitchers. Each row represents a single pitch thrown by a pitcher against a certain batter.

Also, use common sense when someone asks subjective questions like "best", "worst","highest","lowest". There's a minimum threshold of ab or pa that should be considered before annointing someone as the "best" or "worst" or "highest" or "lowest".
Any time you're answering a question that asks the "best","worst","highest" or "lowest", YOU MUST ALWAYS use an appropriate minimum threshold of ab or pa for both pitchers or batters (both allowed or hitted stats)
You should also ALWAYS add a minimum threshold when the question is regarding pitchers.
As long as the word "career" isn't used in the question, include a filter for data only in 2024.
For certain stats like ERA, OPS, Batting Average, and others, you cannot just average the rows. You have to sum the cumulative components of the stats first.
There is a DIFFERENCE between called strikes and called strikeouts. Similarly there is a difference between swinging strikes and swinging strikeouts.
Every time it asks for some average (like batting average, earned runs average...), percentage (like on base percentage...), or rate (like flyball rate, strikeout rate...), you have to look for the formula and do the division
When asked regarding pitchers (if the question has "pitcher" or "allowed" in it, you should consider only starting pitchers (starter=1). Unless the question specifically asks consider relievers

ON EVERY QUERY, MAKE SURE TO INCLUDES CHECKS TO NOT DIVIDE BY ZERO. The numerator can be zero at times but if the denominator is zero that's a problem. 
Ensure all non-aggregated columns in the SELECT statement are included in the GROUP BY clause. In nested queries, match the GROUP BY in the outer query with any non-aggregated columns from the subquery to avoid GroupingError.
There are no NaNs in the tables, so never use "NOT NULL" in any queries
When grouping by to calculate a percentage or a rate, always use a minimum threshold
When generating SQL queries involving mathematical operations, especially division, always ensure that all operands are explicitly cast to the same type (e.g., FLOAT) to avoid type mismatch errors. Use CAST as FLOAT on both
Any time the question is asking for "best," "worst," "highest," or "lowest,", return the top 10 results, not just 1.
Use p_throws column to get pitcher handness
Use batter_team_name if the question asks about "teams against ... pitching/pitchers" 
Use pitcher_team_name if the question asks about "teams against ... batting/batters" 
Use batter_name if the question asks about "players/batters against ... pitching/pitchers" 
Use pitcher_name if the question asks about "playes/pitchers against ... batting/batters" 

Use this glossary for baseball formulas: {}
pitch_dataframe columns are {}

Using that table, return a SQL query that will answer the question.

RETURN ONLY the SQL query, NOTHING else.

""".format(table4,baseball_stats,pitch_df_columns,
          )

content5 = """
You are an expert on football data.
You have to use nfl_main_df to answer a question
nfl_main_df is here {}  
nfl_main_df is on a per game level. Each row ahve stats of a player on a certain game
Always use opponent_team_name in nfl_main_df when you need to get any allowed stat (or when the question asks about defense stats)
nfl_main_df columns are {}
Always use home variable to get home or away stats

Also, use common sense when someone asks subjective questions like "best", "worst","highest","lowest". There's a minimum threshold of ab or pa that should be considered before annointing someone as the "best" or "worst" or "highest" or "lowest".
Any time you're answering a question that asks the "best","worst","highest" or "lowest", YOU MUST ALWAYS use an appropriate minimum threshold of ab or pa for both pitchers or batters (both allowed or hitted stats)
As long as the word "career" isn't used in the question, include a filter for data only in 2024.
Every time it asks for some average (like completion percentage...), percentage (like on base percentage...), or rate (like flyball rate, strikeout rate...), you have to look for the formula and do the division

ON EVERY QUERY, MAKE SURE TO INCLUDES CHECKS TO NOT DIVIDE BY ZERO. The numerator can be zero at times but if the denominator is zero that's a problem. 
Ensure all non-aggregated columns in the SELECT statement are included in the GROUP BY clause. In nested queries, match the GROUP BY in the outer query with any non-aggregated columns from the subquery to avoid GroupingError.
There are no NaNs in the tables, so never use "NOT NULL" in any queries
When grouping by to calculate a percentage or a rate, always use a minimum threshold
When generating SQL queries involving mathematical operations, especially division, always ensure that all operands are explicitly cast to the same type (e.g., FLOAT) to avoid type mismatch errors. Use CAST as FLOAT on both
Any time the question is asking for "best," "worst," "highest," or "lowest,", return the top 10 results, not just 1.
When adding a threshold for QBs always use passes, for receivers always use targets, for RBs always use carries


Using that table, return a SQL query that will answer the question.

RETURN ONLY the SQL query, NOTHING else.


""".format(table5,nfl_cols,
          )

content6 = """
    
You are an expert on football data.
You have to use nfl_play_by_play to answer a question
nfl_play_by_play is here {}  
nfl_play_by_play is on a per play level. Each row have stats of a certain play
Always use opponent_team_name in nfl_main_df when you need to get any allowed stat (or when the question asks about defense stats)
nfl_play_by_play columns are {}
Always use home variable to get home or away stats

game_datetime is about the start time of the game
To get quarter use the column quarter, and to get remaining time use game_seconds_remaining
Also, use common sense when someone asks subjective questions like "best", "worst","highest","lowest". There's a minimum threshold of ab or pa that should be considered before annointing someone as the "best" or "worst" or "highest" or "lowest".
Any time you're answering a question that asks the "best","worst","highest" or "lowest", YOU MUST ALWAYS use an appropriate minimum threshold of ab or pa for both pitchers or batters (both allowed or hitted stats)
As long as the word "career" isn't used in the question, include a filter for data only in 2024.
Every time it asks for some average (like completion percentage...), percentage (like on base percentage...), or rate (like flyball rate, strikeout rate...), you have to look for the formula and do the division

ON EVERY QUERY, MAKE SURE TO INCLUDES CHECKS TO NOT DIVIDE BY ZERO. The numerator can be zero at times but if the denominator is zero that's a problem. 
Ensure all non-aggregated columns in the SELECT statement are included in the GROUP BY clause. In nested queries, match the GROUP BY in the outer query with any non-aggregated columns from the subquery to avoid GroupingError.
There are no NaNs in the tables, so never use "NOT NULL" in any queries
When grouping by to calculate a percentage or a rate, always use a minimum threshold
When generating SQL queries involving mathematical operations, especially division, always ensure that all operands are explicitly cast to the same type (e.g., FLOAT) to avoid type mismatch errors. Use CAST as FLOAT on both
Any time the question is asking for "best," "worst," "highest," or "lowest,", return the top 10 results, not just 1.
When adding a threshold for QBs always use passes, for receivers always use targets, for RBs always use carries


Using that table, return a SQL query that will answer the question.

RETURN ONLY the SQL query, NOTHING else.



""".format(table6,nfl_pbp_cols
          )

content7 = """
    
You are an expert on basketball data.
You have to use nba_main_dataframe to answer a question
nba_main_dataframe is here {}  
nba_main_dataframe is on a per game level. Each row have stats of a player on a certain game
nba_main_dataframe columns are {}

Also, use common sense when someone asks subjective questions like "best", "worst","highest","lowest". There's a minimum threshold of ab or pa that should be considered before annointing someone as the "best" or "worst" or "highest" or "lowest".
Any time you're answering a question that asks the "best","worst","highest" or "lowest", YOU MUST ALWAYS use an appropriate minimum threshold of ab or pa for both pitchers or batters (both allowed or hitted stats)
As long as the word "career" isn't used in the question, use max(season) to filter
Every time it asks for some average (like field goal percentage...), percentage, or rate, you have to look for the formula and do the division
When it asks for the player with the better or worse average you MUST ALWAYS include a threshold
Dont groupby by both team_name and opponent_name

When asked about this season, always use the max(season)
When the question has season in the format 2022-23, or 2021-22. Consider only the first number.
Always use opponent_name when asked about team allowed stats (and you also have to sum up the stats)

ON EVERY QUERY, MAKE SURE TO INCLUDES CHECKS TO NOT DIVIDE BY ZERO. The numerator can be zero at times but if the denominator is zero that's a problem. 
Ensure all non-aggregated columns in the SELECT statement are included in the GROUP BY clause. In nested queries, match the GROUP BY in the outer query with any non-aggregated columns from the subquery to avoid GroupingError.
There are no NaNs in the tables, so never use "NOT NULL" in any queries
When grouping by to calculate a percentage or a rate, always use a minimum threshold
When generating SQL queries involving mathematical operations, especially division, always ensure that all operands are explicitly cast to the same type (e.g., FLOAT) to avoid type mismatch errors. Use CAST as FLOAT on both
Any time the question is asking for "best," "worst," "highest," or "lowest,", return the top 10 results, not just 1.
When filtering position use LIKE, not equal

Using that table, return a SQL query that will answer the question.

RETURN ONLY the SQL query, NOTHING else.

""".format(table7,nba_main_cols
          )

content8 = """
    
You are an expert on basketball data.
You have to use nba_play_by_play to answer a question
nba_play_by_play is here {}  
nba_play_by_play is on a per play level. Each row have stats of a player on a certain play
nba_play_by_play columns are {}

Also, use common sense when someone asks subjective questions like "best", "worst","highest","lowest". There's a minimum threshold of ab or pa that should be considered before annointing someone as the "best" or "worst" or "highest" or "lowest".
Any time you're answering a question that asks the "best","worst","highest" or "lowest", YOU MUST ALWAYS use an appropriate minimum threshold of ab or pa for both pitchers or batters (both allowed or hitted stats)
As long as the word "career" isn't used in the question, use max(season) to filter
When asked about this season, always use the max(season)
Every time it asks for some average (like field goal percentage...), percentage, or rate, you have to look for the formula and do the division
When it asks for the player with the better or worse average you MUST ALWAYS include a threshold
Dont groupby by both team_name and opponent_name

When the question has season in the format 2022-23, or 2021-22. Consider only the first number.
Always use defense_team_name when asked about team allowed stats (and you also have to sum up the stats)

ON EVERY QUERY, MAKE SURE TO INCLUDES CHECKS TO NOT DIVIDE BY ZERO. The numerator can be zero at times but if the denominator is zero that's a problem. 
Ensure all non-aggregated columns in the SELECT statement are included in the GROUP BY clause. In nested queries, match the GROUP BY in the outer query with any non-aggregated columns from the subquery to avoid GroupingError.
There are no NaNs in the tables, so never use "NOT NULL" in any queries
When grouping by to calculate a percentage or a rate, always use a minimum threshold
When generating SQL queries involving mathematical operations, especially division, always ensure that all operands are explicitly cast to the same type (e.g., FLOAT) to avoid type mismatch errors. Use CAST as FLOAT on both
Any time the question is asking for "best," "worst," "highest," or "lowest,", return the top 10 results, not just 1.
When filtering position use LIKE, not equal

Using that table, return a SQL query that will answer the question.

RETURN ONLY the SQL query, NOTHING else.


""".format(table8,nba_pbp_cols
          )


# @app.before_request
# def enforce_https():
#     if not request.is_secure and not app.debug:
#         return redirect(request.url.replace("http://", "https://"), code=301)

@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = 'https://aibettingedge.com'
    response.headers['Access-Control-Allow-Methods'] = 'POST, GET, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type,Authorization'
    response.headers['Access-Control-Allow-Credentials'] = 'true'
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

@app.route('/webhook', methods=['POST', 'OPTIONS'])
def webhook():
    # Handle preflight OPTIONS request for CORS
    if request.method == 'OPTIONS':
        response = make_response()
        response.headers['Access-Control-Allow-Origin'] = 'https://aibettingedge.com'
        response.headers['Access-Control-Allow-Methods'] = 'POST, GET, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type,Authorization'
        response.headers['Access-Control-Allow-Credentials'] = 'true'
        return response
    
    # Handle the POST request
    data = request.get_json()
    question = data['key']
    print_table_record_counts()

    for val in range(3):
        try:
            table_to_use = chatGPT(question,content)
            query = chatGPT1(question, table_to_use)
            result = execute_query(query)
            full_result = chatGPT2(question, result)
            return jsonify({'status': full_result, 'query': query})

        except:
            print('trying again')

def insert_feedback(answer, question, query):
    DATABASE_URL = os.environ['DATABASE_URL']
    conn = None
    try:
        # Connect to the database
        conn = psycopg2.connect(DATABASE_URL, sslmode='require')
        cursor = conn.cursor()

        # Insert the feedback data into the sportgpt_results table
        cursor.execute("""
            INSERT INTO sportgpt_results (answer, question, query)
            VALUES (%s, %s, %s)
        """, (answer, question, query))

        # Commit the transaction
        conn.commit()

        # Close the cursor
        cursor.close()
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Close the connection
        if conn is not None:
            conn.close()

@app.route('/feedback', methods=['POST'])
def feedback():
    data = request.get_json()
    answer = data['answer']
    question = data['question']
    query = data['query']

    # Insert the data into the database
    insert_feedback(answer, question, query)

    return jsonify({"status": "success", "message": "Feedback saved successfully."})


def connect_to_db():
    DATABASE_URL = os.environ['DATABASE_URL']

    # Parse the database URL
    result = urlparse(DATABASE_URL)
    username = result.username
    password = result.password
    database = result.path[1:]
    hostname = result.hostname
    port = result.port

    # Connect to the database
    connection = psycopg2.connect(
        database=database,
        user=username,
        password=password,
        host=hostname,
        port=port
    )

    return connection

def get_table_names(connection):
    query = """
    SELECT table_name
    FROM information_schema.tables
    WHERE table_schema='public'
    """
    cursor = connection.cursor()
    cursor.execute(query)
    tables = cursor.fetchall()
    cursor.close()
    return [table[0] for table in tables]

def print_all_rows(connection, table_name):
    query = f"SELECT * FROM {table_name}"
    cursor = connection.cursor()
    cursor.execute(query)
    rows = cursor.fetchall()
    cursor.close()
    
    print(f"All rows from table {table_name}:")
    for row in rows:
        print(row)

def get_record_count(connection, table_name):
    query = f"SELECT COUNT(*) FROM {table_name}"
    cursor = connection.cursor()
    cursor.execute(query)
    count = cursor.fetchone()[0]
    cursor.close()
    return count

def print_table_record_counts():
    connection = connect_to_db()
    try:
        table_names = get_table_names(connection)
        for table_name in table_names:
            count = get_record_count(connection, table_name)
            print(f"Table {table_name} has {count} records.")
            
            # Check if the table name is 'sportgpt_results'
            if table_name == 'sportgpt_results':
                print_all_rows(connection, table_name)
                
    finally:
        connection.close()

# Call the function to print the record counts

def execute_query(query):
    print(query)
    connection = connect_to_db()
    cursor = connection.cursor()
    cursor.execute(query)
    result = cursor.fetchall()
    cursor.close()
    connection.close()
    return str(result)

def chatGPT(question,content):

    client = OpenAI(
    api_key = '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',
  organization='org-E00A8t9oYwKizNWaI7Wzzu28',
  project='proj_bNm4yh06r27jDgWQuKudK7MO'
)

    print("hi!")
    

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": content},
            {"role": "user", "content": question}
        ],
        max_tokens=2000)


    try:
        table_to_use = response.choices[0].message.content
    except:
        table_to_use = 'No table for this available'

    return table_to_use


def chatGPT1(question,table_to_use):

    client = OpenAI(
    api_key = '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',
  organization='org-E00A8t9oYwKizNWaI7Wzzu28',
  project='proj_bNm4yh06r27jDgWQuKudK7MO'
)
    
    if table_to_use=="mlb_batting_dataframe":
        
        response = client.chat.completions.create(
            model="gpt-4o-2024-05-13",
            messages=[
                {"role": "system", "content": content1},
                {"role": "user", "content": question}
            ],
            max_tokens=2000)

        
    elif table_to_use=="mlb_pitching_dataframe":
        
        response = client.chat.completions.create(
            model="gpt-4o-2024-05-13",
            messages=[
                {"role": "system", "content": content2},
                {"role": "user", "content": question}
            ],
            max_tokens=2000)

            
    elif table_to_use=="plate_appearance_dataframe":
        
        response = client.chat.completions.create(
            model="gpt-4o-2024-05-13",
            messages=[
                {"role": "system", "content": content3},
                {"role": "user", "content": question}
            ],
            max_tokens=2000)


    elif table_to_use=="pitch_dataframe":
        
        response = client.chat.completions.create(
            model="gpt-4o-2024-05-13",
            messages=[
                {"role": "system", "content": content4},
                {"role": "user", "content": question}
            ],
            max_tokens=2000)

            
    elif table_to_use=="nfl_main_df":
        
        response = client.chat.completions.create(
            model="gpt-4o-2024-05-13",
            messages=[
                {"role": "system", "content": content5},
                {"role": "user", "content": question}
            ],
            max_tokens=2000)

                    
    elif table_to_use=="nfl_play_by_play":
        
        response = client.chat.completions.create(
            model="gpt-4o-2024-05-13",
            messages=[
                {"role": "system", "content": content6},
                {"role": "user", "content": question}
            ],
            max_tokens=2000)

    elif table_to_use=="nba_main_dataframe":
        
        response = client.chat.completions.create(
            model="gpt-4o-2024-05-13",
            messages=[
                {"role": "system", "content": content7},
                {"role": "user", "content": question}
            ],
            max_tokens=2000)

    elif table_to_use=="nba_play_by_play":
        
        response = client.chat.completions.create(
            model="gpt-4o-2024-05-13",
            messages=[
                {"role": "system", "content": content8},
                {"role": "user", "content": question}
            ],
            max_tokens=2000)


    try:
        text_response = response.choices[0].message.content
        sql_pattern = re.compile(r'```sql\n(.*?)```', re.DOTALL)
        match = sql_pattern.search(text_response)
        query = match.group(1)
    except:
        query = 'No query for this available'

    return query


def chatGPT2(question,result_string):

    content_2 = '''
You are a sports data expert. 
You will be given the answer to a question, or a list of potential answers to choose from in a string representation of a pandas dataframe. 
Your job is to choose the answer and turn it into an English response.
Always make sure that you consider the sample size when providing an answer.
If the answer has a small relative sample size, return other potential answers too.
If the dataframe includes a dummy variable (a binary column with values like 0 or 1 that indicates the presence or absence of a condition), ensure that you correctly interpret it in the context of the question
The original question is here {}
'''.format(question)

    client = OpenAI(
    api_key = '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',
  organization='org-E00A8t9oYwKizNWaI7Wzzu28',
  project='proj_bNm4yh06r27jDgWQuKudK7MO'
)
    
    response = client.chat.completions.create(
    model="gpt-4o-2024-05-13",
    messages=[
        {"role": "system", "content": content_2},
        {"role": "user", "content":result_string }
    ],
    max_tokens=2000)

    return response.choices[0].message.content




if __name__ == '__main__':
    app.run()