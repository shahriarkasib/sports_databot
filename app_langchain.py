from flask import Flask, render_template, request, Response, stream_with_context
import time
import queue
import threading
from langgraph.checkpoint.memory import MemorySaver
from IPython.display import Image, display
from langgraph.graph import MessagesState
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
import json
from dotenv import load_dotenv
from datetime import datetime
import re
import warnings
import psycopg2
from urllib.parse import urlparse

from langsmith import traceable
import os
from postgre_connection import engine  # Import the PostgreSQL engine

load_dotenv()

# Database credentials are now imported from postgre_connection.py

# Ignore all warnings of a specific type
warnings.filterwarnings("ignore", category=DeprecationWarning)

@traceable(run_type="llm", name="DeepSeek R1 8B")
def get_llm():
    return ChatOllama(model="deepseek-r1:8b", temperature= 0.0)

llm = get_llm()

class State(MessagesState):
    table_metadata: str
    tables: list
    sql: str 
    status: str 
    data: dict
    status_message: str
    result: list
    sql_verifier_response: str
    recursion: int
    table_documentation: dict
    filtered_table_docs: dict
    
# Initialize the StateGraph
graph = StateGraph(State)

@traceable(run_type="tool", name="Load Table Documentation")
def get_table_doc(state: State):
    progress_queue.put("Step 1/8: Loading table documentation...")
    try:
        with open("table_documentation/merged_table_documentation.json", "r", encoding="utf-8") as f:
            merged_doc = json.load(f)
    except Exception as e:
        merged_doc = {"error": f"Failed to load merged documentation: {e}"}
    # Return the loaded documentation in the state along with any necessary fields.
    return {"table_documentation": merged_doc, "recursion": 0, "sql_verifier_response": None}

@traceable(run_type="tool", name="Select Tables")
def select_tables(state: State):
    progress_queue.put("Step 2/8: Selecting relevant tables...")
    """
    Selects the most relevant tables for the user's question based on conversation history.
    Returns only table names in comma-separated format.
    Ensures that tables from only one sport type are selected (NBA, MLB, or NFL).
    Uses conversation history to maintain context across multiple questions.
    """
    # Ensure re module is available
    import re
    
    # Extract user's question and conversation history
    user_query = state["messages"][-1].content
    
    # Get conversation history (up to 3 previous messages if available)
    conversation_history = []
    messages = state.get("messages", [])
    
    # Extract previous user messages (max 3 most recent)
    history_limit = 3
    previous_questions = []
    
    for message in reversed(messages[:-1]):  # Skip the current message
        if isinstance(message, HumanMessage) and len(previous_questions) < history_limit:
            previous_questions.append(message.content)
            
    # Reverse to get chronological order
    previous_questions = list(reversed(previous_questions))
    
    # Get all available tables from the full_table_docs dictionary
    full_table_docs = state.get("table_documentation", {})
    available_tables = list(full_table_docs.keys())
    
    # Define sport-specific tables
    nba_tables = [table for table in available_tables if table.startswith("nba_")]
    mlb_tables = [table for table in available_tables if table.startswith("mlb_") or table in ["plate_appearance_dataframe", "pitch_dataframe"]]
    nfl_tables = [table for table in available_tables if table.startswith("nfl_")]
    
    # Construct the prompt with context from previous conversations
    prompt = f"""
    I need to select the most relevant database tables for this question: "{user_query}"
    
    PREVIOUS CONVERSATION CONTEXT (most recent first):
    """
    
    # Add previous questions to the prompt
    if previous_questions:
        prompt += "\n"
        for question in previous_questions:
            prompt += f"- {question}\n"
    
    prompt += """
    TABLE SELECTION RULES:
    1. Select ONLY tables that are relevant to answering the question above
    2. If the question relates to a specific sport based on the current question or previous context, ONLY select tables for that sport
    3. DO NOT mix tables from different sports (NBA, MLB, NFL) in the same query
    4. If the question is vague but previous questions were about a specific sport, maintain that context
    
    Available tables:
    
    MLB (Baseball) Tables:
    - mlb_batting_dataframe: Contains batting statistics for baseball players
    - mlb_pitching_dataframe: Contains pitching statistics for baseball pitchers
    - plate_appearance_dataframe: Contains per-plate appearance matchups between batters and pitchers
    - pitch_dataframe: Contains detailed pitch-by-pitch data
    
    NFL (Football) Tables:
    - nfl_main_df: Contains per-game statistics for football players
    - nfl_play_by_play: Contains play-by-play details for football games
    
    NBA (Basketball) Tables:
    - nba_main_dataframe: Contains per-game statistics for basketball players
    - nba_play_by_play: Contains play-by-play details for basketball games
    
    Special Case Selection Rules:
    
    FOR MLB Data:
    - If question is about pitching stats, include mlb_pitching_dataframe
    - If question is about batting stats, include mlb_batting_dataframe
    - If question asks about batter vs pitcher matchups, include plate_appearance_dataframe
    - If question is about specific pitch types or velocities, include pitch_dataframe
    
    FOR NFL Data:
    - If question is about football plays or down-by-down analysis, include nfl_play_by_play
    - Otherwise for general NFL player stats, include nfl_main_df
    
    FOR NBA Data:
    - If question is about basketball play-by-play events or time-specific moments, include nba_play_by_play
    - Otherwise for general NBA player or team stats, include nba_main_dataframe
    
    RESPONSE FORMAT:
    Return ONLY the table names separated by commas without any other text, commentary, or explanation.
    For example: "nba_main_dataframe,nba_play_by_play" or "mlb_batting_dataframe" or "nfl_main_df"
    """
    
    # Call the LLM to get the tables
    llm = get_llm()
    response = llm.invoke(prompt)
    response_text = response.content.strip()
    print(f"Selected tables (raw): {response_text}")
    
    # Parse the response to get the table names
    # Remove any quotes, backticks, or code blocks that might be in the response
    response_text = response_text.replace('`', '').replace('"', '').replace("'", "")
    response_text = re.sub(r'```.*?```', '', response_text, flags=re.DOTALL)
    response_text = re.sub(r'```.*', '', response_text, flags=re.DOTALL)
    
    # Split by commas and strip whitespace
    tables = [table.strip() for table in response_text.split(',')]
    
    # Validate the selected tables against available tables
    valid_tables = [table for table in tables if table in available_tables]
    
    # Handle case where tables from mixed sports might be selected
    nba_selected = [table for table in valid_tables if table in nba_tables]
    mlb_selected = [table for table in valid_tables if table in mlb_tables]
    nfl_selected = [table for table in valid_tables if table in nfl_tables]
    
    # Determine which sport's tables to use based on count
    sport_counts = {
        "NBA": len(nba_selected),
        "MLB": len(mlb_selected),
        "NFL": len(nfl_selected)
    }
    
    dominant_sport = max(sport_counts.items(), key=lambda x: x[1])[0]
    
    # Special case handling based on keywords in question and previous context
    question_lower = user_query.lower()
    context_lower = " ".join(previous_questions).lower()
    combined_context = question_lower + " " + context_lower
    
    if dominant_sport == "NBA" or "basketball" in combined_context or "nba" in combined_context:
        if not nba_selected:
            valid_tables = ["nba_main_dataframe"]
        else:
            valid_tables = nba_selected
    elif dominant_sport == "MLB" or "baseball" in combined_context or "mlb" in combined_context:
        if not mlb_selected:
            # Special case: For team record questions in MLB, prefer pitching dataframe which has wins/losses
            if any(term in question_lower for term in ["record", "standing", "best team", "win percentage"]):
                valid_tables = ["mlb_pitching_dataframe"]
            else:
                valid_tables = ["mlb_batting_dataframe"]
        else:
            valid_tables = mlb_selected
    elif dominant_sport == "NFL" or "football" in combined_context or "nfl" in combined_context:
        if not nfl_selected:
            valid_tables = ["nfl_main_df"]
        else:
            valid_tables = nfl_selected
    
    # If no valid tables detected and the question is about team records or standings
    if not valid_tables and any(term in question_lower for term in 
                             ["record", "standing", "best team", "win percentage", "winning record"]):
        # Check previous context first to determine sport
        if "basketball" in combined_context or "nba" in combined_context:
            valid_tables = ["nba_main_dataframe"]
        elif "baseball" in combined_context or "mlb" in combined_context:
            # For baseball team records, pitching dataframe has direct win/loss columns
            valid_tables = ["mlb_pitching_dataframe"]
        elif "football" in combined_context or "nfl" in combined_context:
            valid_tables = ["nfl_main_df"]
        else:
            # Default to NBA if no specific context is available for team record questions
            valid_tables = ["nba_main_dataframe"]
    
    print(f"Selected and validated tables: {valid_tables}")
    
    # Get recursion count and increment it
    recursion = state.get("recursion", 0) + 1
    
    return {"tables": valid_tables, "recursion": recursion}

@traceable(run_type="tool", name="Get Table Documentation")
def get_table_documentation(state: State):
    progress_queue.put("Step 4/8: Fetching table documentation...")
    """
    Fetch documentation for the selected tables.
    Enhances documentation with additional sport-specific context and relationships.
    Includes comprehensive metric documentation for each sport.
    """
    tables = state.get("tables", [])
    full_table_docs = state.get("table_documentation", {})
    
    # Baseball stats glossary from app_prev.py
    baseball_stats = {
        "Hits (H)": "A hit is a single, double, triple or home_run. Use column h or h_allowed (only in mlb_pitching_dataframe)",
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
        "Strike Ratio": "strike/(strike+ball)"
    }
    
    # Create filtered documentation dictionary
    filtered_docs = {}
    for table in tables:
        if table in full_table_docs:
            filtered_docs[table] = full_table_docs[table].copy()  # Create a copy to avoid modifying original
            
            # Add sport type information based on table name
            if table.startswith("nba_"):
                filtered_docs[table]["sport_type"] = "Basketball (NBA)"
                
                # Add comprehensive NBA metrics documentation
                if table == "nba_main_dataframe":
                    filtered_docs[table]["important_notes"] = [
                        "Points are stored in 'pts' column, not 'points'",
                        "Minutes played are stored in 'min' column",
                        "Field goals made/attempted are in 'fgm'/'fga' columns",
                        "Rebounds are in 'reb' column (offensive: 'oreb', defensive: 'dreb')",
                        "When asked about this season, always use the max(season) to filter",
                        "When the question has season in the format 2022-23, or 2021-22, consider only the first number",
                        "Always use opponent_name when asked about team allowed stats (and also sum up the stats)",
                        "Don't group by both team_name and opponent_name in the same query",
                        "When filtering position use LIKE, not equals"
                    ]
                    
                    # Add detailed metrics documentation
                    filtered_docs[table]["metrics_documentation"] = {
                        "scoring_metrics": [
                            {"name": "pts", "description": "Total points scored by the player"},
                            {"name": "fgm", "description": "Field goals made"},
                            {"name": "fga", "description": "Field goals attempted"},
                            {"name": "three_pointer_made", "description": "Three-point shots made"},
                            {"name": "three_pointer_attempt", "description": "Three-point shots attempted"},
                            {"name": "ftm", "description": "Free throws made"},
                            {"name": "fta", "description": "Free throws attempted"}
                        ],
                        "efficiency_metrics": [
                            {"name": "shooting_percentage", "formula": "fgm / fga", "description": "Field goal percentage"},
                            {"name": "three_point_percentage", "formula": "three_pointer_made / three_pointer_attempt", "description": "Three-point shooting percentage"},
                            {"name": "free_throw_percentage", "formula": "ftm / fta", "description": "Free throw percentage"},
                            {"name": "true_shooting", "formula": "pts / (2 * (fga + 0.44 * fta))", "description": "True shooting percentage accounting for all scoring attempts"}
                        ],
                        "other_key_metrics": [
                            {"name": "reb", "description": "Total rebounds"},
                            {"name": "ast", "description": "Total assists"},
                            {"name": "stl", "description": "Total steals"},
                            {"name": "blk", "description": "Total blocks"},
                            {"name": "tov", "description": "Total turnovers"},
                            {"name": "plus_minus", "description": "Plus-minus statistic (team point differential while player was on court)"}
                        ],
                        "thresholds": [
                            {"metric": "min", "min_value": 10, "description": "Minimum minutes played for per-game analysis"},
                            {"metric": "fga", "min_value": 5, "description": "Minimum field goal attempts for shooting percentage analysis"}
                        ],
                        "query_requirements": [
                            "When calculating percentages, cast both operands to FLOAT to avoid type mismatch errors",
                            "For 'best', 'worst', 'highest', or 'lowest' questions, return top 10 results, not just 1",
                            "Ensure all non-aggregated columns in SELECT are included in GROUP BY clause",
                            "Include checks to not divide by zero in all calculations"
                        ]
                    }
                    
                elif table == "nba_play_by_play":
                    filtered_docs[table]["important_notes"] = [
                        "Each row represents a single play in a game",
                        "Use defense_team_name when asked about team allowed stats",
                        "When asked about this season, always use the max(season) to filter",
                        "When the question has season in the format 2022-23, or 2021-22, consider only the first number"
                    ]
                
            elif table.startswith("mlb_"):
                filtered_docs[table]["sport_type"] = "Baseball (MLB)"
                
                # Add MLB baseball stats glossary
                filtered_docs[table]["baseball_stats_glossary"] = baseball_stats
                
                # Add specific notes for MLB tables
                if table == "mlb_batting_dataframe":
                    filtered_docs[table]["important_notes"] = [
                        "Home runs are stored in 'hr' column",
                        "Runs batted in are in 'rbi' column",
                        "Batting average is in 'avg' column",
                        "Each row represents one game's batting stats for a single batter",
                        "Apply minimum threshold (ab >= 3) for meaningful batting analysis",
                        "Unless 'career' is in the question, filter data for current year",
                        "For averages like batting average, OPS, etc., sum components before division",
                        "When asked for 'best', 'worst', 'highest', or 'lowest', use minimum threshold and return top 10"
                    ]
                    
                    # Add detailed metrics documentation for batting
                    filtered_docs[table]["metrics_documentation"] = {
                        "counting_stats": [
                            {"name": "ab", "description": "At bats - official plate appearances excluding walks, hit-by-pitch, sacrifices"},
                            {"name": "h", "description": "Hits - successful at bats reaching base safely"},
                            {"name": "r", "description": "Runs scored"},
                            {"name": "hr", "description": "Home runs"},
                            {"name": "rbi", "description": "Runs batted in"},
                            {"name": "bb", "description": "Walks (base on balls)"},
                            {"name": "k", "description": "Strikeouts"}
                        ],
                        "rate_stats": [
                            {"name": "avg", "formula": "h / ab", "description": "Batting average - hits divided by at bats"},
                            {"name": "obp", "formula": "(h + bb + hbp) / (ab + bb + hbp + sf)", "description": "On-base percentage - rate of reaching base safely"},
                            {"name": "slg", "formula": "total_bases / ab", "description": "Slugging percentage - measure of hitting power, where total_bases = singles*1 + doubles*2 + triples*3 + hr*4"},
                            {"name": "ops", "formula": "obp + slg", "description": "On-base plus slugging - combined measure of getting on base and hitting power"}
                        ],
                        "thresholds": [
                            {"metric": "ab", "min_value": 3, "description": "Minimum at-bats for single game analysis"},
                            {"metric": "ab", "min_value": 502, "description": "Minimum at-bats for season batting title qualification (3.1 per team game)"}
                        ],
                        "query_requirements": [
                            "Cast all operands to FLOAT in division operations",
                            "For 'best', 'worst', 'highest', or 'lowest' questions, return top 10 results",
                            "Include checks to not divide by zero",
                            "Ensure all non-aggregated columns in SELECT are included in GROUP BY",
                            "Don't use NOT NULL (no NaNs in tables)"
                        ]
                    }
                    
                elif table == "mlb_pitching_dataframe":
                    filtered_docs[table]["important_notes"] = [
                        "Earned run average is in 'era' column",
                        "Innings pitched are in 'ip' column",
                        "Strikeouts are in 'k' column",
                        "Each row represents one game pitched by a pitcher",
                        "Consider only starting pitchers (starter=1) unless relievers are specifically requested",
                        "Apply minimum threshold (ip >= 1) for meaningful pitching analysis",
                        "Unless 'career' is in the question, filter data for current year",
                        "When asked for 'best', 'worst', 'highest', or 'lowest', use minimum threshold and return top 10"
                    ]
                    
                    # Add detailed metrics documentation for pitching
                    filtered_docs[table]["metrics_documentation"] = {
                        "counting_stats": [
                            {"name": "ip", "description": "Innings pitched - each out recorded counts as 1/3 inning"},
                            {"name": "er", "description": "Earned runs allowed"},
                            {"name": "k", "description": "Strikeouts recorded"},
                            {"name": "bb", "description": "Walks allowed"},
                            {"name": "h_allowed", "description": "Hits allowed"},
                            {"name": "hr_allowed", "description": "Home runs allowed"}
                        ],
                        "rate_stats": [
                            {"name": "era", "formula": "9 * (er / ip)", "description": "Earned Run Average - earned runs allowed per 9 innings"},
                            {"name": "whip", "formula": "(h_allowed + bb) / ip", "description": "Walks plus Hits per Inning Pitched - baserunners allowed per inning"},
                            {"name": "k_per_9", "formula": "9 * (k / ip)", "description": "Strikeouts per 9 innings"},
                            {"name": "bb_per_9", "formula": "9 * (bb / ip)", "description": "Walks per 9 innings"},
                            {"name": "k_bb_ratio", "formula": "k / bb", "description": "Strikeout-to-walk ratio"}
                        ],
                        "thresholds": [
                            {"metric": "ip", "min_value": 1, "description": "Minimum innings pitched for single game analysis"},
                            {"metric": "ip", "min_value": 162, "description": "Minimum innings pitched for season ERA title qualification (1 per team game)"}
                        ],
                        "query_requirements": [
                            "Cast all operands to FLOAT in division operations",
                            "For 'best', 'worst', 'highest', or 'lowest' questions, return top 10 results",
                            "Include checks to not divide by zero",
                            "Sum cumulative components of stats first before calculating rates",
                            "Use starter=1 to filter for starting pitchers unless relievers are specifically requested"
                        ]
                    }
                
                elif table == "plate_appearance_dataframe":
                    filtered_docs[table]["important_notes"] = [
                        "Each row represents a single plate appearance between a batter and pitcher",
                        "Use p_throws column to get pitcher handedness",
                        "Use batter_team_name for 'teams against pitching/pitchers'",
                        "Use pitcher_team_name for 'teams against batting/batters'",
                        "Use batter_name for 'players/batters against pitching/pitchers'",
                        "Use pitcher_name for 'players/pitchers against batting/batters'",
                        "When asked about hits, home runs, etc., use SUM not COUNT",
                        "Always use this table for questions about counts, runners on base, outs, or handedness"
                    ]
                
                elif table == "pitch_dataframe":
                    filtered_docs[table]["important_notes"] = [
                        "Each row represents a single pitch thrown by a pitcher against a batter",
                        "Use for detailed pitch analysis like whiff rate, contact rate",
                        "Use p_throws column to get pitcher handedness",
                        "Use batter_team_name for 'teams against pitching/pitchers'",
                        "Use pitcher_team_name for 'teams against batting/batters'",
                        "Use batter_name for 'players/batters against pitching/pitchers'",
                        "Use pitcher_name for 'players/pitchers against batting/batters'"
                    ]
                    
                    filtered_docs[table]["metrics_documentation"] = {
                        "pitch_metrics": [
                            {"name": "whiff_rate", "formula": "swinging_strike / swings * 100", "description": "Percentage of swings that miss the ball"},
                            {"name": "strike_ratio", "formula": "strike / (strike + ball)", "description": "Ratio of strikes to total pitches"}
                        ]
                    }
                    
            elif table.startswith("nfl_"):
                filtered_docs[table]["sport_type"] = "Football (NFL)"
                
                # Add specific notes for NFL tables
                if table == "nfl_main_df":
                    filtered_docs[table]["important_notes"] = [
                        "Each row represents one player's performance in a single game",
                        "Use opponent_team_name for team allowed stats or defense stats",
                        "Use home column (1=home, 0=away) for game location filtering",
                        "Apply position-specific thresholds: QB (passes >= 10), RB (carries >= 5), WR/TE (targets >= 3)",
                        "Unless 'career' is in the question, filter data for current year",
                        "When asked for 'best', 'worst', 'highest', or 'lowest', return top 10 results"
                    ]
                    
                    # Add detailed metrics documentation for NFL
                    filtered_docs[table]["metrics_documentation"] = {
                        "passing_metrics": [
                            {"name": "passing_yards", "description": "Total passing yards"},
                            {"name": "passes", "description": "Number of pass attempts"},
                            {"name": "completions", "description": "Number of completed passes", "calculated": "not directly stored, must be calculated if needed"},
                            {"name": "passing_touchdowns", "description": "Passing touchdowns thrown", "calculated": "may be part of touchdowns column"}
                        ],
                        "rushing_metrics": [
                            {"name": "rushing_yards", "description": "Total rushing yards"},
                            {"name": "carries", "description": "Number of rushing attempts"},
                            {"name": "rushing_touchdowns", "description": "Rushing touchdowns scored", "calculated": "may be part of touchdowns column"}
                        ],
                        "receiving_metrics": [
                            {"name": "receiving_yards", "description": "Total receiving yards"},
                            {"name": "targets", "description": "Number of times targeted for a pass"},
                            {"name": "receptions", "description": "Number of passes caught", "calculated": "not directly stored, must be calculated if needed"},
                            {"name": "receiving_touchdowns", "description": "Receiving touchdowns scored", "calculated": "may be part of touchdowns column"}
                        ],
                        "efficiency_metrics": [
                            {"name": "yards_per_carry", "formula": "rushing_yards / carries", "description": "Average yards gained per rushing attempt"},
                            {"name": "yards_per_reception", "formula": "receiving_yards / receptions", "description": "Average yards gained per reception"},
                            {"name": "completion_percentage", "formula": "completions / passes", "description": "Percentage of passes completed"}
                        ],
                        "thresholds": [
                            {"position": "QB", "metric": "passes", "min_value": 10, "description": "Minimum pass attempts for quarterback analysis"},
                            {"position": "RB", "metric": "carries", "min_value": 5, "description": "Minimum carries for running back analysis"},
                            {"position": "WR/TE", "metric": "targets", "min_value": 3, "description": "Minimum targets for receiver analysis"}
                        ],
                        "query_requirements": [
                            "Cast all operands to FLOAT in division operations",
                            "Include checks to not divide by zero",
                            "For 'best', 'worst', 'highest', or 'lowest' questions, return top 10 results",
                            "Ensure all non-aggregated columns in SELECT are included in GROUP BY"
                        ]
                    }
                
                elif table == "nfl_play_by_play":
                    filtered_docs[table]["important_notes"] = [
                        "Each row represents a single play from a specific game",
                        "game_datetime is the start time of the game",
                        "Use quarter column for quarter information",
                        "Use game_seconds_remaining for time remaining",
                        "Always use this table for questions about downs, time remaining, yards to endzone, or yards to 1st down"
                    ]
    
    # Identify relationships between selected tables
    if len(tables) > 1:
        relationships = []
        if "nba_main_dataframe" in tables and "nba_play_by_play" in tables:
            relationships.append("Join nba_main_dataframe and nba_play_by_play using player_name in nba_main_dataframe and offense_player_name in nba_play_by_play")
        
        if "mlb_batting_dataframe" in tables and "plate_appearance_dataframe" in tables:
            relationships.append("Join mlb_batting_dataframe and plate_appearance_dataframe using player_name in mlb_batting_dataframe and batter_name in plate_appearance_dataframe")
        
        if "mlb_pitching_dataframe" in tables and "plate_appearance_dataframe" in tables:
            relationships.append("Join mlb_pitching_dataframe and plate_appearance_dataframe using pitcher_name in mlb_pitching_dataframe and pitcher_name in plate_appearance_dataframe")
        
        if "plate_appearance_dataframe" in tables and "pitch_dataframe" in tables:
            relationships.append("Join plate_appearance_dataframe and pitch_dataframe using batter_name/pitcher_name in both tables")
        
        if "nfl_main_df" in tables and "nfl_play_by_play" in tables:
            relationships.append("Join nfl_main_df and nfl_play_by_play using game_date to match plays to player stats")
        
        if relationships:
            for table in filtered_docs:
                filtered_docs[table]["join_relationships"] = relationships
    
    # Return the enhanced documentation
    print(f"DEBUG - Enhanced documentation for tables: {', '.join(tables)}")
    print(f"DEBUG - filtered_docs keys: {list(filtered_docs.keys())}")
    for table in filtered_docs:
        print(f"DEBUG - Table {table} has keys: {list(filtered_docs[table].keys())}")
        if 'important_notes' in filtered_docs[table]:
            print(f"DEBUG - Table {table} important_notes: {filtered_docs[table]['important_notes'][:2]}")
    
    state_to_return = {
        "filtered_table_docs": filtered_docs,
        "recursion": state.get("recursion", 0) + 1
    }
    
    # Print what we're returning to the state
    print(f"DEBUG - Returning keys to state: {list(state_to_return.keys())}")
    
    return state_to_return

# identify if the question needs to talk with database or a general questions
@traceable(run_type="tool", name="Is Question Need Database")

def is_question_need_database(state: State):
    progress_queue.put("Step 2/8: Determining if question needs a DB query...")
    """
    Determines whether the user's question requires fetching data from the database or if it can be answered generally.
    """
    question = state["messages"][-1].content
    history = state["messages"]
    table_documentation = state.get("table_documentation")
    print(table_documentation)
    # Constructing the decision-making prompt
    prompt = f"""
    You are an expert sports data analyst specializing in MLB, NFL, and NBA performance. Your expertise includes:

- **Player Performance Insights**: Analyze per-game stats, hitting, and pitching data (MLB) to uncover key performance metrics.
- **Game & Play Analytics**: Track detailed game and play-by-play data for NFL (including downs) and NBA (per-game and play-level).
- **Comparative & Ranking Analysis**: Evaluate "best," "worst," "highest," or "lowest" metrics using strict minimum thresholds (e.g., at least 10 appearances) and filter to 2024 unless "career" is specified.
- **Table Selection Guidance**: Choose the most relevant table—mlb_batting_dataframe for hitters, mlb_pitching_dataframe for pitchers, plate_appearance_dataframe for matchup details, pitch_dataframe for pitch-level data, and the corresponding NFL/NBA tables as needed.


    Your task is to determine if the **user's question requires querying the database** or if it can be answered **without database access** using prior knowledge or conversation history.
    ### Decision Criteria:
      1. Identify if the question explicitly requests **data values**. The data that you have access to will be given as  a table_documentation to you.
      2. Check if the question requires retrieving **specific metrics from the database**. You should be able to identify if the metric can be answered from the table_documentation shared.
      3. Consider **historical conversation** to determine if the answer is already known or if it requires fetching new data.
      4. If the question is about **whether a metric exists** but doesn't require an actual number, classify it as a general question.
      5. If the question is about getting data or which needs data and relavant to the information in the table_documentation, then respond with "Database" even if the question is not asking for a specific metric or it does not contain exact information.
      Like it can ask for questions related to marketing point of view. Try to use some relevant metrics and KPIs to answer the question and respond with "Database"

    ### Response Rules:
      - If the **question requires querying the database**, respond with **"Database"**.
      - If the **question can be answered without database access**, respond with **"General"**.

    **Input Data:**
      - **Previous Conversation History (if applicable):**
      ```{history}```
    - **User Question:** {question}
    - **table_documentation: {table_documentation}
    **Now, classify the question appropriately. Return only "Database" or "General" with no extra text.**
    """

    # Invoke LLM for classification
    response = llm.invoke(prompt)
    
    print("Database Check Response:")
    print(response.content.strip())

    return response.content.strip()

@traceable(run_type="tool", name="General Answer")
def general_answer(state: State):
    progress_queue.put("Skipping all the steps, as this question do not need help from database...")
    """
    Generates a professional, concise response for general data-related questions For CustomGPT Analysos 
    using table documentation. If the information is not available, provide a professional response 
    indicating the limitation without making assumptions.
    """
    history = state["messages"]
    question = state["messages"][-1].content

    # Ensure table metadata and documentation are properly formatted as JSON
    table_doc = state.get("table_documentation")
    # Constructing the answer generation prompt
    prompt = f"""
    You are an expert sports data analyst specializing in MLB, NFL, and NBA performance. Your expertise includes:

- **Player Performance Insights**: Analyze per-game stats, hitting, and pitching data (MLB) to uncover key performance metrics.
- **Game & Play Analytics**: Track detailed game and play-by-play data for NFL (including downs) and NBA (per-game and play-level).
- **Comparative & Ranking Analysis**: Evaluate "best," "worst," "highest," or "lowest" metrics using strict minimum thresholds (e.g., at least 10 appearances) and filter to 2024 unless "career" is specified.
- **Table Selection Guidance**: Choose the most relevant table—mlb_batting_dataframe for hitters, mlb_pitching_dataframe for pitchers, plate_appearance_dataframe for matchup details, pitch_dataframe for pitch-level data, and the corresponding NFL/NBA tables as needed.


    Your task is to answer the  user's question **only if the necessary information is available in the provided 
    table documentation**. If the answer **cannot be derived from the documentation**, provide a professional 
    response but do not answer anything from your earlier knowledge except the knowledge of table documentation.

    ### Response Guidelines:
    1. **If the required information is in the table documentation, provide a clear and well-structured answer.**
    2. **If the information is NOT available, respond professionally
    
    ### Provided Information:
    - **User Question:** "{question}"
    - **Table Documentation:**  
      ```json
      {table_doc}
      ```
      ```
    - **Previous Conversation History (if needed):**  
      ```{history}```
    """

    print("General Answer Prompt:")
    print(prompt)

    # Invoke LLM for generating the answer
    response = llm.invoke(prompt)
    progress_queue.put("[DONE]")
    
    # Ensure a professional response if data is not available
    answer = response.content.strip()
    return {"messages": answer}

def connect_to_db():
    """
    Returns a connection using the engine imported from postgre_connection.py
    """
    connection = engine.raw_connection()
    return connection

def generate_sql(state: State):
    progress_queue.put("Step 5/8: Generating SQL query...")
    """
    Generates an SQL query based on selected tables and their documentation.
    Ensures proper aggregation and filtering based on sport-specific data.
    Uses comprehensive metrics documentation for accurate calculations and thresholds.
    """
    # Ensure re module is available
    import re
    
    # Debug the entire state
    print("DEBUG - Full state keys:", [k for k in dir(state) if not k.startswith('_')])
    print("DEBUG - State has filtered_table_docs:", hasattr(state, 'filtered_table_docs'))
    
    # Get essential information
    user_query = state["messages"][-1].content
    tables = state.get("tables", [])
    filtered_docs = state.get("filtered_table_docs", {})
    
    # Debug print for filtered_table_docs
    print("DEBUG - Selected tables:", tables)
    print("DEBUG - filtered_docs has keys:", list(filtered_docs.keys()))
    for table in tables:
        print(f"DEBUG - Table {table} exists in filtered_docs: {table in filtered_docs}")
        if table in filtered_docs:
            table_doc = filtered_docs.get(table, {})
            print(f"DEBUG - Table {table} has keys: {list(table_doc.keys())}")
            print(f"DEBUG - Table {table} has important_notes: {'important_notes' in table_doc}")
            if 'important_notes' in table_doc:
                print(f"DEBUG - First few notes for {table}: {table_doc['important_notes'][:2]}")
    
    # Determine the sport type for specific SQL instructions
    sport_type = None
    if any(table.startswith("nba_") for table in tables):
        sport_type = "NBA"
    elif any(table.startswith("mlb_") for table in tables):
        sport_type = "MLB"
    elif any(table.startswith("nfl_") for table in tables):
        sport_type = "NFL"
    
    # Build the prompt with detailed documentation and sport-specific instructions
    prompt = f"""
    You are an expert SQL writer for sports analytics.
    
    USER QUESTION: "{user_query}"
    
    SELECTED TABLES: {', '.join(tables)}
    """
    
    # Add comprehensive documentation for each table
    for table in tables:
        table_doc = filtered_docs.get(table, {})
        prompt += f"\n\nTABLE: {table}"
        
        if "description" in table_doc:
            prompt += f"\nDESCRIPTION: {table_doc.get('description', '')}"
        
        if "important_notes" in table_doc:
            prompt += "\nIMPORTANT NOTES:"
            for note in table_doc.get("important_notes", []):
                prompt += f"\n- {note}"
        
        if "columns" in table_doc:
            prompt += "\nCOLUMNS:"
            for col in table_doc.get("columns", []):
                prompt += f"\n- {col.get('name')}: {col.get('description', '')}"
        
        # Add metrics documentation
        metrics_doc = table_doc.get("metrics_documentation", {})
        if metrics_doc:
            prompt += "\n\nMETRICS GUIDELINES:"
            
            # Add thresholds first as they're most important for calculations
            if "thresholds" in metrics_doc:
                prompt += "\nTHRESHOLDS:"
                for threshold in metrics_doc["thresholds"]:
                    position_prefix = f"{threshold.get('position', '')}: " if 'position' in threshold else ""
                    prompt += f"\n- {position_prefix}{threshold.get('metric', '')} >= {threshold.get('min_value', '')} ({threshold.get('description', '')})"
            
            # Add query requirements
            if "query_requirements" in metrics_doc:
                prompt += "\n\nQUERY REQUIREMENTS:"
                for req in metrics_doc["query_requirements"]:
                    prompt += f"\n- {req}"
            
            # Add formulas for calculated metrics
            calculated_metrics = []
            for category, metrics in metrics_doc.items():
                if category not in ["thresholds", "query_requirements"]:
                    for metric in metrics:
                        if "formula" in metric:
                            calculated_metrics.append(f"- {metric.get('name', '')}: {metric.get('formula')} ({metric.get('description', '')})")
            
            if calculated_metrics:
                prompt += "\n\nCALCULATED METRICS:"
                for metric in calculated_metrics:
                    prompt += f"\n{metric}"
    
    # Add baseball stats glossary if available
    for table in tables:
        table_doc = filtered_docs.get(table, {})
        if "baseball_stats_glossary" in table_doc:
            prompt += "\nBASEBALL STATS GLOSSARY:\n"
            glossary = table_doc.get("baseball_stats_glossary", {})
            for stat, description in list(glossary.items())[:15]:  # Limit to 15 most relevant
                prompt += f"- {stat}: {description}\n"
    
    # Add relationship information if available
    for table in tables:
        table_doc = filtered_docs.get(table, {})
        if "join_relationships" in table_doc:
            prompt += "\nJOIN RELATIONSHIPS:\n"
            for relationship in table_doc.get("join_relationships", []):
                prompt += f"- {relationship}\n"
            break  # Only need to add once
    
    # Add sport-specific SQL generation instructions from app_prev.py
    if sport_type == "NBA":
        prompt += """
        
        NBA-SPECIFIC SQL INSTRUCTIONS:
        1. Always use 'pts' for points, NOT 'points'
        2. For player stats, filter by minimum minutes if analyzing performance (e.g., min >= 10)
        3. Use max(season) to filter for current season unless question asks about career stats
        4. When question has season in format 2022-23 or 2021-22, consider only the first number
        5. Use opponent_name when asked about team allowed stats (and sum up the stats)
        6. Do not group by both team_name and opponent_name in the same query
        7. When filtering position, use LIKE operator, not equals
        8. For calculating shooting percentages:
           - Field goal %: fgm / fga
           - Three-point %: three_pointer_made / three_pointer_attempt
           - Free throw %: ftm / fta
        9. For efficiency metrics:
           - True shooting %: pts / (2 * (fga + 0.44 * fta))
           
        TEAM RECORD CALCULATION:
        - For team records/standings: Group by team_name and calculate:
          - COUNT(CASE WHEN team_points > opponent_points THEN 1 END) as wins
          - COUNT(CASE WHEN team_points < opponent_points THEN 1 END) as losses
          - (wins * 1.0) / NULLIF((wins + losses), 0) as win_percentage
        - Include minimum game threshold (e.g., MIN 20 games played)
        - Sort by win_percentage DESC for "best record" questions
        - For NBA specifically, use max(season) for current season records
        """
    elif sport_type == "MLB":
        prompt += """
        
        MLB-SPECIFIC SQL INSTRUCTIONS:
        1. For batting stats, apply minimum threshold of ab >= 3 for meaningful analysis
        2. For pitching stats, apply minimum threshold of ip >= 1
        3. Consider only starting pitchers (starter=1) unless question specifically asks about relievers
        4. Filter for current year data unless the question explicitly uses the word "career"
        5. For certain stats like ERA, OPS, Batting Average, sum the cumulative components first, then calculate
        6. For plate_appearance_dataframe:
           - Use p_throws column to get pitcher handedness
           - Use batter_team_name for "teams against pitchers"
           - Use pitcher_team_name for "teams against batters" 
           - Use batter_name for "players/batters against pitchers"
           - Use pitcher_name for "players/pitchers against batters"
           - When asked about hits, home runs, use SUM and not COUNT
        7. Rate stats formulas:
           - Batting average (avg): h / ab
           - On-base percentage (obp): (h + bb + hbp) / (ab + bb + hbp + sf)
           - Slugging percentage (slg): total_bases / ab, where total_bases = singles*1 + doubles*2 + triples*3 + hr*4
           - OPS: obp + slg
           - ERA: 9 * (er / ip)
           - WHIP: (h_allowed + bb) / ip
           
        TEAM RECORD CALCULATION:
        - For team records/standings in MLB:
          - For pitching records: Use mlb_pitching_dataframe and SUM(win) as wins, SUM(loss) as losses
          - Calculate win_percentage as (wins * 1.0) / NULLIF((wins + losses), 0)
          - Group by team_name
          - Apply appropriate minimum game threshold
          - Sort by win_percentage DESC for "best record" questions
        """
    elif sport_type == "NFL":
        prompt += """
        
        NFL-SPECIFIC SQL INSTRUCTIONS:
        1. Position-specific thresholds:
           - QB: passes >= 10
           - RB: carries >= 5
           - WR/TE: targets >= 3
        2. Filter for current year data unless the question explicitly uses the word "career"
        3. Use opponent_team_name for allowed stats or defense stats
        4. For home/away distinction, use the home column (1=home, 0=away)
        5. For nfl_play_by_play:
           - For quarter info, use quarter column
           - For time remaining, use game_seconds_remaining
        6. Efficiency metrics formulas:
           - Yards per carry: rushing_yards / carries
           - Yards per reception: receiving_yards / receptions
           - Completion percentage: completions / passes
           
        TEAM RECORD CALCULATION:
        - For team records/standings in NFL:
          - Group by team_name and calculate:
          - COUNT(CASE WHEN team_score > opponent_score THEN 1 END) as wins
          - COUNT(CASE WHEN team_score < opponent_score THEN 1 END) as losses
          - COUNT(CASE WHEN team_score = opponent_score THEN 1 END) as ties (NFL can have ties)
          - Calculate win_percentage as ((wins * 1.0) + (ties * 0.5)) / NULLIF((wins + losses + ties), 0)
          - Apply minimum game threshold
          - Sort by win_percentage DESC for "best record" questions
        """
    
    # Add general SQL requirements
    prompt += """
    
    CRITICAL SQL REQUIREMENTS:
    
    1. USE EXACT TABLE AND COLUMN NAMES - do not make up columns that don't exist
    2. Do not fabricate table joins that are not specified in the documentation
    3. For division operations, cast operands to FLOAT to ensure accurate results
    4. Include appropriate GROUP BY clause for all non-aggregated columns in SELECT
    5. Add checks to prevent division by zero (NULLIF or similar)
    6. Return top 10 results for "best," "worst," "highest," or "lowest" questions
    7. When calculating averages, percentages, or rates, use the exact formulas specified in the metrics documentation
    8. Add appropriate WHERE filters based on the question
    9. Add appropriate ORDER BY clause based on the question
    10. For win percentage calculations, always use multiplication by 1.0 to ensure FLOAT conversion
    
    RETURN ONLY the SQL query, NOTHING else - no explanations, comments or anything other than the SQL query itself.
    """
    
    # Invoke LLM to generate SQL
    print("Print the prompt for generating SQL:" , prompt)
    llm = get_llm()
    response = llm.invoke(prompt)
    sql_text = response.content.strip()
    
    # Clean up the SQL (remove backticks or code blocks if present)
    sql_text = sql_text.replace('`', '')
    sql_text = re.sub(r'```sql\s*', '', sql_text)
    sql_text = re.sub(r'```.*?```', '', sql_text, flags=re.DOTALL)
    sql_text = re.sub(r'```\s*', '', sql_text)
    # Remove the 'sql' keyword if it appears at the beginning of the query
    sql_text = re.sub(r'^sql\s+', '', sql_text, flags=re.IGNORECASE)
    
    # Return the SQL along with an incremented recursion count
    print(f"Generated SQL: {sql_text}")
    return {"sql": sql_text, "recursion": state.get("recursion", 0) + 1}



from datetime import datetime

def sql_verifier(state: State):
    progress_queue.put("Step 6/8: Verifying the SQL...")
    """
    Validates the SQL query using table documentation and sport-specific rules.
    Performs detailed checks for proper aggregation and filtering.
    Uses comprehensive metrics documentation to verify calculations and thresholds.
    """
    # Get essential information
    user_query = state["messages"][-1].content
    sql_query = state.get("sql", "")
    tables = state.get("tables", [])
    filtered_docs = state.get("filtered_table_docs", {})
    
    # Check if this is a team record/standings question
    is_record_question = any(term in user_query.lower() for term in 
                           ["record", "standing", "best team", "winning team", "worst team", 
                            "win percentage", "winning record", "win-loss"])
    
    # Determine the sport type for specific verification rules
    sport_type = None
    if any(table.startswith("nba_") for table in tables):
        sport_type = "NBA"
    elif any(table.startswith("mlb_") for table in tables):
        sport_type = "MLB"
    elif any(table.startswith("nfl_") for table in tables):
        sport_type = "NFL"
    
    # Extract column names from filtered_docs for verification
    all_columns = {}
    for table, doc in filtered_docs.items():
        columns = [col.get("name") for col in doc.get("columns", [])]
        all_columns[table] = columns
    
    # Construct verification prompt
    prompt = f"""
    You are an expert on {sport_type if sport_type else "sports"} data.
    
    VERIFICATION TASK: Check if this SQL query correctly answers: "{user_query}"
    
    SQL Query to verify:
    ```sql
    {sql_query}
    ```
    
    Available tables and their columns:
    """
    
    # Add table and column information
    for table, columns in all_columns.items():
        prompt += f"\n{table}: {', '.join(columns)}"
        
        # Add important notes if available
        table_doc = filtered_docs.get(table, {})
        important_notes = table_doc.get("important_notes", [])
        if important_notes:
            prompt += "\nIMPORTANT NOTES:"
            for note in important_notes:
                prompt += f"\n- {note}"
                
        # Add metrics documentation for verification
        metrics_doc = table_doc.get("metrics_documentation", {})
        if metrics_doc:
            prompt += "\n\nMETRICS GUIDELINES:"
            
            # Add thresholds first as they're most important for verification
            if "thresholds" in metrics_doc:
                prompt += "\nTHRESHOLDS:"
                for threshold in metrics_doc["thresholds"]:
                    position_prefix = f"{threshold.get('position', '')}: " if 'position' in threshold else ""
                    prompt += f"\n- {position_prefix}{threshold.get('metric', '')} >= {threshold.get('min_value', '')} ({threshold.get('description', '')})"
            
            # Add query requirements
            if "query_requirements" in metrics_doc:
                prompt += "\n\nQUERY REQUIREMENTS:"
                for req in metrics_doc["query_requirements"]:
                    prompt += f"\n- {req}"
            
            # Add formulas for calculated metrics
            calculated_metrics = []
            for category, metrics in metrics_doc.items():
                if category not in ["thresholds", "query_requirements"]:
                    for metric in metrics:
                        if "formula" in metric:
                            calculated_metrics.append(f"- {metric.get('name', '')}: {metric.get('formula')} ({metric.get('description', '')})")
            
            if calculated_metrics:
                prompt += "\n\nCALCULATED METRICS:"
                for metric in calculated_metrics:
                    prompt += f"\n{metric}"
    
    # Add special verification checklist for team record/standings questions if relevant
    if is_record_question:
        prompt += """
        
        TEAM RECORD VERIFICATION CHECKLIST:
        1. Check that the query properly calculates win-loss record by:
           - For NBA/NFL: Using appropriate CASE statements to count wins and losses
           - For MLB: Using the win/loss columns directly if available
        2. Verify the win percentage calculation uses: (wins * 1.0) / NULLIF((wins + losses), 0)
        3. Check for a minimum game threshold to exclude teams with too few games
        4. Verify proper grouping by team_name
        5. Confirm sorting is by win_percentage DESC (for "best" record) or ASC (for "worst" record)
        6. For current season: Check that the latest season filter is applied
        7. For specific leagues:
           - NBA: Confirm team_name and opponent_name are used correctly 
           - MLB: Check if the right table is used (mlb_pitching_dataframe has direct win/loss columns)
           - NFL: Verify ties are handled correctly in win percentage calculation (0.5 value for ties)
        """
    
    # Add sport-specific verification instructions from app_prev.py
    if sport_type == "NBA":
        prompt += """
        
        NBA-SPECIFIC VERIFICATION CHECKLIST:
        1. 'pts' column is used for points (NOT 'points')
        2. Proper minimum thresholds are applied (min >= 10 for per-game stats)
        3. max(season) is used for current season filtering unless career stats are requested
        4. When season format is 2022-23 or 2021-22, only the first number is used
        5. LIKE operator is used for position filtering, not equals
        6. opponent_name is used for team allowed stats
        7. team_name and opponent_name are not both in GROUP BY
        8. Correct calculation of percentages:
           - Field goal %: fgm / fga (not fga / fgm)
           - Three-point %: three_pointer_made / three_pointer_attempt
           - Free throw %: ftm / fta
           - True shooting %: pts / (2 * (fga + 0.44 * fta))
        9. Top 10 results are returned for "best," "worst," "highest," or "lowest" questions
        """
    elif sport_type == "MLB":
        prompt += """
        
        MLB-SPECIFIC VERIFICATION CHECKLIST:
        1. Correct use of baseball statistics abbreviations (hr, rbi, avg, era, etc.)
        2. Applied minimum thresholds:
           - For batting: ab >= 3 for meaningful analysis
           - For pitching: ip >= 1 for meaningful analysis
        3. Only starter pitchers (starter=1) are considered unless relievers are specifically requested
        4. Current year filter is applied unless "career" is in the question
        5. Components of rate stats are summed before division:
           - For batting average: sum(h)/sum(ab), not avg(avg)
           - For ERA: 9*sum(er)/sum(ip), not avg(era)
        6. Correctly calculated rate stats using exact formulas:
           - Batting average (avg): h / ab
           - On-base percentage (obp): (h + bb + hbp) / (ab + bb + hbp + sf)
           - Slugging percentage (slg): total_bases / ab
           - OPS: obp + slg
           - ERA: 9 * (er / ip)
           - WHIP: (h_allowed + bb) / ip
        7. For plate_appearance_dataframe:
           - p_throws column is used for pitcher handedness
           - batter_team_name for "teams against pitchers"
           - pitcher_team_name for "teams against batters"
        8. SUM is used for counting hits, home runs, etc. (not COUNT)
        9. Top 10 results are returned for "best," "worst," "highest," or "lowest" questions
        """
    elif sport_type == "NFL":
        prompt += """
        
        NFL-SPECIFIC VERIFICATION CHECKLIST:
        1. Position-specific thresholds are applied:
           - QB: passes >= 10
           - RB: carries >= 5
           - WR/TE: targets >= 3
        2. Current year filter is applied unless "career" is in the question
        3. opponent_team_name is used for allowed stats or defense stats
        4. home column (1=home, 0=away) is used correctly for game location
        5. For play-by-play data:
           - quarter column is used for quarter information
           - game_seconds_remaining is used for time remaining
        6. Correct calculation of efficiency metrics:
           - Yards per carry: rushing_yards / carries
           - Yards per reception: receiving_yards / receptions
           - Completion percentage: completions / passes
        7. Top 10 results are returned for "best," "worst," "highest," or "lowest" questions
        """
    
    # Add general verification criteria from app_prev.py
    prompt += """
    
    CRITICAL VERIFICATION CRITERIA:
    
    1. TABLE AND COLUMN NAMES:
       - All table and column names exist and are used correctly
       - No syntax errors or invalid SQL constructs
       - No references to non-existent columns
    
    2. AGGREGATION VERIFICATION:
       - GROUP BY clause included whenever using aggregation functions
       - All non-aggregated columns in SELECT are included in GROUP BY clause
       - GROUP BY in outer query matches non-aggregated columns from subquery
       - Proper functions used (AVG for averages, SUM for totals, etc.)
       - CAST as FLOAT used on both operands in division operations
       - HAVING clauses used appropriately for filtering aggregated results
    
    3. DIVISION SAFETY:
       - Checks included to prevent division by zero
       - Numerators can be zero, but denominators must be checked
    
    4. TOP RESULTS:
       - For questions asking "best," "worst," "highest," or "lowest," exactly 10 results are returned
    
    5. MINIMUM THRESHOLDS:
       - Appropriate minimum thresholds applied for meaningful analysis
       - Position-specific thresholds applied where relevant
    
    6. NULL HANDLING:
       - "NOT NULL" should never be used (no NaNs in tables)
    
    7. ANSWER VALIDATION:
       - The SQL query correctly answers all aspects of the user's question
       - Results will be in a useful format
       - Appropriate sorting applied (ORDER BY) for ranking queries
       - LIMIT 10 used for top/bottom results
    
    RESPONSE FORMAT (provide ONE of the following responses only):
    - If the SQL query is completely correct: "Yes"
    - If there are aggregation issues: "Generate Query Again: [precise explanation of aggregation problem]"
    - If there are filtering issues: "Generate Query Again: [precise explanation of filtering problem]"
    - If there are column/table name issues: "Generate Query Again: [precise explanation of name problem]"
    - If there are join issues: "Generate Query Again: [precise explanation of join problem]"
    - If calculations are incorrect: "Generate Query Again: [precise explanation of calculation error]"
    - If minimum thresholds are missing: "Generate Query Again: [precise explanation of missing threshold]"
    - If division by zero is possible: "Generate Query Again: [precise explanation of division by zero risk]"
    - For any other issues: "Generate Query Again: [precise explanation of the issue]"
    """
    
    # Invoke LLM for verification
    llm = get_llm()
    response = llm.invoke(prompt)
    response_text = response.content.strip()
    
    print(f"SQL Verification: {response_text}")
    
    return {"sql_verifier_response": response_text}



def next_node_selector(state:State):
    response = state.get("sql_verifier_response", "")
    print(f"Selecting next node based on: {response}")
    
    # Simple check for recursion limit
    if state.get("recursion", 0) >= 5:
        return "No Answer"
    
    # Check for simple "Yes" response (exact match)
    if response.strip() == "Yes":
        return "Yes"
    
    # Check for "Generate Query Again" prefix
    if response.startswith("Generate Query Again"):
        return "Generate Query Again"
    
    # Default to "No Answer" if response doesn't match expected formats
    return "No Answer"


def execute_query(state: State):
    progress_queue.put("Step 7/8: Executing the SQL query...")
    """
    Executes the SQL query against the database and returns the results.
    """
    sql = state.get("sql", "")
    
    # Print query for debugging
    print("Executing query:", sql)
    
    try:
        # Connect to the database
        connection = connect_to_db()
        with connection.cursor() as cursor:
            cursor.execute(sql)
            result = cursor.fetchall()
            
            # Get column names from cursor description
            columns = [desc[0] for desc in cursor.description] if cursor.description else []
            
            # Convert results to list of dictionaries for easier handling
            data = [dict(zip(columns, row)) for row in result]
        
        print("Query Executed Successfully")
        print(data)
        
        return {"status": "success", "data": data}
    except Exception as e:
        error_message = str(e)
        print(f"Error executing query: {error_message}")
        return {"status": "error", "status_message": error_message}

def execution_check(state: State):
    """
    Checks for SQL execution errors and formats them for the verifier.
    """
    status = state.get("status", "")
    if status == "error":
        error_msg = state.get("status_message", "Unknown error")
        return {"sql_verifier_response": f"Generate Query Again: SQL error - {error_msg}"} 
    return {}
      
def execution_error(state:State):
    if state.get("status")=="error":
        return "Yes"
    else:
        return "No"
      
def exceptional_case(state:State):
    return {"messages":"Please ask the question in a different way, I was not able to get the answer properly. Sorry!!"}
    
def result_summary(state: State):
    progress_queue.put("Step 8/8: Summarizing Result...")
    """
    Summarizes the result based on the given question and formats the output in a well-structured manner.
    If the data is tabular, it will be presented as a Markdown table. Never skip any result. Understand
    the result properly and answer according to the questions.
    """
    data = state.get("data", "")
    history = state["messages"]
    question = state["messages"][-1].content

    prompt = f"""
              You are a professional data analyst. Your task is to generate a structured and professional summary of the provided data based on the user's question.

              ### Guidelines:
              1. **Formatting**:
                - If the data is tabular, format it as a well-structured Markdown table.
                - If needed, provide a brief textual summary before the table. But always give the data too. Never get rid of the data provided to you.
              2. **Interpretation**:
                - If the data is purely numerical, interpret its meaning relative to the question.
                - Use historical context only if necessary, but do not explicitly mention it.
              3. **Clarity & Professionalism**:
                - Keep the response clear, concise, and well-organized.
                - Ignore null values; if no data is available, respond professionally stating that data is not available.
              4. **Multi-Turn Context**:
                - Consider previous conversation history to ensure completeness, if relevant.

              ### Provided Information:
              - **User Question:** {question}
              - **Data:**
              ```{data}```
              - **Previous Conversation History (if needed):**
              ```{history}```

              Generate a professional, structured summary of the results and also mention the mtric or KPI used to yieltd the result.
              """


    summary = llm.invoke(prompt)
    print("printing the summary")
    print(summary)
    progress_queue.put("[DONE]")
    
    return {"messages": summary}


# Add nodes to the graph
graph.add_node("general_answer", general_answer)
graph.add_node("table_doc_node", get_table_doc)
graph.add_node("select_tables_node", select_tables)
graph.add_node("table_doc_filtered", get_table_documentation)
graph.add_node("sql_generation", generate_sql)
graph.add_node("sql_verifier", sql_verifier)
graph.add_node("query_execution", execute_query)
graph.add_node("execution_check", execution_check)
graph.add_node("result_summary", result_summary)
graph.add_node("exceptional_case", exceptional_case)

# Define edges between nodes
graph.add_edge(START, "table_doc_node")
graph.add_conditional_edges(
                        "table_doc_node",
                        is_question_need_database,
                        {
        "General": "general_answer",
        "Database": "select_tables_node"  # Changed to go through table selection first
    }
)
graph.add_edge("general_answer", END)
graph.add_edge("select_tables_node", "table_doc_filtered")  # First select tables, then get their docs
graph.add_edge("table_doc_filtered", "sql_generation")  # Then generate SQL with filtered docs
graph.add_edge("sql_generation", "sql_verifier")
graph.add_conditional_edges(
                            "sql_verifier",
                            next_node_selector,
                            {
        "Yes": "query_execution",
        "Generate Query Again": "sql_generation",
        "No Answer": "exceptional_case"
    }
)
graph.add_edge("exceptional_case", END)
graph.add_edge("query_execution", "execution_check")
graph.add_conditional_edges(
                            "execution_check",
                            execution_error,
                            {
        "Yes": "sql_generation",
                                "No": "result_summary"
                            }
                            )
graph.add_edge("result_summary", END)

memory = MemorySaver()
graph_run = graph.compile(checkpointer=memory)

@traceable(run_type="chain", name="CustomGPT Analytics")
def bot_answer(user_message:str, thread_id = 1):
    config = {"configurable": {"thread_id": thread_id}}

    # Start conversation
    output = graph_run.invoke({"messages": [HumanMessage(content=user_message)]}, config) 
    print("Printing the Output")
    print(output)
    for m in output['messages'][-1:]:
        m.pretty_print()
    return output
      
  
app = Flask(__name__)
# Global thread-safe progress queue (used by nodes via progress_queue.put)
progress_queue = queue.Queue()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    """
    Handle chatbot requests for SQL query generation and execution with streaming response.
    """
    data = request.get_json()
    print("Received data:", data)
    def generate(data):
        user_message = data.get("message", "")
        print("User Message:", user_message)
        
        # Create a queue to store the response from the thread
        response_queue = queue.Queue()
        
        # Modified thread to store the response
        def process_response():
            response = bot_answer(user_message)
            response_queue.put(response)
        
        response_thread = threading.Thread(target=process_response)
        response_thread.start()
        
        while response_thread.is_alive():
            try:
                progress = progress_queue.get(timeout=1)
                if progress == "[DONE]":
                    continue
                yield json.dumps({'type': 'progress', 'data': progress}) + '\n'
            except queue.Empty:
                continue
        
        # Get the response from the queue instead of calling bot_answer again
        response = response_queue.get()
        result = response["messages"][-1].content
        yield json.dumps({'type': 'answer', 'data': result}) + '\n'

    return Response(stream_with_context(generate(data)), 
                   content_type="application/x-ndjson")  # Changed content type

# Background thread to monitor progress_queue
def progress_monitor():
    while True:
        try:
            # Wait up to 30 seconds for a new message
            message = progress_queue.get(timeout=25)
            progress_queue.put(message)
        except queue.Empty:
            # Send a heartbeat if no messages appear within 30 seconds
            pass

# Start the progress monitor thread as a daemon so it exits with the app
threading.Thread(target=progress_monitor, daemon=True).start()

if __name__ == "__main__":
  app.run(debug=False)