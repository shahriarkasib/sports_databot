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
from question_categorization import categorize_question, get_kpi_documentation_for_categories


load_dotenv()

# Database credentials are now imported from postgre_connection.py

# Ignore all warnings of a specific type
warnings.filterwarnings("ignore", category=DeprecationWarning)

@traceable(run_type="llm", name="DeepSeek R1 8B")
def get_llm():
    return ChatOpenAI(model="gpt-4o", temperature= 0.0)

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
    kpi_documentation: dict
    kpi_categories: list  # New field to store the identified KPI categories
    relevant_kpi_docs: dict  # New field to store the relevant KPI documentation
    
# Initialize the StateGraph
graph = StateGraph(State)

@traceable(run_type="tool", name="Load Table Documentation")
def get_table_doc(state: State):
    progress_queue.put("Step 1/8: Loading table documentation...")
    try:
        with open("table_documentation/merged_table_documentation.json", "r", encoding="utf-8") as f:
            merged_doc = json.load(f)
        with open("kpi_documentation.json", "r", encoding="utf-8") as f:
            kpi_doc = json.load(f)
    except Exception as e:
        merged_doc = {"error": f"Failed to load merged documentation: {e}"}
    # Return the loaded documentation in the state along with any necessary fields.
    return {"table_documentation": merged_doc, "kpi_documentation": kpi_doc, "recursion": 0, "sql_verifier_response": None}

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
    kpi_documentation = state.get("kpi_documentation", {})
    
    # Add previous questions to the prompt
    question_context = ""
    if previous_questions:
        question_context += "\n"
        for question in previous_questions:
            question_context += f"- {question}\n"
    
    prompt = f"""
    I need to select the most relevant database tables for this question: "{user_query}"
    
    PREVIOUS CONVERSATION CONTEXT (most recent first):
    TABLE SELECTION RULES:
    1. Select ONLY tables that are relevant to answering the question above
    2. If the question relates to a specific sport based on the current question or previous context, ONLY select tables for that sport
    3. DO NOT mix tables from different sports (NBA, MLB, NFL) in the same query
    4. If the question is vague but previous questions were about a specific sport, maintain that context
    
    use the following available tables, table documentation and kpi documentation to select the most relevant tables:
    available tables:
    {available_tables}
    
    Table Documentation:
    {full_table_docs}
    
    KPI Documentation:
    {kpi_documentation}
    
    Previous Conversation Context:
    {question_context}
    
    Response Format:
    Return ONLY the table names separated by commas without any other text, commentary, or explanation.
    For example: "nba_main_dataframe,nba_play_by_play" or "mlb_batting_dataframe" or "nfl_main_df" 
   
    """
    
    # Call the LLM to get the tables
    llm = get_llm()
    response = llm.invoke(prompt)
    response_text = response.content.strip()
    
    # Parse the response to get the table names
    # Remove any quotes, backticks, or code blocks that might be in the response
    response_text = response_text.replace('`', '').replace('"', '').replace("'", "")
    response_text = re.sub(r'```.*?```', '', response_text, flags=re.DOTALL)
    response_text = re.sub(r'```.*', '', response_text, flags=re.DOTALL)
    
    # Split by commas and strip whitespace
    tables = [table.strip() for table in response_text.split(',')]
    print("selectedtables")
    print(tables)
    
    return {"tables":  tables, "recursion": 0}

@traceable(run_type="tool", name="Get Table Documentation")
def get_filtered_table_documentation(state: State):
    progress_queue.put("Step 4/8: Fetching table documentation...")
    """
    Return the selected tables documentation from the original table documentation.
    """
    tables = state.get("tables", [])
    table_docs = state.get("table_documentation", {})
    
    # Create a filtered dictionary containing documentation for only the selected tables
    filtered_table_docs = {}
    
    # Filter table docs to only include selected tables
    for table in tables:
        if table in table_docs:
            filtered_table_docs[table] = table_docs[table]
            print(f"Using documentation for table: {table}")
    
    # Return the filtered documentation in the state
    return {"filtered_table_docs": filtered_table_docs}


@traceable(run_type="tool", name="Categorize Question by KPI")
def categorize_question_kpi(state: State,llm=get_llm()):
    """
    Analyzes the user question to identify relevant KPI categories
    and retrieves the appropriate documentation for those categories.
    Updates the state with the categorized KPIs and related documentation.
    """
    progress_queue.put("Step 3.5/8: Categorizing question for relevant KPIs...")
    
    # Enhanced progress message
    progress_message = {
        "step": "KPI Categorization",
        "details": "Analyzing question to identify relevant KPI categories",
        "data": "Matching your question to our KPI framework..."
    }
    progress_queue.put(json.dumps(progress_message))
    
    # Extract the user's question
    user_query = state["messages"][-1].content
    print(f"Categorizing question for KPIs: {user_query}")
    
    try:
        # Get the current language model
        current_model = 'gpt-4o-mini'
        # Add more detailed progress update
        categorization_progress = {
            "step": "KPI Analysis",
            "details": "Identifying relevant KPI categories",
            "data": f"Analyzing your question using {current_model}..."
        }
        progress_queue.put(json.dumps(categorization_progress))
        
        # Load all KPI categories for reference
        try:
            print("Loading KPI documentation...")
            with open("kpi_documentation.json", "r") as f:
                all_kpi_data = json.load(f)
                category_list = all_kpi_data.get("category_list", [])
        except Exception as e:
            print(f"Error loading category list: {e}")
            category_list = []
        
        # Categorize the question
        categories = categorize_question(user_query, llm=llm)
        print(f"Identified KPI categories: {categories}")
        
        # If we have categories, get the relevant documentation
        if categories:
            # Get the documentation for these categories
            kpi_docs = get_kpi_documentation_for_categories(categories)
            print(f"Retrieved KPI documentation: {kpi_docs}")
            # Create a comprehensive display of the categories with their examples
            category_details = []
            
            # First, include detailed info for the selected categories
            for cat_name in categories:
                # Find the category in the category_list for examples
                cat_info = next((cat for cat in category_list if cat.get("category") == cat_name), None)
                
                if cat_info:
                    examples = ", ".join(cat_info.get("examples", []))
                    category_details.append(f"{cat_name}: {cat_info.get('description', '')} (Examples: {examples})")
                else:
                    # If not in category_list, build from kpi_docs
                    for category in kpi_docs.get("kpi_categories", []):
                        if category.get("category") == cat_name:
                            kpi_names = [kpi.get("name") for kpi in category.get("kpis", [])]
                            examples = ", ".join(kpi_names[:5])
                            category_details.append(f"{cat_name}: {category.get('description', '')} (Examples: {examples})")
            
            # Also include a brief list of metrics for these categories
            all_kpi_names = []
            if kpi_docs and "kpi_categories" in kpi_docs:
                for category in kpi_docs["kpi_categories"]:
                    for kpi in category.get("kpis", []):
                        kpi_name = kpi.get("name", "")
                        all_kpi_names.append(kpi_name)
            
            # Build a comprehensive message with all details
            category_desc_text = ""
            if category_details:
                category_desc_text = "\n\nRelevant Categories:\n" + "\n".join(f"- {desc}" for desc in category_details)
                
                if all_kpi_names:
                    kpi_list_text = ", ".join(all_kpi_names)
                    category_desc_text += f"\n\nAll Available KPIs for these categories:\n{kpi_list_text}"
            
            # Include a list of other available categories for reference
            if len(category_list) > len(categories):
                other_categories = [cat.get("category") for cat in category_list if cat.get("category") not in categories]
                if other_categories:
                    category_desc_text += "\n\nOther Available Categories:\n" + ", ".join(other_categories)
            
            # Update the state with the categories and documentation
            progress_message = {
                "step": "KPI Documentation",
                "details": f"Retrieved documentation for {len(categories)} relevant KPI categories",
                "data": f"Found relevant KPI categories: {', '.join(categories)}{category_desc_text}"
            }
            progress_queue.put(json.dumps(progress_message))
            
            # Also print the categories with descriptions for logging
            print(f"KPI Documentation: {kpi_docs}")
            print(f"Categories: {categories}")
            
            return {
                "kpi_categories": categories,
                "relevant_kpi_docs": kpi_docs
            }
        else:
            # No categories were identified
            print("No KPI categories identified for this question")
            
            # If we have the category_list, include it in the message for reference
            categories_info = ""
            if category_list:
                categories_info = "\n\nAvailable categories:\n" + "\n".join([f"- {cat.get('category')}: {cat.get('description')}" for cat in category_list])
            
            progress_message = {
                "step": "KPI Documentation",
                "details": "No relevant KPIs found",
                "data": f"This question doesn't match our defined KPI categories.{categories_info}"
            }
            progress_queue.put(json.dumps(progress_message))
            
            return {
                "kpi_categories": [],
                "relevant_kpi_docs": {}
            }
    except Exception as e:
        # Log the error and continue the workflow
        print(f"Error in KPI categorization: {str(e)}")
        
        error_message = {
            "step": "KPI Categorization Error",
            "details": "Error during KPI analysis",
            "data": f"Error: {str(e)}. Continuing without KPI categorization."
        }
        progress_queue.put(json.dumps(error_message))
        
        # Return empty values to not block the workflow
        return {
            "kpi_categories": [],
            "relevant_kpi_docs": {}
        }


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
    
    # Get essential information
    user_query = state["messages"][-1].content
    tables = state.get("tables", [])
    # Use filtered_table_docs instead of table_documentation
    table_docs = state.get("filtered_table_docs", {})
    filtered_docs = table_docs.copy()
    sport_type = None
    kpi_documentation = state.get("relevant_kpi_docs", {})
    
    if any(table.startswith("nba_") for table in tables):
        sport_type = "NBA"
    elif any(table.startswith("mlb_") or table.startswith("plate_appearance") or table.startswith("pitching") for table in tables):
        sport_type = "MLB"
    elif any(table.startswith("nfl_") or table.startswith("nfl_play_by_play") for table in tables):
        sport_type = "NFL"
    
    # Build the prompt with detailed documentation and sport-specific instructions
    prompt = f"""
    You are an expert SQL writer for sports analytics. Write an SQL query to answer the user's question.
    You will have seelcted tables, table documnetations and sport specific instructions.
    
    USER QUESTION: "{user_query}"
    
    SELECTED TABLES: {', '.join(tables)}
    
    **2024 IS THE CURRENT SEASON OR CURRENT YEAR**
    
    ** THERE IS NO COLUMN CALLED 'SEASON' IN THE TABLES**
    
    Use the below table documentation to figure out which columns to use and what's the meaning
    of each columns.
    
    TABLE DOCUMENTATION:
    {table_docs}
    
    use the below kpi documentation to figure out which kpi to use and what's the meaning of each kpi.
    also use the formula exactly like it's given in the kpi documentation. if the formula is not given, 
    use the formula from your previous knowledge about the particular sport. Make sure to use coalesce(column_name,0)
    for each single metrics calculation if you need to use multiple metrics to determine a single column.
    KPI DOCUMENTATION:
    {kpi_documentation}
    
    "sample_query": "SELECT player_name, COALESCE(COEALESCE(CAST(SUM(h) AS DECIMAL(10,3)),0) / NULLIF(SUM(ab), 0),0) AS batting_average FROM mlb_batting_dataframe WHERE game_date BETWEEN '[start_date]' AND '[end_date]' GROUP BY player_name HAVING SUM(ab) >= 100 ORDER BY batting_average DESC LIMIT 10;"

    """
    
    # # Add sport-specific SQL generation instructions
    # if sport_type == "NBA":
    #     prompt += """
        
    #     NBA-SPECIFIC SQL INSTRUCTIONS:
    #     1. Always use 'pts' for points, NOT 'points'
    #     2. For player stats, filter by minimum minutes if analyzing performance (e.g., min >= 10)
    #     3. Use max(season) to filter for current season unless question asks about career stats
    #     4. When question has season in format 2022-23, or 2021-22, consider only the first number
    #     5. Use opponent_name when asked about team allowed stats (and sum up the stats)
    #     6. Do not group by both team_name and opponent_name in the same query
    #     7. When filtering position, use LIKE operator, not equals
    #     8. For calculating shooting percentages:
    #        - Field goal %: fgm / fga
    #        - Three-point %: three_pointer_made / three_pointer_attempt
    #        - Free throw %: ftm / fta
    #     9. For efficiency metrics:
    #        - True shooting %: pts / (2 * (fga + 0.44 * fta))
           
    #     TEAM RECORD CALCULATION:
    #     - For team records/standings: Group by team_name and calculate:
    #       - COUNT(CASE WHEN team_points > opponent_points THEN 1 END) as wins
    #       - COUNT(CASE WHEN team_points < opponent_points THEN 1 END) as losses
    #       - (wins * 1.0) / NULLIF((wins + losses), 0) as win_percentage
    #     - Include minimum game threshold (e.g., MIN 20 games played)
    #     - Sort by win_percentage DESC for "best record" questions
    #     - For NBA specifically, use max(season) for current season records
    #     """
    # elif sport_type == "MLB":
    #     prompt += """
        
    #     MLB-SPECIFIC SQL INSTRUCTIONS:
    #     1. For batting stats, apply minimum threshold of ab >= 3 for meaningful analysis
    #     2. For pitching stats, apply minimum threshold of ip >= 1
    #     3. Consider only starting pitchers (starter=1) unless question specifically asks about relievers
    #     4. Filter for current year data unless the question explicitly uses the word "career"
    #     5. For certain stats like ERA, OPS, Batting Average, sum the cumulative components first, then calculate
    #     6. For plate_appearance_dataframe:
    #        - Use p_throws column to get pitcher handedness
    #        - Use batter_team_name for "teams against pitchers"
    #        - Use pitcher_team_name for "teams against batters" 
    #        - Use batter_name for "players/batters against pitchers"
    #        - Use pitcher_name for "players/pitchers against batters"
    #        - When asked about hits, home runs, use SUM and not COUNT
    #     7. Rate stats formulas:
    #        - Batting average (avg): h / ab
    #        - On-base percentage (obp): (h + bb + hbp) / (ab + bb + hbp + sf)
    #        - Slugging percentage (slg): total_bases / ab, where total_bases = singles*1 + doubles*2 + triples*3 + hr*4
    #        - OPS: obp + slg
    #        - ERA: 9 * (er / ip)
    #        - WHIP: (h_allowed + bb) / ip
           
    #     TEAM RECORD CALCULATION:
    #     - For team records/standings in MLB:
    #       - For pitching records: Use mlb_pitching_dataframe and SUM(win) as wins, SUM(loss) as losses
    #       - Calculate win_percentage as (wins * 1.0) / NULLIF((wins + losses), 0)
    #       - Group by team_name
    #       - Apply appropriate minimum game threshold
    #       - Sort by win_percentage DESC for "best record" questions
    #     """
    # elif sport_type == "NFL":
    #     prompt += """
        
    #     NFL-SPECIFIC SQL INSTRUCTIONS:
    #     1. Position-specific thresholds:
    #        - QB: passes >= 10
    #        - RB: carries >= 5
    #        - WR/TE: targets >= 3
    #     2. Filter for current year data unless the question explicitly uses the word "career"
    #     3. Use opponent_team_name for allowed stats or defense stats
    #     4. For home/away distinction, use the home column (1=home, 0=away)
    #     5. For nfl_play_by_play:
    #        - For quarter info, use quarter column
    #        - For time remaining, use game_seconds_remaining
    #     6. Efficiency metrics formulas:
    #        - Yards per carry: rushing_yards / carries
    #        - Yards per reception: receiving_yards / receptions
    #        - Completion percentage: completions / passes
           
    #     TEAM RECORD CALCULATION:
    #     - For team records/standings in NFL:
    #       - Group by team_name and calculate:
    #       - COUNT(CASE WHEN team_score > opponent_score THEN 1 END) as wins
    #       - COUNT(CASE WHEN team_score < opponent_score THEN 1 END) as losses
    #       - COUNT(CASE WHEN team_score = opponent_score THEN 1 END) as ties (NFL can have ties)
    #       - Calculate win_percentage as ((wins * 1.0) + (ties * 0.5)) / NULLIF((wins + losses + ties), 0)
    #       - Apply minimum game threshold
    #       - Sort by win_percentage DESC for "best record" questions
    #     """
    
    # Add general SQL requirements
    prompt += """
    
    CRITICAL SQL REQUIREMENTS:
    1. Always use coalesce(column_name,0) for each single metrics calculation if you need to use multiple metrics to determine a single column.
    2. If you need to use multiple metrics to determine a single column, make sure to select all the metrics in the query.
    3. Do not fabricate table joins that are not specified in the documentation
    4. For division operations, cast operands to FLOAT to ensure accurate results
    5. Include appropriate GROUP BY clause for all non-aggregated columns in SELECT
    6. Add checks to prevent division by zero (NULLIF or similar)
    7. Return top 10 results for "best," "worst," "highest," or "lowest" questions
    8. When calculating averages, percentages, or rates, use the exact formulas specified in the metrics documentation
    9. Add appropriate WHERE filters based on the question
    10. Add appropriate ORDER BY clause based on the question
    11. For win percentage calculations, always use multiplication by 1.0 to ensure FLOAT conversion
    
    RETURN ONLY the SQL query, NOTHING else - no explanations, comments or anything other than the SQL query itself.
    """
    
    print("Prompt for SQL generation: ", prompt)
    # Invoke LLM to generate SQL
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
    # Use filtered_table_docs instead of table_documentation
    table_docs = state.get("filtered_table_docs", {})
    kpi_documentation = state.get("relevant_kpi_docs", {})
    
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
    
    # Construct verification prompt
    prompt = f"""
    You are an expert on {sport_type if sport_type else "sports"} data.
    
    VERIFICATION TASK: Check if this SQL query correctly answers: "{user_query}"
    
    SQL Query to verify:
    ```sql
    {sql_query}
    
    KPI Documentation to use for verification:
    {kpi_documentation}
    ```
    
    Available tables and their columns:
    {table_docs}
    """


    
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
    
    # # Add sport-specific verification instructions from app_prev.py
    # if sport_type == "NBA":
    #     prompt += """
        
    #     NBA-SPECIFIC VERIFICATION CHECKLIST:
    #     1. 'pts' column is used for points (NOT 'points')
    #     2. Proper minimum thresholds are applied (min >= 10 for per-game stats)
    #     3. max(season) is used for current season filtering unless career stats are requested
    #     4. When season format is 2022-23 or 2021-22, only the first number is used
    #     5. LIKE operator is used for position filtering, not equals
    #     6. opponent_name is used for team allowed stats
    #     7. team_name and opponent_name are not both in GROUP BY
    #     8. Correct calculation of percentages:
    #        - Field goal %: fgm / fga (not fga / fgm)
    #        - Three-point %: three_pointer_made / three_pointer_attempt
    #        - Free throw %: ftm / fta
    #        - True shooting %: pts / (2 * (fga + 0.44 * fta))
    #     9. Top 10 results are returned for "best," "worst," "highest," or "lowest" questions
    #     """
    # elif sport_type == "MLB":
    #     prompt += """
        
    #     MLB-SPECIFIC VERIFICATION CHECKLIST:
    #     1. Correct use of baseball statistics abbreviations (hr, rbi, avg, era, etc.)
    #     2. Applied minimum thresholds:
    #        - For batting: ab >= 3 for meaningful analysis
    #        - For pitching: ip >= 1 for meaningful analysis
    #     3. Only starter pitchers (starter=1) are considered unless relievers are specifically requested
    #     4. Current year filter is applied unless "career" is in the question
    #     5. Components of rate stats are summed before division:
    #        - For batting average: sum(h)/sum(ab), not avg(avg)
    #        - For ERA: 9*sum(er)/sum(ip), not avg(era)
    #     6. Correctly calculated rate stats using exact formulas:
    #        - Batting average (avg): h / ab
    #        - On-base percentage (obp): (h + bb + hbp) / (ab + bb + hbp + sf)
    #        - Slugging percentage (slg): total_bases / ab
    #        - OPS: obp + slg
    #        - ERA: 9 * (er / ip)
    #        - WHIP: (h_allowed + bb) / ip
    #     7. For plate_appearance_dataframe:
    #        - p_throws column is used for pitcher handedness
    #        - batter_team_name for "teams against pitchers"
    #        - pitcher_team_name for "teams against batters"
    #     8. SUM is used for counting hits, home runs, etc. (not COUNT)
    #     9. Top 10 results are returned for "best," "worst," "highest," or "lowest" questions
    #     """
    # elif sport_type == "NFL":
    #     prompt += """
        
    #     NFL-SPECIFIC VERIFICATION CHECKLIST:
    #     1. Position-specific thresholds are applied:
    #        - QB: passes >= 10
    #        - RB: carries >= 5
    #        - WR/TE: targets >= 3
    #     2. Current year filter is applied unless "career" is in the question
    #     3. opponent_team_name is used for allowed stats or defense stats
    #     4. home column (1=home, 0=away) is used correctly for game location
    #     5. For play-by-play data:
    #        - quarter column is used for quarter information
    #        - game_seconds_remaining is used for time remaining
    #     6. Correct calculation of efficiency metrics:
    #        - Yards per carry: rushing_yards / carries
    #        - Yards per reception: receiving_yards / receptions
    #        - Completion percentage: completions / passes
    #     7. Top 10 results are returned for "best," "worst," "highest," or "lowest" questions
    #     """
    
    # # Add general verification criteria from app_prev.py
    # prompt += """
    
    # CRITICAL VERIFICATION CRITERIA:
    
    # 1. TABLE AND COLUMN NAMES:
    #    - All table and column names exist and are used correctly
    #    - No syntax errors or invalid SQL constructs
    #    - No references to non-existent columns
    
    # 2. AGGREGATION VERIFICATION:
    #    - GROUP BY clause included whenever using aggregation functions
    #    - All non-aggregated columns in SELECT are included in GROUP BY clause
    #    - GROUP BY in outer query matches non-aggregated columns from subquery
    #    - Proper functions used (AVG for averages, SUM for totals, etc.)
    #    - CAST as FLOAT used on both operands in division operations
    #    - HAVING clauses used appropriately for filtering aggregated results
    
    # 3. DIVISION SAFETY:
    #    - Checks included to prevent division by zero
    #    - Numerators can be zero, but denominators must be checked
    
    # 4. TOP RESULTS:
    #    - For questions asking "best," "worst," "highest," or "lowest," exactly 10 results are returned
    
    # 5. MINIMUM THRESHOLDS:
    #    - Appropriate minimum thresholds applied for meaningful analysis
    #    - Position-specific thresholds applied where relevant
    
    # 6. NULL HANDLING:
    #    - "NOT NULL" should never be used (no NaNs in tables)
    
    # 7. ANSWER VALIDATION:
    #    - The SQL query correctly answers all aspects of the user's question
    #    - Results will be in a useful format
    #    - Appropriate sorting applied (ORDER BY) for ranking queries
    #    - LIMIT 10 used for top/bottom results
    
    # RESPONSE FORMAT (provide ONE of the following responses only):
    # - If the SQL query is completely correct: "Yes"
    # - If there are aggregation issues: "Generate Query Again: [precise explanation of aggregation problem]"
    # - If there are filtering issues: "Generate Query Again: [precise explanation of filtering problem]"
    # - If there are column/table name issues: "Generate Query Again: [precise explanation of name problem]"
    # - If there are join issues: "Generate Query Again: [precise explanation of join problem]"
    # - If calculations are incorrect: "Generate Query Again: [precise explanation of calculation error]"
    # - If minimum thresholds are missing: "Generate Query Again: [precise explanation of missing threshold]"
    # - If division by zero is possible: "Generate Query Again: [precise explanation of division by zero risk]"
    # - For any other issues: "Generate Query Again: [precise explanation of the issue]"
    # """
    
    
    prompt+="""
     ### Verification Instructions:
    1. Just verify if there is any syntax error that might throw error in execution.
    2. Check if the table names and column names used are correct.
    3. Check if the query is using previous history correctly only if it is applicable for the user query.

    ### Response Guidelines:
    - If the SQL query is the query seems to be correct in terms of syntax, table names, and column names is correct respond with: Yes, 
        No need to make it more accurate or interprete any thing extra.
    - If there is a **syntax, or table or column name mismatch issue** or **wrong table selection** respond with: Generate Query Again along with an explanation of the issue.
       Always suggest the corrected SQL query if it need Generate Query Again.
    Evaluate the provided SQL query and return the appropriate response according to these rules.
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
    if "Yes" in response.strip():
        return "Yes"
    
    # Check for "Generate Query Again" prefix
    if "Generate Query Again" in response.strip():
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
    print("\n=== SQL Query Execution ===")
    print("Query to execute:", sql)
    print("========================\n")
    
    try:
        # Use the SQLAlchemy engine from postgre_connection.py
        with engine.connect() as connection:
            # Convert SQL string to SQLAlchemy text object
            from sqlalchemy import text
            result = connection.execute(text(sql))
            # Get column names from result keys
            columns = result.keys()
            # Convert results to list of dictionaries
            data = [dict(zip(columns, row)) for row in result]
        
        print("=== Query Execution Success ===")
        print("Number of rows returned:", len(data))
        print("Columns:", columns)
        print("First row sample:", data[0] if data else "No data")
        print("===========================\n")
        
        return {"status": "success", "data": data}
    except Exception as e:
        print("\n=== Database Execution Error ===")
        print("Error Type:", type(e).__name__)
        print("Error Message:", str(e))
        print("SQL Query:", sql)
        print("============================\n")
        
        error_message = str(e)
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
                - Include all the metrics passed to you in the data.
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
    progress_queue.put("[DONE]")
    
    return {"messages": summary}


# Add nodes to the graph
graph.add_node("table_doc_node", get_table_doc)
graph.add_node("select_tables_node", select_tables)
graph.add_node("table_doc_filtered", get_filtered_table_documentation)
graph.add_node("kpi_categorization", categorize_question_kpi)
graph.add_node("sql_generation", generate_sql)
graph.add_node("sql_verifier", sql_verifier)
graph.add_node("query_execution", execute_query)
graph.add_node("execution_check", execution_check)
graph.add_node("result_summary", result_summary)
graph.add_node("exceptional_case", exceptional_case)

# Define edges between nodes
graph.add_edge(START, "table_doc_node")
graph.add_edge("table_doc_node", "select_tables_node")
graph.add_edge("select_tables_node", "table_doc_filtered")  # First select tables, then get their docs
graph.add_edge("table_doc_filtered", "kpi_categorization")
graph.add_edge("kpi_categorization", "sql_generation")  # Then generate SQL with filtered docs
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