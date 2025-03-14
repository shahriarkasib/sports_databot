import os
import json
from langchain_groq import ChatGroq

# Load KPI categories
def load_kpi_categories():
    """Load the category names, descriptions and example KPIs from KPI documentation."""
    try:
        with open("kpi_documentation.json", "r") as f:
            kpi_data = json.load(f)
            
            # Use the new category_list if available
            if "category_list" in kpi_data:
                print("Loading KPI categories from category_list...")
                return kpi_data["category_list"]
            
            # Fallback to the old method if category_list is not available
            categories = []
            for category in kpi_data["kpi_categories"]:
                # Get KPI names in this category for examples
                kpi_names = [kpi["name"] for kpi in category.get("kpis", [])]
                example_kpis = ", ".join(kpi_names[:5])  # Limit to 5 examples
                
                categories.append({
                    "category": category["category"],
                    "description": category["description"],
                    "example_kpis": example_kpis if kpi_names else "None specified"
                })
            return categories
    except Exception as e:
        print(f"Error loading KPI categories: {e}")
        return []

def categorize_question(question, llm=None):
    """
    Categorize a user question according to KPI categories.
    Returns a list of relevant category names.
    """
    categories = load_kpi_categories()
    if not categories:
        # Fallback if categories can't be loaded
        return ["Financial Performance Metrics"]
    
    # Format categories for the prompt with richer descriptions including example KPIs
    categories_text = ""
    for i, cat in enumerate(categories):
        categories_text += f"{i+1}. {cat['category']}:\n"
        categories_text += f"   Description: {cat['description']}\n"
        
        # Use examples field if available, otherwise use example_kpis
        if "examples" in cat:
            examples = ", ".join(cat["examples"])
            categories_text += f"   Example metrics: {examples}\n\n"
        elif "example_kpis" in cat:
            categories_text += f"   Example metrics: {cat['example_kpis']}\n\n"
        else:
            categories_text += "\n"
    
    prompt = f"""
   You are a restaurant data analytics expert. You will be provided with two inputs:

    **categories_text**: A text block containing descriptions of various category list with KPI examples.
    **question**: A question for which you need to identify the most relevant category list.
    
    ###Your Task:
    
    **Analyze the Category List, Description and Example KPIs provided in categories_text.**
    **Determine which category or categories contain the metrics most relevant for answering the question.**
    **Return your answer as a JSON array of strings, where each string is a category name.**
    
    ###Output Requirements:
    
    Output ONLY a JSON array containing the names of the relevant KPI categories.
    Do not include any additional text, explanation, or formatting outside the JSON array.
    
    categories_text: {categories_text}\n
    
    question: "{question}"
    
    """
    
    try:
        llm = ChatGroq(model='llama-3.3-70b-versatile', temperature=0.0)
        response = llm.invoke(prompt).content.strip()
        
        # Extract the JSON array from the response
        if "[" in response and "]" in response:
            json_str = response[response.find("["):response.rfind("]")+1]
            categories = json.loads(json_str)
            return categories
        else:
            # Try parsing the whole response if not in proper format
            return json.loads(response)
    except Exception as e:
        print(f"Error categorizing question: {e}")
        # Return a default category if there's an error
        return ["Financial Performance Metrics"]

def get_kpi_documentation_for_categories(categories):
    """
    Retrieve only the KPI documentation for the specified categories.
    """
    try:
        with open("kpi_documentation.json", "r") as f:
            full_doc = json.load(f)
            
            # Create a subset of the documentation
            subset_doc = {
                "metadata": full_doc["metadata"],
                "kpi_categories": [],
                "sql_generation_guidelines": full_doc["sql_generation_guidelines"]
            }
            
            # Add only the selected categories
            for category in full_doc["kpi_categories"]:
                if category["category"] in categories:
                    subset_doc["kpi_categories"].append(category)
            
            return subset_doc
    except Exception as e:
        print(f"Error retrieving KPI documentation: {e}")
        return None

# # Example usage:
# categories = categorize_question("Which restaurant has the greatest number of return customers?")
# print(categories)
# relevant_docs = get_kpi_documentation_for_categories(categories)
# print(relevant_docs)