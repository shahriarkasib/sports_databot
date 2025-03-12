import sqlalchemy
from sqlalchemy import inspect
import pandas as pd

database_url = 'database url'

# Create the engine using the database URL
engine = sqlalchemy.create_engine(database_url)

def get_all_tables_and_columns():
    """
    Get all table names and their columns from the PostgreSQL database.
    Returns a dictionary with table names as keys and lists of column names as values.
    """
    inspector = inspect(engine)
    tables_and_columns = {}
    
    # Get all table names
    table_names = inspector.get_table_names()
    
    # For each table, get the column information
    for table_name in table_names:
        columns = inspector.get_columns(table_name)
        column_info = [{"name": column['name'], "type": str(column['type'])} for column in columns]
        tables_and_columns[table_name] = column_info
    
    return tables_and_columns

def print_tables_and_columns():
    """
    Print all tables and their columns in a readable format.
    """
    tables_and_columns = get_all_tables_and_columns()
    
    if not tables_and_columns:
        print("No tables found in the database.")
        return
    
    print("DATABASE SCHEMA:")
    print("===============")
    
    for table_name, columns in tables_and_columns.items():
        print(f"\nTable: {table_name}")
        print("-" * (len(table_name) + 7))
        
        for idx, column in enumerate(columns, 1):
            print(f"{idx}. {column['name']} ({column['type']})")

def get_schema_as_dataframe():
    """
    Convert the tables and columns information to a pandas DataFrame.
    Returns a DataFrame with columns: 'table_name', 'column_name', 'data_type'
    """
    tables_and_columns = get_all_tables_and_columns()
    
    # Create a list to store all rows
    rows = []
    
    # Populate the rows list with table and column information
    for table_name, columns in tables_and_columns.items():
        for column in columns:
            rows.append({
                'table_name': table_name,
                'column_name': column['name'],
                'data_type': column['type']
            })
    
    # Create a DataFrame from the rows
    df = pd.DataFrame(rows)
    
    return df

# Example usage - uncomment to run
# print_tables_and_columns()

# To get the data as a dictionary for further processing
tables_data = get_all_tables_and_columns()
print(tables_data)

# Convert to DataFrame and print
schema_df = get_schema_as_dataframe()
print("\nDatabase Schema as DataFrame:")
print(schema_df)
schema_df.to_csv('database_schema.csv', index=False)
