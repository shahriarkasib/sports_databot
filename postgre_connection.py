import sqlalchemy
from sqlalchemy import inspect
import pandas as pd

database_url = 'postgresql+psycopg2://u6t22h6qa851ge:pdd922f62af664ee296fdfcfaaf39ba95d3fcecddd6714fc69e029da8bfaa86e5@c4g0h0kljo97tk.cluster-czrs8kj4isg7.us-east-1.rds.amazonaws.com:5432/dequvcgggnb24g'

# Create the engine using the database URL
engine = sqlalchemy.create_engine(database_url)
