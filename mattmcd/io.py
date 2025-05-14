import os
import numpy as np
import pandas as pd
import sqlalchemy
from sqlalchemy import create_engine, Table, func, MetaData, select, case, and_, or_, literal, URL
from sqlalchemy.exc import NoSuchTableError
from pathlib import Path
import json

def pg_engine():
    with open(os.path.expanduser('~/.mattmcd/postgres.json'), 'r') as f:
        creds = json.load(f)['mattmcd']

    engine = create_engine(
        f"{creds['drivername']}://{creds['username']}:{creds['password']}"
        f"@{creds['host']}:{creds['port']}/"
    )
    return engine
