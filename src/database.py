from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL
import pandas as pd

from src.config import DB_CONFIG


def get_engine():
    db_url = URL.create(
        drivername="postgresql+psycopg2",
        username=DB_CONFIG["user"],
        password=DB_CONFIG["password"],
        host=DB_CONFIG["host"],
        port=DB_CONFIG["port"],
        database=DB_CONFIG["database"],
    )

    return create_engine(db_url)


def test_connection():
    try:
        engine = get_engine()

        with engine.connect() as conn:
            version = conn.execute(text("SELECT version();")).scalar()
            print("\n✅ PostgreSQL connection successful!\n")
            print(version)

        return True

    except Exception as e:
        print("\n❌ Connection failed\n")
        print(e)
        return False


def read_sql(query: str) -> pd.DataFrame:
    engine = get_engine()
    return pd.read_sql(query, engine)