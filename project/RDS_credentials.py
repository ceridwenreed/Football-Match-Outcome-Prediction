

from sqlalchemy import create_engine
from secrets import(database_type,
                    dbapi, 
                    endpoint,
                    user,
                    database,
                    password,
                    port)

#RDS database credentials
DATABASE_TYPE = database_type
DBAPI = dbapi
ENDPOINT = endpoint
USER = user
PASSWORD = password
DATABASE = database
PORT = port

# connect to RDS
engine = create_engine(f"{DATABASE_TYPE}+{DBAPI}://{USER}:{PASSWORD}@{ENDPOINT}:{PORT}/{DATABASE}").connect() 