from neo4j import GraphDatabase
import random, time
from .secret_manager import get_neo4j_credentials

# Read from the .env file
secret_dict = get_neo4j_credentials()

local_url = secret_dict["local_neo4j_url"]
local_username = secret_dict["local_neo4j_username"]
local_password = secret_dict["local_neo4j_password"]
local_driver = GraphDatabase.driver(local_url, auth=(local_username, local_password))

class Neo4jClient():
    def __init__(self, driver=local_driver):
        self.driver=driver
    
    def execute_query(self, query, parameters=None, max_retries=3,database="neo4j"):
        attempt = 0
        while attempt < max_retries:
            try:
                with self.driver.session(database=database) as session:
                    result = session.run(query, parameters)
                    return [record for record in result]
            except Exception as e:
                attempt += 1
                if attempt < max_retries:
                    wait_time = 2 ** attempt + random.random()  # Exponential backoff with jitter
                    print(f"Attempt {attempt} failed with error {e}. Retrying in {wait_time:.2f} seconds...")
                    time.sleep(wait_time)
                else:
                    print(f"Error! All {max_retries} attempts failed. Raising the exception.")
