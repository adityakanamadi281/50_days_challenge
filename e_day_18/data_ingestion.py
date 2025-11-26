# Data Collection  from different sources

# 1). Load data from CSV/Excel

import pandas as pd

df=pd.read_csv(r"C:\Users\adity\50_days_challenge\e_day_18\data\Walmart_data.csv")

print(df)


# 2). Load data from Databases

import mysql.connector 

      # connect to mysql databse

db=mysql.connector.connect(
    host="localhost",
    user="root",
    password="Kanamadi@2002",
    database="walmart"
)
query="SELECT * FROM Sales;"
sql_db=pd.read_sql(query, db)

print(sql_db)






# 3).  Load from MongoDB

from pymongo import MongoClient

client = MongoClient("mongodb://localhost:27017/")
db1 = client["mydatabase"]
collection = db1["users"]

mongo_data = pd.DataFrame(list(collection.find()))
#print(mongo_data.head())







# 4).  Load from web Scraping

import requests

url="https://chatgpt.com/c/6926b4fc-b74c-8320-ad87-c34731030e9c"

response=requests.get(url)
api_data=pd.DataFrame(response.json())
print(response)