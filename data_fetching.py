# Example python program to read data from a PostgreSQL table

# and load into a pandas DataFrame

# import psycopg2 as pg
# import pandas as pd
#
# engine = pg.connect("dbname='ai-hunt' user='postgres' host='139.162.115.227' password='Paracel2022!@#'")
# print("Engine connects successfully.")
# print(engine.deferrable)
#
# df = pd.read_sql('SELECT * FROM web_server.aa_monitors', con=engine)
# print("Table connects successfully.")
#
# df.to_excel("test_car_auction_price_data.xlsx", index=True)
# print("Create Excel file successfully")

import logging
import time
import datetime as dt
import psycopg2 as pg
import pandas as pd


logging.basicConfig(filename="log_fetching.txt",
                    format='%(asctime)s %(message)s',
                    filemode='w')


def calcProcessTime(starttime):
    telapsed = time.time() - starttime
    testimated = telapsed

    finishtime = starttime + testimated
    finishtime = dt.datetime.fromtimestamp(finishtime).strftime("%H:%M:%S")  # in time

    lefttime = testimated - telapsed  # in seconds

    return int(telapsed), int(lefttime), finishtime


logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

engine = pg.connect("dbname='aihunt_staging' user='postgres' host='172.104.118.38' password='paracel2022'")
print("Engine connects successfully.")
logger.info("Engine connects successfully.")
start_time = time.time()
df = pd.read_sql('SELECT * FROM public.raw_historical_prices_asnet', con=engine)
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time}")
print("Table connects successfully.")
logger.info("Table connects successfully.")

df.to_csv("data/asnet_prices.csv", index=True)
print("Create Excel file successfully")
logger.info("Create Excel file successfully")