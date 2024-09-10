from selenium import webdriver
from bs4 import BeautifulSoup
import pandas as pd
import time

# creating instance of webdriver
driver = webdriver.Edge()

# Open the webpage
url = "https://en.wikipedia.org/wiki/List_of_largest_companies_in_the_United_States_by_revenue"
driver.get(url)

# Allow some time for the page to load
time.sleep(3)

# Extract page source using BeautifulSoup
soup = BeautifulSoup(driver.page_source, "html.parser")

tables = soup.find_all("table")
print(len(tables))  # This will tell you how many tables were found

table = soup.select_one(
    "#mw-content-text > div.mw-content-ltr.mw-parser-output > table:nth-child(13)"
)
print(table)

headers = [header.text.strip() for header in table.find_all("th")]

rows = []
for row in table.find_all("tr")[1:]:
    cols = row.find_all("td")
    row_data = [col.text.strip() for col in cols]
    rows.append(row_data)

# Create a DataFrame
df = pd.DataFrame(rows, columns=headers)
df
