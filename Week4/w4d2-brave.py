from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from bs4 import BeautifulSoup
import pandas as pd
import time

# Specify the path to Brave browser and ChromeDriver
brave_path = r"C:\Program Files\BraveSoftware\Brave-Browser\Application\brave.exe"
chrome_driver_path = r"C:\WebDrivers\chromedriver.exe"

# Set up Chrome options for Brave
options = webdriver.ChromeOptions()
options.binary_location = brave_path

# Create the Service object with the path to ChromeDriver
service = Service(chrome_driver_path)

# Create ChromeDriver instance for Brave with Service and options
driver = webdriver.Chrome(service=service, options=options)

# Open the webpage
url = "https://en.wikipedia.org/wiki/List_of_largest_companies_in_the_United_States_by_revenue"
driver.get(url)

# Allow some time for the page to load
time.sleep(3)

# Extract page source using BeautifulSoup
soup = BeautifulSoup(driver.page_source, "html.parser")

# Find all tables in the page
tables = soup.find_all("table")
print(len(tables))  # This will tell you how many tables were found

# Select the desired table
table = soup.select_one(
    "#mw-content-text > div.mw-content-ltr.mw-parser-output > table:nth-child(13)"
)
print(table)

# Extract table headers
headers = [header.text.strip() for header in table.find_all("th")]

# Extract table rows
rows = []
for row in table.find_all("tr")[1:]:
    cols = row.find_all("td")
    row_data = [col.text.strip() for col in cols]
    rows.append(row_data)

# Create a DataFrame
df = pd.DataFrame(rows, columns=headers)
df
