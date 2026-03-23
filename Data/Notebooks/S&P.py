import os
import requests
import json
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
sp_usr = os.getenv("SP_USERNAME")
sp_pwd = os.getenv("SP_PASSWORD")

# ── Step 1: Get token ─────────────────────────────────────────────────────────
token_url = "https://api-ciq.marketintelligence.spglobal.com/gdsapi/rest/authenticate/api/v1/token"

r = requests.post(
    token_url,
    headers={"Content-Type": "application/x-www-form-urlencoded"},
    data={"username": sp_usr, "password": sp_pwd}
)
r.raise_for_status()
access_token = r.json()["access_token"]
print("✅ Authenticated")

# ── Step 2: Query the API ─────────────────────────────────────────────────────
api_url = "https://api-ciq.marketintelligence.spglobal.com/gdsapi/rest/v3/clientservice.json"

headers = {
    "Authorization": f"Bearer {access_token}",
    "Content-Type": "application/json"
}

payload = {
    "inputRequests": [
        # Company name
        {"function": "GDSP", "identifier": "AAPL:", "mnemonic": "IQ_COMPANY_NAME"},
        # Latest close price
        {"function": "GDSP", "identifier": "AAPL:", "mnemonic": "IQ_CLOSEPRICE"},
        # Total revenue - latest fiscal year
        {"function": "GDSP", "identifier": "AAPL:", "mnemonic": "IQ_TOTAL_REV",
         "properties": {"periodType": "IQ_FY"}},
        # Revenue history - last 5 fiscal years
        {"function": "GDST", "identifier": "AAPL:", "mnemonic": "IQ_TOTAL_REV",
         "properties": {"periodType": "IQ_FY", "startDate": "01/01/2019", 
                        "endDate": "", "frequency": "A"}},
    ]
}

resp = requests.post(api_url, headers=headers, json=payload)
resp.raise_for_status()

data = resp.json()["GDSSDKResponse"]
for item in data:
    print(f"\n── {item['Mnemonic']} ──")
    if item.get("ErrMsg"):
        print(f"  Error: {item['ErrMsg']}")
    elif item.get("Rows"):
        for row in item["Rows"]:
            print(f"  {row['Row']}")