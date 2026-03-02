# ingestion/csv_ingestion.py

import os
import pandas as pd
from langchain_core.documents import Document


def process_company_profile(path):
    df = pd.read_csv(path)
    row = df.iloc[0]

    text = f"""
    Company: {row['Longname']}
    Sector: {row['Sector']}
    Industry: {row['Industry']}
    Market Cap: {row['Marketcap']}
    Business Summary: {row['Longbusinesssummary']}
    """

    return Document(
        page_content=text,
        metadata={
            "source": "csv",
            "type": "company_profile"
        }
    )


def process_monthly_stock(path):
    df = pd.read_csv(path)
    df['Date'] = pd.to_datetime(df['Date'])
    df['Month'] = df['Date'].dt.to_period('M')

    documents = []

    for month, group in df.groupby('Month'):
        text = f"Microsoft stock performance for {month}:\n"

        for _, row in group.iterrows():
            text += f"{row['Date'].date()}: Close {row['Close']}, Volume {row['Volume']}\n"

        documents.append(
            Document(
                page_content=text,
                metadata={
                    "source": "csv",
                    "type": "stock_prices",
                    "month": str(month)
                }
            )
        )

    return documents


def process_market_index(path):
    df = pd.read_csv(path)
    df['Date'] = pd.to_datetime(df['Date'])
    df['Month'] = df['Date'].dt.to_period('M')

    documents = []

    for month, group in df.groupby('Month'):
        text = f"S&P 500 performance for {month}:\n"

        for _, row in group.iterrows():
            text += f"{row['Date'].date()}: Index {row['S&P500']}\n"

        documents.append(
            Document(
                page_content=text,
                metadata={
                    "source": "csv",
                    "type": "market_index",
                    "month": str(month)
                }
            )
        )

    return documents


def ingest_csv_folder(folder_path):
    documents = []

    for file in os.listdir(folder_path):
        path = os.path.join(folder_path, file)

        if file == "msft_info_clean.csv":
            print(f"    📊 Processing CSV: {file}...")
            docs = [process_company_profile(path)]
            documents.extend(docs)

        elif file == "msft_prices_2024.csv":
            print(f"    📊 Processing CSV: {file}...")
            docs = process_monthly_stock(path)
            documents.extend(docs)

        elif file == "market_index_2024.csv":
            print(f"    📊 Processing CSV: {file}...")
            docs = process_market_index(path)
            documents.extend(docs)

    return documents
