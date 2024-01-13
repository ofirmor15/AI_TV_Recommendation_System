import pandas as pd
import openai
import pickle
from openai import OpenAI

openai.api_key = "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"
# Read the CSV file
input_datapath = ".vscode\imdb_tvshows - imdb_tvshows.csv"

df = pd.read_csv(input_datapath)  # Add this line to create the DataFrame


def get_embedding(text):
    model = "text-embedding-ada-002"
    text = text.replace("\n", " ")
    response = openai.Embedding.create(input=text, engine=model)
    return response["data"][0]["embedding"]


df["ada_embedding"] = df.combine.apply(
    lambda x: get_embedding(x, model="text-embedding-ada-002")
)
df.to_csv("output/embedded_1k_reviews.csv", index=False)
