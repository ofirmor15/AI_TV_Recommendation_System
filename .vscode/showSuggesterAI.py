import base64
import openai
import pandas as pd
import pickle
import os
import logging
from thefuzz import process
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import requests
from PIL import Image
from io import BytesIO
from openai import OpenAI
import time
from dotenv import load_dotenv

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
openai.api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI()
logging.basicConfig(
    filename="logging.log",
    level=logging.INFO,
    format="%(asctime)s:%(levelname)s:%(message)s",
)


def get_embedding(tv_show_description):
    embedding = client.embeddings.create(
        input=tv_show_description, model="text-embedding-ada-002"
    )
    return embedding.data[0].embedding


def generate_embeddings(csv_file_path):
    # Load TV shows data
    tv_shows_df = pd.read_csv(csv_file_path)

    # Initialize embeddings dictionary
    embeddings_dict = {}

    # Iterate over each TV show
    for _, row in tv_shows_df.iterrows():
        tv_show_title = row["Title"]
        tv_show_description = row["Description"]

        # Get embedding for TV show
        embedding = get_embedding(tv_show_description)

        # Add embedding to dictionary
        embeddings_dict[tv_show_title] = embedding

    return embeddings_dict


def load_embeddings_from_pickle(pickle_file):
    if os.path.exists(pickle_file):
        with open(pickle_file, "rb") as file:
            embeddings_dict = pickle.load(file)
    else:
        embeddings_dict = generate_embeddings()
        save_embeddings_to_pickle(embeddings_dict, pickle_file)
    return embeddings_dict


def save_embeddings_to_pickle(embeddings_dict, pickle_file):
    with open(pickle_file, "wb") as file:
        pickle.dump(embeddings_dict, file)


def get_user_favorite_shows(tv_shows_list):
    count = 0
    while True:
        count += 1
        user_input = input(
            "Which TV shows did you love watching? Separate them by a comma.\nMake sure to enter more than 1 show: "
        )
        user_shows = [show.strip() for show in user_input.split(",")]

        if len(user_shows) > 1:
            # Fuzzy matching for each show
            matched_shows = []
            for user_show in user_shows:
                closest_match = process.extractOne(user_show, tv_shows_list)
                matched_shows.append(closest_match[0] if closest_match else user_show)

            confirmation = input(
                f"Just to make sure, do you mean {', '.join(matched_shows)}? (y/n): "
            )
            if confirmation.lower() == "y":
                return matched_shows
            else:
                print(
                    "Sorry about that. Let's try again, please make sure to write the names of the TV shows correctly."
                )
        else:
            print("Please enter more than one show.")
            if count == 3:
                print("Please restart the program and try again...")
                return None


def calculate_average_vector(shows, embeddings_dict):
    vectors = [embeddings_dict[show] for show in shows if show in embeddings_dict]
    return np.mean(vectors, axis=0)


def get_recommendations(input_shows, embeddings_dict, top_n=5):
    # Return empty list or dictionary if embeddings_dict is empty
    if not embeddings_dict:
        return {}  # Or return [] depending on what your function is expected to return

    average_vector = calculate_average_vector(input_shows, embeddings_dict)

    similarities = {}
    for show, vector in embeddings_dict.items():
        if show not in input_shows:
            similarity = cosine_similarity([average_vector], [vector])[0][0]
            similarities[show] = similarity

    if not similarities:
        return {}  # Or return [] depending on what your function is expected to return

    # Sort the shows based on similarity scores
    recommended_shows = sorted(similarities, key=similarities.get, reverse=True)[:top_n]

    # Convert similarity scores to percentages
    max_similarity = max(similarities.values())
    recommendations = {
        show: round((similarities[show] / max_similarity) * 100, 2)
        for show in recommended_shows
    }

    return recommendations


def generate_image_with_dalle(newshow_and_description):
    response = client.images.generate(
        model="dall-e-3",
        prompt=f"The following text is a TV show name and description:\n{newshow_and_description}\nCreate a TV poster for this show. Make it as realistic as possible and in the style of a Hollywood TV show poster.\n\nImage:",
        size="1024x1024",
        quality="standard",
        n=1,
        response_format="url",
    )
    image_url = response.data[0].url
    return image_url


def generate_show_name(tv_shows):
    # Split the input string into individual TV show names
    show_list = []
    for show in tv_shows:
        show_list.append(show)

    message = {
        "role": "user",
        "content": "Based on the following TV shows, suggest a name for a new TV show and a description for that show: "
        + ", ".join(show_list)
        + "\nMake sure do write in the following format:\nNew Show Name:\nNew Show Description:",
    }

    try:
        # Call GPT-3.5-turbo using the chat API
        response = client.chat.completions.create(
            model="gpt-3.5-turbo", messages=[message]
        )

        return response.choices[0].message.content
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def open_dalle_image(response):
    # Check if the response is a URL
    if isinstance(response, str) and response.startswith(("http://", "https://")):
        # Handle URL
        response = requests.get(response)
        if response.status_code == 200:
            image = Image.open(BytesIO(response.content))
            image.show()
        else:
            print("Failed to retrieve the image from URL.")

    # Check if the response is Base64 encoded
    elif isinstance(response, str) and ";base64," in response:
        base64_data = response.split(";base64,")[-1]
        image_data = base64.b64decode(base64_data)
        image = Image.open(BytesIO(image_data))
        image.show()

    # Handle binary data (assuming it's not URL or Base64)
    elif isinstance(response, bytes):
        image = Image.open(BytesIO(response))
        image.show()

    else:
        print("Unknown image format.")


def main():
    # Path for the CSV file and pickle file
    csv_file_path = ".vscode\imdb_tvshows - imdb_tvshows.csv"
    pickle_file = "tv_shows_embeddings.pkl"

    # Load embeddings
    if os.path.exists(pickle_file):
        embeddings_dict = load_embeddings_from_pickle(pickle_file)
    else:
        # Ensure the CSV file exists
        if os.path.exists(csv_file_path):
            embeddings_dict = generate_embeddings(csv_file_path)
            save_embeddings_to_pickle(embeddings_dict, pickle_file)
        else:
            logging.exception("CSV file not found.")
            return

    # Get user favorite shows
    user_favorite_shows = get_user_favorite_shows(list(embeddings_dict.keys()))

    # Generate recommendations
    recommendations = get_recommendations(user_favorite_shows, embeddings_dict)
    shows = ""
    print("Here are the TV shows that I think you would love:")
    for show, score in recommendations.items():
        shows += f"{show} ({score}%)\n"
    print(shows)
    show_name_and_description = generate_show_name(user_favorite_shows)
    recommneded_show_and_description = generate_show_name(shows)
    show1name = show_name_and_description.split("New Show Description:")[0].strip()
    show1description = show_name_and_description.split("New Show Description:")[
        1
    ].strip()
    show2name = recommneded_show_and_description.split("New Show Description:")[
        0
    ].strip()
    show2description = recommneded_show_and_description.split("New Show Description:")[
        1
    ].strip()

    logging.info(
        "Show + description:"
        + show1name
        + "\nRecommendation:"
        + recommneded_show_and_description
        + "\nOnly show name:"
        + show1name
        + "\nOnly show description:"
        + show1description
        + "\nOnly recommendation name:"
        + show2name
        + "\nOnly recommendation description:"
        + show2description
    )
    print(
        f"”I have also created just for you two shows which I think you would love. Show #1 is based on the fact that you loved the input shows that you gave me. Its name is {show1name} and it is about: {show1description}. Show #2 is based on the shows that I recommended for you. Its name is {show2name} and it is about: {show2description}. Here are also the 2 tv show ads. Hope you like them!"
    )
    image_1 = generate_image_with_dalle(show_name_and_description)
    image_2 = generate_image_with_dalle(recommneded_show_and_description)
    open_dalle_image(image_1)
    open_dalle_image(image_2)


if __name__ == "__main__":
    main()
