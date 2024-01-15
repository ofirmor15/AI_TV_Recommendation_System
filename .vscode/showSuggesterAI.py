import openai
import pandas as pd
import pickle
import os
import logging
from thefuzz import process
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

openai.api_key = "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"


def get_embedding(tv_show_description):
    embedding = openai.Embedding.create(
        input=tv_show_description, engine="text-embedding-ada-002"
    )["data"][0]["embedding"]
    return embedding


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
    while True:
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


def calculate_average_vector(shows, embeddings_dict):
    vectors = [embeddings_dict[show] for show in shows if show in embeddings_dict]
    return np.mean(vectors, axis=0)


def get_recommendations(input_shows, embeddings_dict, top_n=5):
    average_vector = calculate_average_vector(input_shows, embeddings_dict)

    similarities = {}
    for show, vector in embeddings_dict.items():
        if show not in input_shows:
            similarity = cosine_similarity([average_vector], [vector])[0][0]
            similarities[show] = similarity

    # Sort the shows based on similarity scores
    recommended_shows = sorted(similarities, key=similarities.get, reverse=True)[:top_n]

    # Convert similarity scores to percentages
    max_similarity = max(similarities.values())
    recommendations = {
        show: round((similarities[show] / max_similarity) * 100, 2)
        for show in recommended_shows
    }

    return recommendations


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
    print("Here are the TV shows that I think you would love:")
    for show, score in recommendations.items():
        print(f"{show} ({score}%)")


if __name__ == "__main__":
    main()
