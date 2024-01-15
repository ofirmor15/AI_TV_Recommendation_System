from showSuggesterAI import get_embedding
import os
import pickle
from showSuggesterAI import save_embeddings_to_pickle
from showSuggesterAI import get_embedding
from showSuggesterAI import load_embeddings_from_pickle
from showSuggesterAI import calculate_average_vector


def test_get_embedding():
    # Test case 1
    assert (
        type(
            get_embedding(
                "A high school chemistry teacher diagnosed with inoperable lung cancer turns to manufacturing and selling methamphetamine in order to secure his family's future"
            )
        )
        == list
    )

    # Test case 2
    assert (
        type(
            get_embedding(
                "When a young boy disappears, his mother, a police chief and his friends must confront terrifying supernatural forces in order to get him back."
            )
        )
        == list
    )

    # Test case 3
    assert (
        type(
            get_embedding(
                "A father recounts to his children - through a series of flashbacks - the journey he and his four best friends took leading up to him meeting their mother."
            )
        )
        == list
    )


def test_save_embeddings_to_pickle():
    # Create a temporary pickle file
    pickle_file = "temp.pickle"

    # Create a dummy embeddings dictionary
    embeddings_dict = {"show1": [0.1, 0.2, 0.3], "show2": [0.4, 0.5, 0.6]}

    # Save the embeddings to the pickle file
    save_embeddings_to_pickle(embeddings_dict, pickle_file)

    # Check if the pickle file exists
    assert os.path.exists(pickle_file)

    # Load the saved embeddings from the pickle file
    with open(pickle_file, "rb") as f:
        saved_embeddings_dict = pickle.load(f)

    # Check if the loaded embeddings match the original embeddings
    assert saved_embeddings_dict == embeddings_dict

    # Delete the temporary pickle file
    os.remove(pickle_file)


def test_load_embeddings_from_pickle():
    # Create a temporary pickle file
    pickle_file = "temp.pickle"

    # Create a dummy embeddings dictionary
    embeddings_dict = {"show1": [0.1, 0.2, 0.3], "show2": [0.4, 0.5, 0.6]}

    # Save the embeddings to the pickle file
    with open(pickle_file, "wb") as f:
        pickle.dump(embeddings_dict, f)

    # Load the saved embeddings from the pickle file
    loaded_embeddings_dict = load_embeddings_from_pickle(pickle_file)

    # Check if the loaded embeddings match the original embeddings
    assert loaded_embeddings_dict == embeddings_dict

    # Delete the temporary pickle file
    os.remove(pickle_file)


def test_calculate_average_vector():
    # Create a dummy embeddings dictionary
    embeddings_dict = {
        "show1": [0.1, 0.2, 0.3],
        "show2": [0.4, 0.5, 0.6],
        "show3": [0.7, 0.8, 0.9],
    }

    # Test case 1
    input_shows = ["show1", "show2"]
    expected_result = [0.25, 0.35, 0.45]
    assert calculate_average_vector(input_shows, embeddings_dict) == expected_result

    # Test case 2
    input_shows = ["show2", "show3"]
    expected_result = [0.55, 0.65, 0.75]
    assert calculate_average_vector(input_shows, embeddings_dict) == expected_result

    # Test case 3
    input_shows = ["show1", "show3"]
    expected_result = [0.4, 0.5, 0.6]
    assert calculate_average_vector(input_shows, embeddings_dict) == expected_result
