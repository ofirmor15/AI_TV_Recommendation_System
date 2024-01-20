import pytest
import os
import pandas as pd
import numpy as np
import pickle
from unittest.mock import patch, MagicMock
from showSuggesterAI import (
    generate_embeddings,
    load_embeddings_from_pickle,
    save_embeddings_to_pickle,
    get_user_favorite_shows,
    calculate_average_vector,
    get_recommendations,
)


@pytest.fixture
def setup_embeddings():
    return {"show1": np.array([1, 2, 3]), "show2": np.array([4, 5, 6])}


def test_generate_embeddings_valid(setup_embeddings):
    setup_embeddings["desc1"] = np.array([7, 8, 9])
    setup_embeddings["desc2"] = np.array([10, 11, 12])
    with patch("pandas.read_csv") as mock_read_csv:
        mock_read_csv.return_value = pd.DataFrame(
            {"Title": ["show1", "show2"], "Description": ["desc1", "desc2"]}
        )
        with patch(
            "showSuggesterAI.get_embedding", side_effect=lambda x: setup_embeddings[x]
        ):
            embeddings = generate_embeddings("dummy/path.csv")
            assert "show1" in embeddings and "show2" in embeddings


def test_generate_embeddings_invalid_path():
    with pytest.raises(FileNotFoundError):
        generate_embeddings("invalid/path.csv")


def test_load_embeddings_from_pickle_existing_file(setup_embeddings, tmp_path):
    pickle_file = tmp_path / "test_embeddings.pkl"

    # Use pickle to dump the dictionary to ensure compatibility
    with open(pickle_file, "wb") as f:
        pickle.dump(setup_embeddings, f)

    loaded_embeddings = load_embeddings_from_pickle(pickle_file)
    assert "show1" in loaded_embeddings and "show2" in loaded_embeddings


def test_load_embeddings_from_pickle_non_existing_file(tmp_path):
    non_existing_pickle_file = tmp_path / "non_existing.pkl"
    with patch(
        "showSuggesterAI.generate_embeddings", return_value={}
    ) as mock_generate_embeddings:
        loaded_embeddings = load_embeddings_from_pickle(non_existing_pickle_file)
        mock_generate_embeddings.assert_called_once()
        assert loaded_embeddings == {}


def test_save_embeddings_to_pickle(setup_embeddings, tmp_path):
    pickle_file = tmp_path / "test_embeddings.pkl"
    save_embeddings_to_pickle(setup_embeddings, pickle_file)
    assert os.path.exists(pickle_file)


def test_calculate_average_vector(setup_embeddings):
    average_vector = calculate_average_vector(["show1", "show2"], setup_embeddings)
    assert average_vector.tolist() == [2.5, 3.5, 4.5]


@patch("builtins.input", side_effect=["show1, show2", "y"])
def test_get_user_favorite_shows_valid(mock_input):
    with patch(
        "showSuggesterAI.process.extractOne", side_effect=lambda show, _: (show, 100)
    ):
        favorite_shows = get_user_favorite_shows(["show1", "show2", "show3"])
        assert favorite_shows == ["show1", "show2"]


def test_get_recommendations(setup_embeddings):
    recommendations = get_recommendations(["show1"], setup_embeddings, top_n=1)
    assert "show2" in recommendations
    assert len(recommendations) == 1


def test_generate_embeddings_with_missing_descriptions(setup_embeddings):
    with patch("pandas.read_csv") as mock_read_csv:
        mock_read_csv.return_value = pd.DataFrame(
            {"Title": ["show1", "show3"], "Description": ["desc1", ""]}
        )
        with patch(
            "showSuggesterAI.get_embedding",
            side_effect=lambda x: setup_embeddings.get(x, np.array([])),
        ):
            embeddings = generate_embeddings("dummy/path.csv")
            # Check if both shows are in embeddings, including the one with empty description
            assert "show1" in embeddings and "show3" in embeddings


def test_load_embeddings_from_pickle_corrupt_file(tmp_path):
    corrupt_pickle_file = tmp_path / "corrupt.pkl"
    with open(corrupt_pickle_file, "wb") as f:
        f.write(b"not a pickle")

    with pytest.raises(Exception):  # Replace with specific exception if known
        load_embeddings_from_pickle(corrupt_pickle_file)


def test_get_user_favorite_shows_single_input():
    input_responses = iter(["show1", "y"])

    def input_side_effect(_):
        response = next(
            input_responses,
            "y",
        )  # Default to 'y' after the first response
        print(f"Mock input response: {response}")  # Diagnostic print
        return response

    with patch("builtins.input", side_effect=input_side_effect):
        with patch(
            "showSuggesterAI.process.extractOne",
            side_effect=lambda show, _: (show, 100),
        ):
            favorite_shows = get_user_favorite_shows(["show1", "show2", "show3"])
            assert favorite_shows == None


def test_calculate_average_vector_with_missing_shows(setup_embeddings):
    average_vector = calculate_average_vector(
        ["show1", "missing_show"], setup_embeddings
    )
    expected_vector = np.array(
        [1, 2, 3]
    )  # Expected result based on how your function handles missing shows
    np.testing.assert_array_equal(average_vector, expected_vector)


def test_get_recommendations_with_empty_embeddings():
    recommendations = get_recommendations(["show1"], {}, top_n=1)
    assert len(recommendations) == 0
