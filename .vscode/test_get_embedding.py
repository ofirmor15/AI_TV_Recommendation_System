from showSuggesterAI import get_embedding


def test_get_embedding():
    # Test case 1
    assert type(get_embedding("Friends")) == list

    # Test case 2
    assert type(get_embedding("Breaking bad")) == list

    # Test case 3
    assert type(get_embedding("Stranger things")) == list


def main():
    test_get_embedding()
