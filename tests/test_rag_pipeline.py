import json

import pytest

from rag_pipeline import load_places, RAGPipeline


# --- load_places ---------------------------------------------------------

def test_load_places_loads_the_real_data_file():
    places = load_places()
    assert len(places) > 0
    for place in places:
        assert "name" in place
        assert "desc" in place


def test_load_places_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_places(tmp_path / "does_not_exist.json")


def test_load_places_empty_list_raises(tmp_path):
    data_file = tmp_path / "empty.json"
    data_file.write_text("[]")
    with pytest.raises(ValueError):
        load_places(data_file)


# --- RAGPipeline (integration - downloads/loads real models) -------------

@pytest.fixture(scope="module")
def small_pipeline(tmp_path_factory):
    """A RAGPipeline over a tiny, fixed knowledge base so retrieval is unambiguous."""
    data_file = tmp_path_factory.mktemp("data") / "places.json"
    data_file.write_text(json.dumps([
        {"name": "Eiffel Tower", "desc": "An iron tower in Paris, France."},
        {"name": "Big Ben", "desc": "A clock tower in London, England."},
    ]))
    return RAGPipeline(data_path=data_file)


def test_retrieve_returns_the_closest_place(small_pipeline):
    top = small_pipeline.retrieve("What tower is in Paris?", k=1)
    assert "Eiffel Tower" in top[0]


def test_retrieve_k_returns_requested_count(small_pipeline):
    top = small_pipeline.retrieve("Tell me about a tower", k=2)
    assert len(top) == 2


def test_ask_returns_a_non_empty_answer(small_pipeline):
    answer = small_pipeline.ask("What is the Eiffel Tower?")
    assert isinstance(answer, str)
    assert len(answer.strip()) > 0
