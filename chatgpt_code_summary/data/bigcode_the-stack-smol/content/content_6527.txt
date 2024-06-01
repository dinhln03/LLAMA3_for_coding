import pytest

from app import crud
from app.schemas import EpisodeCreate
from app.schemas.episode import EpisodeSearch
from app.tests.utils import random_segment


def test_get_episode(db):
    ep_in = EpisodeCreate(name="ep1", air_date="2022-03-04", segment=random_segment())
    ep = crud.episode.create(db, ep_in)

    assert crud.episode.get(db, ep.id)


def test_get_multi_episode(db):
    ep_in = EpisodeCreate(name="ep11", air_date="2022-03-04", segment=random_segment())
    crud.episode.create(db, ep_in)
    ep_in = EpisodeCreate(name="ep12", air_date="2022-03-04", segment=random_segment())
    crud.episode.create(db, ep_in)

    assert len(crud.episode.get_multi(db)) == 2


@pytest.mark.parametrize(
    "test_input,expected",
    [
        ({"air_date__lte": "2022-03-01"}, 1),
        ({"air_date__gte": "2022-04-01"}, 4),
        ({"air_date__gte": "2022-03-01", "air_date__lte": "2022-05-01"}, 3),
    ],
)
def test_date_search_episode(db, setup, test_input, expected):
    obj_in = EpisodeSearch(**test_input)
    assert len(crud.episode.search(db, obj_in)) == expected


@pytest.mark.parametrize(
    "test_input,expected",
    [
        ({"name__icontains": "Nibelheim"}, 0),
        ({"name__icontains": "Midgar"}, 1),
        ({"name__icontains": "mId"}, 1),
        ({"name__icontains": "GaR"}, 1),
    ],
)
def test_name_search_episode(db, setup, test_input, expected):
    obj_in = EpisodeSearch(**test_input)
    assert len(crud.episode.search(db, obj_in)) == expected


@pytest.mark.parametrize(
    "test_input,expected",
    [
        ({"segment": "s01e15"}, 0),
        ({"segment": "S01e01"}, 1),
    ],
)
def test_segment_search_episode(db, setup, test_input, expected):
    obj_in = EpisodeSearch(**test_input)
    assert len(crud.episode.search(db, obj_in)) == expected
