import pytest

from app import app


@pytest.fixture()
def client():
    app.testing = True
    return app.test_client()


def test_index(client):
    r = client.get("/")
    assert r.status_code == 200 and b"Signal" in r.data


def test_decide_get(client):
    j = client.get("/api/decide?ns=2&ew=4&policy=optimal").get_json()
    assert j["action"] in ("EW_GREEN", "NS_GREEN")
    assert j["matches_optimal"] is True


def test_decide_post_json(client):
    r = client.post("/api/decide", json={"ns": 3, "ew": 0, "policy": "monte_carlo"})
    assert r.status_code == 200 and r.get_json()["state"] == {"ns": 3, "ew": 0}


@pytest.mark.parametrize("qs", ["ns=4&ew=0", "ns=-1&ew=0", "ns=a&ew=1", "ew=2", "ns=1&ew=6", "ns=1&ew=1&policy=nope"])
def test_decide_rejects_bad_input(client, qs):
    r = client.get(f"/api/decide?{qs}")
    assert r.status_code == 400 and "error" in r.get_json()


def test_policies_payload(client):
    j = client.get("/api/policies").get_json()
    assert set(j["policies"]) >= {"q_learning", "monte_carlo", "optimal"}
    assert len(j["policies"]["q_learning"]["Q"]) == 4
