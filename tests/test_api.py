GOOD = {"type": "M", "air_temperature": 300, "process_temperature": 310,
        "rotational_speed": 1500, "torque": 40, "tool_wear": 100}


def test_health(client):
    j = client.get("/health").json()
    assert j["status"] == "ok" and j["model_loaded"] and j["knowledge_chunks"] > 0


def test_predict_valid(client):
    r = client.post("/predict", json=GOOD)
    assert r.status_code == 200
    j = r.json()
    assert 0 <= j["failure_probability"] <= 1 and "disclaimer" in j and isinstance(j["alert"], bool)


def test_predict_high_wear_returns_in_range(client):
    r = client.post("/predict", json={**GOOD, "tool_wear": 600, "torque": 70})
    assert r.status_code == 200
    assert any("outside the training range" in w for w in r.json()["warnings"])


def test_invalid_inputs_rejected(client):
    assert client.post("/predict", json={**GOOD, "torque": -5}).status_code == 422
    assert client.post("/predict", json={**GOOD, "type": "X"}).status_code == 422
    assert client.post("/predict", json={**GOOD, "air_temperature": 5000}).status_code == 422
    bad = dict(GOOD); bad.pop("torque")
    assert client.post("/predict", json=bad).status_code == 422
    assert client.post("/predict", json={**GOOD, "extra": 1}).status_code == 422
    assert client.post("/predict", json={**GOOD, "tool_wear": "abc"}).status_code == 422


def test_model_missing_gives_503(client_no_model):
    assert client_no_model.get("/health").json()["status"] == "degraded"
    assert client_no_model.post("/predict", json=GOOD).status_code == 503


def test_guidance_endpoint(client):
    j = client.post("/guidance", json={"query": "replace worn cutting tool insert"}).json()
    assert j["passages"] and j["passages"][0]["source"] == "test_manual.txt"


def test_guidance_irrelevant_query_returns_no_evidence(client):
    j = client.post("/guidance", json={"query": "quantum chromodynamics lattice gauge"}).json()
    assert j["passages"] == [] and "No sufficiently relevant" in j["message"]


def test_guidance_validation(client):
    assert client.post("/guidance", json={"query": "x"}).status_code == 422
