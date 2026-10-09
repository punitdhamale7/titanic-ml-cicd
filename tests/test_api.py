
from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)


def test_home_endpoint():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["version"] == "1.0.0"


def test_prediction_endpoint():
    passenger = {
        "Pclass": 3,
        "Sex": "male",
        "Age": 22.0,
        "SibSp": 1,
        "Parch": 0,
        "Fare": 7.25,
        "Embarked": "S"
    }

    response = client.post("/predict", json=passenger)

    assert response.status_code == 200

    result = response.json()
    assert result["prediction"] in [0, 1]
    assert result["result"] in ["Survived", "Did not survive"]
    assert 0.0 <= result["survival_probability"] <= 1.0
