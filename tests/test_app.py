import json
import pathlib

import pytest

from evopt.app import app


@pytest.mark.parametrize("scenario_path", [path for path in pathlib.Path(__file__).parent.glob("examples/scenario_*.json")])
def test_charge_schedule(scenario_path: pathlib.Path):
    """
    Test the /optimize endpoint of the EV optimization API.
    This test checks if the endpoint correctly processes a sample input
    and returns the expected output structure.
    """

    client = app.test_client()

    json_payload = json.loads(scenario_path.read_text())
    response = client.post("/optimize/charge-schedule", json=json_payload)

    assert response.status_code == 200
    assert response.json["status"] == "Optimal"
