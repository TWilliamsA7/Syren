import json
import os
import sys
import tempfile
import time
import types

from backend import gemini


class FakeInteractions:
    def create(self, **kwargs):
        FakeInteractions.last_call = kwargs
        return types.SimpleNamespace(output_text="A Boeing 737 cruising at 35,000 ft.")


# Stand in for the google-genai package, so the test needs no key, network or install.
def install_fake_genai():
    genai = types.ModuleType("google.genai")
    genai.Client = lambda: types.SimpleNamespace(interactions=FakeInteractions())
    google = types.ModuleType("google")
    google.genai = genai
    sys.modules["google"], sys.modules["google.genai"] = google, genai


def test_load_env_file():
    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, ".env")
        with open(path, "w") as f:
            f.write('# comment\nSYREN_TEST_KEY="abc123"\nSYREN_TEST_KEPT=from-file\n')
        os.environ["SYREN_TEST_KEPT"] = "from-shell"
        gemini.load_env_file(path)
    assert os.environ.pop("SYREN_TEST_KEY") == "abc123", "quotes are stripped"
    assert os.environ.pop("SYREN_TEST_KEPT") == "from-shell", "the shell wins over .env"


def test_missing_key_is_explained():
    saved_key, saved_file = os.environ.pop("GEMINI_API_KEY", None), gemini.ENV_FILE
    gemini.ENV_FILE = "/nonexistent/.env"
    try:
        gemini.ask_about_aircraft({"icao24": "a80595"})
        assert False, "should need a key"
    except RuntimeError as error:
        assert "GEMINI_API_KEY" in str(error)
    finally:
        gemini.ENV_FILE = saved_file
        if saved_key is not None:
            os.environ["GEMINI_API_KEY"] = saved_key


def test_ask_sends_the_flight_state():
    install_fake_genai()
    os.environ.setdefault("GEMINI_API_KEY", "test-key")
    state = {"icao24": "a80595", "flight_id": "DAL123", "status": {"squawk": "7700"}}
    assert gemini.ask_about_aircraft(state) == "A Boeing 737 cruising at 35,000 ft."
    call = FakeInteractions.last_call
    assert call["model"] == gemini.MODEL
    assert json.dumps(state, indent=2) in call["input"], "the whole FlightState goes in the prompt"


def wait_for_answer(job_id, timeout_s=5):
    deadline = time.time() + timeout_s
    while (status := gemini.question_status(job_id))["status"] == "working":
        assert time.time() < deadline, "the background question never finished"
        time.sleep(0.01)
    return status


def test_question_runs_in_the_background():
    install_fake_genai()
    os.environ.setdefault("GEMINI_API_KEY", "test-key")
    job_id = gemini.start_question({"icao24": "a80595"})
    assert wait_for_answer(job_id) == {"status": "done", "answer": "A Boeing 737 cruising at 35,000 ft."}
    assert gemini.question_status("no-such-job") is None


def test_failed_question_reports_the_error():
    def broken(state):
        raise RuntimeError("quota exceeded")

    saved = gemini.ask_about_aircraft
    gemini.ask_about_aircraft = broken
    try:
        job_id = gemini.start_question({"icao24": "a80595"})
        assert wait_for_answer(job_id) == {"status": "error", "error": "quota exceeded"}
    finally:
        gemini.ask_about_aircraft = saved


if __name__ == "__main__":
    test_load_env_file()
    test_missing_key_is_explained()
    test_ask_sends_the_flight_state()
    test_question_runs_in_the_background()
    test_failed_question_reports_the_error()
    print("all tests passed")
