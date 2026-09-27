import json
import os
import tempfile
import time

from data.replay import Player, history_status, skip, start_history, stop_history
from data.test_trace import point, write_trace
from data.trace import open_jsonl
from shared.flight_state import make_flight_state


def state(icao24, timestamp):
    return make_flight_state(timestamp=timestamp, icao24=icao24, latitude=28.4, longitude=-81.3,
                             on_ground=False, source="adsb_icao", origin="history")


def write_states(path, states):
    with open_jsonl(path, "wt") as f:
        for s in states:
            f.write(json.dumps(s) + "\n")


def icao24s_at(player, clock):
    player.clock = clock
    return sorted(json.loads(line)["icao24"] for line in player.snapshot())


def wait_for(state, timeout_s=10):
    deadline = time.time() + timeout_s
    while history_status()["state"] != state:
        assert time.time() < deadline, history_status()
        time.sleep(0.01)
    return history_status()


def test_player():
    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, "states.jsonl.gz")
        write_states(path, [state("bbbbbb", 170), state("aaaaaa", 100),  # out of order on purpose
                            state("aaaaaa", 102), state("bbbbbb", 100.5)])
        with open_jsonl(path, "at") as f:  # a line with its keys in another order
            f.write(json.dumps({"icao24": "cccccc", **state("cccccc", 169)}) + "\n")
        player = Player(path)
        assert (player.first, player.last, player.clock) == (100, 170, 100)
        assert icao24s_at(player, 100) == ["aaaaaa"]
        assert icao24s_at(player, 110) == ["aaaaaa", "bbbbbb"]
        player.clock = 110
        assert json.loads(player.snapshot()[0])["timestamp"] == 102, "latest state of each aircraft"
        assert icao24s_at(player, 170) == ["bbbbbb", "cccccc"], "aaaaaa dropped after 60 s of silence"
        assert icao24s_at(player, 110) == ["aaaaaa", "bbbbbb"], "going back works"
        player.skip(-1000)
        assert player.clock == 100, "can't go before the first state"
        player.skip(1000)
        assert player.clock == 170, "can't go past the last state"


def test_player_runs_detection_and_recomputes_on_rewind():
    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, "states.jsonl")
        emergency = state("a80595", 100)
        emergency["status"]["squawk"] = "7700"
        clean = state("a80595", 110)
        write_states(path, [emergency, clean])
        player = Player(path)

        first = json.loads(player.snapshot()[0])
        assert any(a["type"] == "EMERGENCY_SQUAWK" for a in first["detection"]["anomalies"])

        player.clock = 110
        second = json.loads(player.snapshot()[0])
        assert second["detection"]["anomalies"] == []

        player.clock = 100
        rewound = json.loads(player.snapshot()[0])
        assert any(a["type"] == "EMERGENCY_SQUAWK" for a in rewound["detection"]["anomalies"])


def test_start_skip_stop():
    with tempfile.TemporaryDirectory() as root:
        day = os.path.join(root, "fixed", "2026-09-24")
        write_trace(day, "a80595", [point(16 * 3600 + 10 * i) for i in range(100)])
        out_path = os.path.join(root, "history.jsonl")

        start_history("2026-09-24", "16:00", 1, interval_s=0.05, out_path=out_path, root=root)
        status = wait_for("playing")
        first = status["first"]
        assert status["date"] == "2026-09-24" and status["last"] == first + 990
        time.sleep(0.2)
        with open(out_path) as f:
            assert [json.loads(line)["icao24"] for line in f] == ["a80595"]

        skip(500)
        assert history_status()["clock"] >= first + 500
        skip(-10000)
        assert history_status()["clock"] == first

        stop_history()
        assert history_status()["state"] == "stopped"
        time.sleep(0.2)
        assert os.path.exists(out_path), "history.jsonl stays after exiting date mode"
        try:
            skip(60)
            assert False, "skip needs a day playing"
        except RuntimeError:
            pass


def test_bad_date_fails():
    with tempfile.TemporaryDirectory() as root:
        out_path = os.path.join(root, "history.jsonl")
        with open(out_path, "w") as f:
            f.write("planes from the last date\n")
        try:
            start_history("2010-01-01", out_path=out_path, root=root)
            assert False, "invalid dates should fail before a replay starts"
        except ValueError as error:
            assert "not a day in the history archive" in str(error)
        with open(out_path) as f:
            assert f.read() == "planes from the last date\n", "invalid dates must not alter replay output"


def test_concurrent_public_cache_miss_is_rejected():
    import data.replay as replay

    with tempfile.TemporaryDirectory() as root:
        with replay._lock:
            was_downloading = replay._download_in_progress
            replay._download_in_progress = True
        try:
            try:
                start_history("2026-09-24", root=root)
                assert False, "a second cache miss should be rejected while one download runs"
            except RuntimeError as error:
                assert "already in progress" in str(error)
        finally:
            with replay._lock:
                replay._download_in_progress = was_downloading


if __name__ == "__main__":
    test_player()
    test_start_skip_stop()
    test_bad_date_fails()
    test_concurrent_public_cache_miss_is_rejected()
    print("all tests passed")
