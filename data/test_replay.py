
import json
import os
import tempfile

from data.replay import replay, snapshots
from data.trace import open_jsonl
from shared.flight_state import make_flight_state


def state(icao24, timestamp):
    return make_flight_state(timestamp=timestamp, icao24=icao24, latitude=28.4, longitude=-81.3,
                             on_ground=False, source="adsb_icao", origin="history")


STATES = [state("aaaaaa", 100), state("bbbbbb", 100.5), state("aaaaaa", 102), state("bbbbbb", 170)]


def test_snapshots():
    frames = list(snapshots(STATES, 10))
    assert [clock for clock, _ in frames] == [100, 110, 120, 130, 140, 150, 160, 170]
    assert frames[0][1] == [state("aaaaaa", 100)]
    assert frames[1][1] == [state("aaaaaa", 102), state("bbbbbb", 100.5)], "latest state of each"
    assert len(frames[6][1]) == 2, "both still heard within GONE_AFTER_S"
    assert frames[7][1] == [state("bbbbbb", 170)], "aaaaaa dropped after 60 s of silence"
    assert list(snapshots([], 10)) == []


def test_replay_writes_the_last_snapshot():
    with tempfile.TemporaryDirectory() as folder:
        states_path = os.path.join(folder, "states.jsonl.gz")
        with open_jsonl(states_path, "wt") as f:
            for s in STATES:
                f.write(json.dumps(s) + "\n")
        out_path = os.path.join(folder, "history.jsonl")
        replay(states_path, out_path, speed=1000, interval_s=0.01)
        with open(out_path) as f:
            assert [json.loads(line) for line in f] == [state("bbbbbb", 170)]
        assert not os.path.exists(out_path + ".tmp")


if __name__ == "__main__":
    test_snapshots()
    test_replay_writes_the_last_snapshot()
    print("all tests passed")