import gzip
import io
import os
import pathlib
import sys
import tarfile
import tempfile
import threading
import time
import datetime

from data.history import (
    available_range, days_on_disk, find_release, setup_fixed_days, start_swap, swap_day,
    swap_status, trace_path,
)

TRACES = {
    "./traces/95/trace_full_a80595.json": gzip.compress(b'{"icao": "a80595", "trace": []}'),
    "./traces/95/trace_full_~26f495.json": gzip.compress(b'{"icao": "~26f495", "trace": []}'),
    "./traces/0c/trace_full_abc10c.json": gzip.compress(b'{"icao": "abc10c", "trace": []}'),
}
OTHER_FILES = {"./README.txt": b"readme", "./acas/acas.csv.gz": gzip.compress(b"acas")}


def fake_release(folder):
    """A day's archive like adsb.lol's, split into two parts, as file:// URLs."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as tar:
        for name, content in {**OTHER_FILES, **TRACES}.items():
            info = tarfile.TarInfo(name)
            info.size = len(content)
            tar.addfile(info, io.BytesIO(content))
    data = buffer.getvalue()
    split = len(data) // 2 + 7  # deliberately not on a tar block boundary
    parts = []
    for suffix, chunk in (("aa", data[:split]), ("ab", data[split:])):
        path = pathlib.Path(folder, f"day.tar.{suffix}")
        path.write_bytes(chunk)
        parts.append((path.as_uri(), len(chunk)))
    return parts


def no_download(date):
    raise AssertionError(f"should not have downloaded {date}")


def wait_for_swap(timeout_s=10):
    deadline = time.time() + timeout_s
    while swap_status()["state"] == "downloading":
        assert time.time() < deadline, "background swap took too long"
        time.sleep(0.02)
    return swap_status()


def test_only_traces_are_kept():
    with tempfile.TemporaryDirectory() as root:
        folder = swap_day("2026-09-20", root, find=lambda date: fake_release(root))
        for name, content in TRACES.items():
            icao24 = os.path.basename(name)[len("trace_full_"):-len(".json")]
            with open(trace_path("2026-09-20", icao24, root), "rb") as f:
                assert f.read() == content, name
        assert not os.path.exists(os.path.join(folder, "README.txt"))
        assert not os.path.exists(os.path.join(folder, "acas"))
        assert trace_path("2026-09-20", "A80595", root) is not None, "icao24 is case-insensitive"
        assert trace_path("2026-09-20", "ffffff", root) is None


def test_setup_fixed_days():
    with tempfile.TemporaryDirectory() as root:
        fake = lambda date: fake_release(root)
        setup_fixed_days(["2026-09-22", "2026-09-23"], root, find=fake)
        assert days_on_disk(root) == {"fixed": ["2026-09-22", "2026-09-23"], "swap": None}
        setup_fixed_days(["2026-09-23"], root, find=no_download)
        assert days_on_disk(root) == {"fixed": ["2026-09-23"], "swap": None}


def test_swap_replaces_only_the_swap_day():
    with tempfile.TemporaryDirectory() as root:
        fake = lambda date: fake_release(root)
        setup_fixed_days(["2026-09-24"], root, find=fake)
        swap_day("2026-09-20", root, find=fake)
        swap_day("2026-09-21", root, find=fake)
        assert days_on_disk(root) == {"fixed": ["2026-09-24"], "swap": "2026-09-21"}


def test_swap_to_a_day_on_disk_downloads_nothing():
    with tempfile.TemporaryDirectory() as root:
        fake = lambda date: fake_release(root)
        setup_fixed_days(["2026-09-24"], root, find=fake)
        swap_day("2026-09-20", root, find=fake)
        swap_day("2026-09-24", root, find=no_download)
        swap_day("2026-09-20", root, find=no_download)
        assert days_on_disk(root) == {"fixed": ["2026-09-24"], "swap": "2026-09-20"}


def test_unreleased_date_keeps_old_swap_day():
    def not_released(date):
        raise LookupError(f"no adsb.lol history release found for {date}")

    with tempfile.TemporaryDirectory() as root:
        swap_day("2026-09-20", root, find=lambda date: fake_release(root))
        try:
            swap_day("2026-09-19", root, find=not_released)
        except LookupError:
            pass
        assert days_on_disk(root)["swap"] == "2026-09-20"


def test_available_range():
    first, last = available_range()
    yesterday = datetime.datetime.now(datetime.timezone.utc).date() - datetime.timedelta(days=1)
    assert first == "2023-02-16"
    assert last == yesterday.isoformat()


def test_dates_outside_the_archive_are_rejected():
    for date in ("2026-13-45", "20260924", "2010-01-01", "2023-02-15", "2099-01-01"):
        try:
            swap_day(date, find=no_download)
            assert False, f"{date} should be rejected"
        except ValueError:
            pass


def test_start_swap_in_background():
    with tempfile.TemporaryDirectory() as root:
        start_swap("2026-09-20", root, find=lambda date: fake_release(root))
        status = wait_for_swap()
        assert status["state"] == "ready", status
        assert status["done_bytes"] == status["total_bytes"] > 0, status
        assert days_on_disk(root)["swap"] == "2026-09-20"


def test_one_swap_at_a_time():
    with tempfile.TemporaryDirectory() as root:
        gate = threading.Event()

        def slow_fake(date):
            gate.wait(5)
            return fake_release(root)

        start_swap("2026-09-20", root, find=slow_fake)
        try:
            start_swap("2026-09-21", root, find=slow_fake)
            assert False, "a second swap should be refused while one is running"
        except RuntimeError:
            pass
        finally:
            gate.set()
        assert wait_for_swap()["state"] == "ready"
        assert days_on_disk(root)["swap"] == "2026-09-20"


def test_failed_swap_is_reported():
    with tempfile.TemporaryDirectory() as root:
        start_swap("2026-09-20", root, find=no_download)
        status = wait_for_swap()
        assert status["state"] == "failed" and "should not have downloaded" in status["error"]


def check_live_releases():
    for date in ("2026-09-24", "2024-01-01", "2023-02-20"):
        parts = find_release(date)
        tag = parts[0][0].rsplit("/", 1)[1].split(".tar")[0]
        print(f"{date}: {tag}, {len(parts)} part(s), {sum(size for _, size in parts) / 1e9:.1f} GB")


if __name__ == "__main__":
    test_only_traces_are_kept()
    test_setup_fixed_days()
    test_swap_replaces_only_the_swap_day()
    test_swap_to_a_day_on_disk_downloads_nothing()
    test_unreleased_date_keeps_old_swap_day()
    test_available_range()
    test_dates_outside_the_archive_are_rejected()
    test_start_swap_in_background()
    test_one_swap_at_a_time()
    test_failed_swap_is_reported()
    print("offline tests passed")
    if "--live" in sys.argv:
        check_live_releases()