import argparse
import datetime
import time

import requests


def dl_bitmidi(m=0, n=125000):
    """Downloads midi files from bitmidi.com by traversing the site's generic
    index from m to n."""
    url_stem = "https://bitmidi.com/uploads/"
    start = time.time()
    last_midi = m - 1
    count = 0
    for i in range(m, n):
        print(
            f"Processed Index: {i}/{n} - last MIDI file: {last_midi} - {i-last_midi} indices ago",
            end="\r",
        )
        file = str(i) + ".mid"
        url = url_stem + file
        while True:
            try:
                response = requests.get(url, timeout=5)
                break
            except requests.exceptions.Timeout:
                continue
        # If the response is 520, then the file doesn't exists
        if response.status_code == 520:
            continue
        try:
            open("data/bitmidi/" + file, "wb").write(response.content)
            last_midi = i
            count += 1
        except Exception:
            continue
    end = time.time()
    duration = str(datetime.timedelta(seconds=round(end - start)))
    print("")
    print(f"Finished Downloading {count} Midi Files from {m} to {n} in {duration}")


def get_args():
    parser = argparse.ArgumentParser(
        description="Downloads midi files from bitmidi.com"
    )
    parser.add_argument(
        "-f",
        "--from_idx",
        type=int,
        required=False,
        help="From which midi file to start",
        default=0,
    )
    parser.add_argument(
        "-t",
        "--to_idx",
        type=int,
        required=False,
        help="To which midi file to end",
        default=125000,
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    dl_bitmidi(args.from_idx, args.to_idx)
