import argparse
import datetime
import os
import sys
import time

import requests


def dl_bitmidi(start=0, finish=125000, cache=True, early_stop=True, extend=False):
    """Downloads midi files from bitmidi.com by traversing the site's generic
    index from m to n.

    Args:
        m (int, optional): Starting index. Defaults to 0.
        n (int, optional): End index. Defaults to 125000.
        cache (bool, optional): Cache is used to avoid downloading files that
        have already been downloaded by scanning the corresponding directory.
        Defaults to True.
        early_stop (bool, optional): Early stopping means that the function
        will stop when no new midis are found for 2500 indices. Defaults to
        True.
        extend (bool, optional): Extend means that the function will extend the
        search span (n) by 2500 indices if a new midi was found within the last
        2500 indices (=sets early_stop to True).
        Defaults to False.
    """
    STOP_AFTER = 2500
    url_stem = "https://bitmidi.com/uploads/"
    start_time = time.time()
    if extend:
        print("Extend is set to True. Early stop will be set to True.")
        early_stop = True

    if cache:
        print("Using cached files")
        start = max(
            max([int(i.split(".")[0]) for i in os.listdir("data/bitmidi")]) + 1, start
        )

    last_midi = start - 1
    m = start
    count = 0

    while m <= finish:
        no_midi = m - last_midi - 1
        file = str(m) + ".mid"
        url = url_stem + file

        print(
            f"Processed Index: {m}/{finish} - last MIDI file: {last_midi} "
            f"- {no_midi} indices ago - total: {count}      ",
            end="\r",
        )

        # early stop
        if early_stop and no_midi >= STOP_AFTER:
            print("\nEarly stop")
            finish = m
            break
        # Bypass server if request returns an exception, like Timeout or
        # ConnectionError.
        while True:
            try:
                response = requests.get(url, timeout=5)
                break
            except requests.exceptions.Timeout:
                continue
            except requests.exceptions.ConnectionError:
                continue

        # If the response is 520, then the file doesn't exists
        if response.status_code != 520:
            open("data/bitmidi/" + file, "wb").write(response.content)
            last_midi = m
            count += 1

        if extend & (m == finish):
            finish += 100

        m += 1

    end_time = time.time()
    duration = str(datetime.timedelta(seconds=round(end_time - start_time)))
    print(
        f"\nFinished Downloading {count} Midi Files from "
        f"Index {start} to {finish} ({finish-start}) in {duration}"
    )


def get_args():
    """Gets arguments from the command line."""
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
    parser.add_argument(
        "-c",
        "--cache",
        action="store_true",
        help="Use cached files",
        default=True,
    )
    parser.add_argument(
        "-es",
        "--early_stop",
        action="store_true",
        help="Stop early when no new files are found for 2500 indices",
        default=True,
    )
    parser.add_argument(
        "-e",
        "--extend",
        action="store_true",
        help="Extend the search span by 2500 indices if a \
            new midi was found within the last 2500 indices",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    try:
        dl_bitmidi(args.from_idx, args.to_idx, args.cache, args.early_stop, args.extend)
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt")
        sys.exit(0)
