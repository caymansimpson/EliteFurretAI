# -*- coding: utf-8 -*-
"""This script scrapes pokepast.es and turns them into text files for elitefurretai.
It accepts a csv in the format of regulation -> pokepaste link.
"""

import csv
import os
import os.path
import re
import sys
import threading
import time
from collections import deque
from queue import Queue
from typing import Dict, List

import requests
from bs4 import BeautifulSoup
from poke_env.data.normalize import to_id_str

queue: Queue = Queue()
results: deque = deque()
lock = threading.Lock()
tried = 0


# Threaded Scraper
def scrape_html_page():

    # Declare global variables
    global tried
    global queue
    global lock
    global results

    while not queue.empty():
        url, frmt = queue.get()
        try:

            # Print out progress
            with lock:
                tried += 1
                sys.stdout.write(
                    f"\033[2K\rRead {tried} pokepastes so far... starting on {url}"
                )
                sys.stdout.flush()

            # Fetch the webpage
            response = requests.get(url)
            if response.status_code != 200:
                raise Exception(
                    f"Failed to fetch the page. Status code: {response.status_code}"
                )

            # Parse the HTML
            soup = BeautifulSoup(response.text, "html.parser")

            # Extract text from <aside> elements, where the name is in the <h1> within each <aside>
            team_name = ""
            aside = soup.find_all("aside")[0]
            h1_text = aside.find("h1").get_text(strip=True) if aside.find("h1") else ""
            team_name = to_id_str(h1_text.lower())

            # Extract text from <pre> elements. There is some variation here.
            mons = []
            for text in [pre.get_text(strip=True) for pre in soup.find_all("pre")]:

                # Add newlines where they don't exist from scraped text, and spaces after colons
                mon = text.replace("Ability:", "\nAbility: ")
                mon = mon.replace("Shiny:", "\nShiny: ")
                mon = mon.replace("Level:", "\nLevel: ")
                mon = mon.replace("EVs:", "\nEVs: ")
                mon = mon.replace("Tera Type:", "\nTera Type: ")
                mon = mon.replace("Item:", "\nItem: ")
                mon = re.sub(r"([A-Z][a-z]+) Nature", r"\n\1 Nature", mon)
                mon = mon.replace("IVs:", "\nIVs: ")

                # Temporarily remove Move names with hyphens
                mon = (
                    mon.replace("U-turn", "Uturn")
                    .replace("Will-O-Wisp", "WillOWisp")
                    .replace("Freeze-Dry", "FreezeDry")
                )
                mon = mon.replace("Self-Destruct", "SelfDestruct").replace(
                    "Double-Edge", "DoubleEdge"
                )
                mon = (
                    mon.replace("Baby-Doll Eyes", "BabyDollEyes")
                    .replace("X-Scissor", "XScissor")
                    .replace("Soft-Boiled", "SoftBoiled")
                )

                # Arbitrarily split the text into two, where the only hyphens in the second half
                # should be delineating moves
                split = mon.split("\nAbility:")

                # Repair text with moves split to each line, and restore original Move names
                mon = split[0] + "\nAbility:" + re.sub(r"\s?+- ?", "\n- ", split[1])
                mon = (
                    mon.replace("Uturn", "U-turn")
                    .replace("WillOWisp", "Will-O-Wisp")
                    .replace("FreezeDry", "Freeze-Dry")
                )
                mon = mon.replace("SelfDestruct", "Self-Destruct").replace(
                    "DoubleEdge", "Double-Edge"
                )
                mon = (
                    mon.replace("BabyDollEyes", "Baby-Doll Eyes")
                    .replace("XScissor", "X-Scissor")
                    .replace("SoftBoiled", "Soft-Boiled")
                )

                # Do some extra cleaning on nicknames, slashes, genders, items and colons; these are inconsistent
                mon = re.sub(r":\s?", ": ", mon)
                mon = re.sub(r":\s+", ": ", mon)
                mon = re.sub(r"@", " @ ", mon)
                mon = re.sub(r"\s+@\s+", " @ ", mon)
                mon = re.sub(r"/", " / ", mon)
                mon = re.sub(r"\s+/\s+", " / ", mon)
                mon = re.sub(r"\(M\)", " (M) ", mon)
                mon = re.sub(r"\s+\(M\)\s+", " (M) ", mon)
                mon = re.sub(r"\(F\)", " (F) ", mon)
                mon = re.sub(r"\s+\(F\)\s+", " (F) ", mon)

                # Add to list
                mons.append(mon)

            # Save results
            results.append(
                {
                    "team_name": team_name,
                    "mons": mons,
                    "format": frmt,
                    "url": url,
                }
            )

        except Exception as e:
            print(f"Error scraping {url}: {str(e)}")
        finally:
            queue.task_done()


def main():
    # Declare global variables
    global tried
    global queue
    global lock
    global results

    # Local filename that has regulation -> pokepaste link
    file_to_pokepastes = sys.argv[1]

    # CSV columns are letters of the regulation for VGC2024
    links: Dict[str, List[str]] = {
        "G": [],
        "H": [],
        "F": [],
        "E": [],
    }

    print("Loading data from " + file_to_pokepastes + "...")

    # Prepare variables for reading
    total = 0
    start = time.time()
    num_threads = 30

    # Load data from CSV
    with open(file_to_pokepastes, "r") as file:
        csv_reader = csv.reader(file, dialect="excel")
        for i, row in enumerate(csv_reader, 1):
            if i > 1 and row[1] != "":
                links[row[0]].append(row[1])
                total += 1

        print(f"Success! Read {total} pokepastes!")

    # Tracking progress
    print(f"Now starting to crawl all {total} of them w/ {num_threads} threads...")

    # Go through every link and queue it
    for key in links:
        for link in links[key]:
            queue.put((link, "gen9vgc2024reg" + key.lower()))

    # Start threads
    threads = []
    for _ in range(num_threads):
        t = threading.Thread(target=scrape_html_page)
        t.start()
        threads.append(t)

    # Wait for queue to empty
    queue.join()

    # Wait for all threads to finish
    for t in threads:
        t.join()

    # Prepare variables for parsing
    errors = 0
    error_msgs: Dict[str, List[str]] = {"bad_name": [], "zoroark": [], "missing_EVs": []}

    # Now process the results
    for r in results:
        team_name, mons, frmt, _ = r["team_name"], r["mons"], r["format"], r["url"]

        # Just to remove less valuable teams
        if (
            "untitled" in team_name.lower()
            or "copy" in team_name.lower()
            or len(team_name.strip()) <= 3
        ):
            errors += 1
            error_msgs["bad_name"].append(team_name)

        # AI says GTFO
        elif any(map(lambda x: "zoroark" in x.lower(), mons)):
            errors += 1
            error_msgs["zoroark"].append(team_name)

        # Not a legit team to learn from
        elif any(map(lambda x: "EVs" not in x, mons)):
            errors += 1
            error_msgs["missing_EVs"].append(team_name)

        # Good teams
        else:
            # Programatically find the appropriate directory and write teams there
            filename = os.path.join("data/teams/" + frmt + "/", team_name + ".txt")
            with open(filename, "w") as f:
                f.write("\n\n".join(mons))

    # Print out final statistics
    duration = time.time() - start
    print(
        f"Finished! Went through {tried} links, with {tried - errors} successes in {int(duration)} secs!"
    )
    print("\n============ Errors ============")
    if len(error_msgs["bad_name"]) > 0:
        print(
            f"Not saving teams that don't have legitimate team name: [{', '.join(sorted(error_msgs['bad_name']))}]\n"
        )
    if len(error_msgs["zoroark"]) > 0:
        print(
            f"I refuse to parse this team because I found zoroark: [{', '.join(sorted(error_msgs['zoroark']))}]\n"
        )
    if len(error_msgs["missing_EVs"]) > 0:
        print(
            f"Cant parse team because it has a mon without EVs: [{', '.join(sorted(error_msgs['missing_EVs']))}]\n"
        )


if __name__ == "__main__":
    main()
