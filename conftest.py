# -*- coding: utf-8 -*-
import os

import orjson
from pytest import fixture

FIXTURE_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data/fixture")


@fixture
def single_battle_json_anon():
    with open(os.path.join(FIXTURE_DIR, "gen7anythinggoesanon.json")) as f:
        return orjson.loads(f.read())


@fixture
def single_battle_json():
    with open(os.path.join(FIXTURE_DIR, "gen9randombattle-222.log.json")) as f:
        return orjson.loads(f.read())


@fixture
def double_battle_json():
    with open(os.path.join(FIXTURE_DIR, "gen6doublesou-1.log.json")) as f:
        return orjson.loads(f.read())


@fixture
def example_doubles_request():
    with open(os.path.join(FIXTURE_DIR, "example_doubles_request.json")) as f:
        return orjson.loads(f.read())


@fixture
def example_vgc_teampreview_request():
    with open(os.path.join(FIXTURE_DIR, "example_vgc_teampreview_request.json")) as f:
        return orjson.loads(f.read())


@fixture
def example_singles_request():
    with open(os.path.join(FIXTURE_DIR, "example_singles_request.json")) as f:
        return orjson.loads(f.read())


def clean_logs(txt):
    return txt.replace('"', "").replace("'", '"').replace("\\", "")


def read_logs(file_text):
    events = []
    for line in file_text.split("\n"):
        if len(line.strip()) == 0:
            continue
        events.append(orjson.loads(clean_logs(line)))
    return events


@fixture
def speed_logs():
    with open(os.path.join(FIXTURE_DIR, "speed_logs.txt")) as f:
        return read_logs(f.read())


@fixture
def residual_logs():
    with open(os.path.join(FIXTURE_DIR, "residual_logs.txt")) as f:
        return read_logs(f.read())


@fixture
def edgecase_logs():
    with open(os.path.join(FIXTURE_DIR, "edgecase_logs.txt")) as f:
        return read_logs(f.read())


@fixture
def uturn_logs():
    with open(os.path.join(FIXTURE_DIR, "uturn_logs.txt")) as f:
        return read_logs(f.read())


@fixture
def vgc_battle_p1_logs():
    with open(os.path.join(FIXTURE_DIR, "vgc_battle_p1_logs.txt")) as f:
        return read_logs(f.read())


@fixture
def vgc_battle_p2_logs():
    with open(os.path.join(FIXTURE_DIR, "vgc_battle_p2_logs.txt")) as f:
        return read_logs(f.read())


@fixture
def vgc_battle_team():
    with open(os.path.join(FIXTURE_DIR, "vgc_battle_team.txt")) as f:
        return f.read()
