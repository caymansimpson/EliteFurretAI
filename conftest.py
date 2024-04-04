# -*- coding: utf-8 -*-
import os

import orjson
from pytest import fixture

FIXTURE_DIR = os.path.join("data/fixture")


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
def example_singles_request():
    with open(os.path.join(FIXTURE_DIR, "example_singles_request.json")) as f:
        return orjson.loads(f.read())
