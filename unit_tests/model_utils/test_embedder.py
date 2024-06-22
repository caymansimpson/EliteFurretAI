# -*- coding: utf-8 -*-
from poke_env.data import GenData
from poke_env.environment import Move, Pokemon

from elitefurretai.model_utils.embedder import Embedder


def move_generator(gen=9):
    for move in GenData.from_gen(gen).moves:
        yield Move(move, gen=gen)
        yield Move("z" + move, gen=gen)


def mon_generator(gen=9):
    for species in GenData.from_gen(gen).pokedex:
        if species != "missingno":
            yield Pokemon(gen=gen, species=species)


# TODO: test implementation on one move
def test_embed_move():
    embedder = Embedder()

    # Test that every move has the same length, and that each move is different
    none_move = embedder.embed_move(None)
    moves = set()
    moves.add(tuple(none_move))
    for move in move_generator():
        embedded_move = embedder.embed_move(move)
        assert len(embedded_move) == len(none_move)
        assert tuple(embedded_move) not in moves
        moves.add(tuple(embedded_move))


# TODO: test implementation on one pokemon; from battle
def test_embed_pokemon():
    embedder = Embedder()

    # Test that every mon has the same length, and that each mon is different
    none_mon = embedder.embed_pokemon(None)
    mons = set()
    mons.add(tuple(none_mon))
    for mon in mon_generator():
        embedded_mon = embedder.embed_pokemon(mon)
        assert len(embedded_mon) == len(none_mon)
        assert tuple(embedded_mon) not in mons
        mons.add(tuple(embedded_mon))


# TODO: test implementation on one opponent pokemon; from battle
def test_embed_oponent_pokemon():
    embedder = Embedder()

    # Test that every mon has the same length, and that each mon is different
    none_mon = embedder.embed_pokemon(None)
    mons = set()
    mons.add(tuple(none_mon))
    for mon in mon_generator():
        embedded_mon = embedder.embed_opponent_pokemon(mon)
        assert len(embedded_mon) == len(none_mon)
        assert tuple(embedded_mon) not in mons
        mons.add(tuple(embedded_mon))


# TODO: Recreate battle, and then add a request
def test_embed_team_preview(double_battle_json):
    raise NotImplementedError


# TODO: Recreate battle, and then add a request
def test_embed_turn(double_battle_json):
    raise NotImplementedError
