"""elephant — tests: the Plato-based Agentic RPG.

The elephant as dungeon master: rooms have temperatures, perception
checks are rolls (two numbers show direction, three show rate of
change), the plot is a deadband, and the narration is ALWAYS a shadow
— never the terrain itself.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from elephant.plato_rpg import (
    ARCHETYPES,
    Deadband,
    PersonalElephant,
    RPGEngine,
    RPGLog,
    RPGPlayer,
    RPGWorld,
    perception_check,
    report_words,
    run_scenario,
)
from elephant.room import Message


# ---------------------------------------------------------------------- #
# Helpers                                                                #
# ---------------------------------------------------------------------- #
def _warm_sunshine(world: RPGWorld, name: str = "Sunny") -> RPGPlayer:
    """A player whose native voice is very warm — so entering a warm
    room, their ripple pushes it warmer and the roll reads warming."""
    elephant = PersonalElephant(
        name,
        vibe={"mood": 1.0, "joke_landing": 1.0, "presence": 0.8,
              "volume": 0.6, "earnestness": 0.5, "cynicism": 0.0,
              "panic": 0.0},
        charisma=0.6, acclimation_rate=0.3,
        title="the sunshine, who carries the warmth with them",
    )
    return RPGPlayer(name, archetype="comedian", world=world,
                     start="The Yard", goal="find the warmth",
                     elephant=elephant)


def _warming_tavern() -> RPGWorld:
    """A world with a warm room whose own recent history is warming
    (presence and laughter climb across its beats) and a cold yard."""
    world = RPGWorld()
    seeds = [
        Message(author="barmaid", text="The hearth is lit. Nothing special about the night yet.", ts=100),
        Message(author="sailor", text="The hearth is warm and the company is good — a kind room, a soft night.", ts=300),
        Message(author="barmaid", text="Haha! The laughter is bright and alive now — a wonderful, glowing, warm room!", ts=500, reactions={"😂": 2}),
    ]
    world.add_room("The Tap", seed_messages=seeds,
                   description="a warm taproom", hour=21.0)
    world.add_room("The Yard", description="a cold yard", hour=22.0)
    world.add_edge("The Tap", "The Yard", "the gate")
    return world


# ---------------------------------------------------------------------- #
# The perception check as a roll                                         #
# ---------------------------------------------------------------------- #
def test_entering_a_warm_room_perceives_warming():
    """A player entering a warm room runs the perception check — the
    room's direction/rate over its recent history IS what they feel."""
    world = _warming_tavern()
    player = _warm_sunshine(world)

    report = player.enter("The Tap")

    assert report.n_readings >= 3
    assert report.warmth_direction > 0, (
        f"expected warming, got direction {report.warmth_direction}")
    words = report_words(report, player.pulse.noise_floor)
    assert "warming" in words
    assert "holding" not in words
    # The report is a PerceptionReport — the pulse's roll, not a vector.
    assert hasattr(report, "whole_hand") and report.whole_hand


def test_two_numbers_show_direction_three_show_rate():
    """The room's history gives direction from the last two readings
    and rate of change from the last three+ (the macro read)."""
    world = RPGWorld()
    # Accelerating warmth: presence climbs, then laughter and volume
    # land together on the last beat — direction from the last two
    # readings, rate from the last three+.
    seeds = [
        Message(author="a", text="A quiet room.", ts=100),
        Message(author="b", text="The room is filling up — more people, more voices, still quiet.", ts=300),
        Message(author="c", text="Haha! A toast! The place is warm and alive — wonderful, bright, glowing!", ts=500, reactions={"😂": 2}),
    ]
    world.add_room("The Hearth", seed_messages=seeds, description="a room")
    world.add_room("The Door", description="outside")
    world.add_edge("The Hearth", "The Door", "door")

    player = _warm_sunshine(world)
    report = player.enter("The Hearth")

    assert report.warmth_direction > 0       # two numbers: direction
    # With three+ readings the rate is defined (a number, possibly 0).
    assert isinstance(report.warmth_rate, float)
    assert report.n_readings >= 3


def test_personal_read_filters_by_what_the_character_cares_about():
    """Each prisoner sees a different shadow: a cynicism-first brooder
    notices the cynicism climbing, not the mood."""
    world = _warming_tavern()
    brooder = RPGPlayer("Ilsa", archetype="brooder", world=world,
                        start="The Yard", goal="watch the room")
    brooder.enter("The Tap")
    # Make cynicism move in the room and re-roll.
    room = world.rooms["The Tap"]
    room.say("Ilsa", "Sure. Sure. Whatever. Obviously. Uh-huh, right, of course.", ts=600)
    room.re_read(t=600)
    report = brooder.read_room()
    read = brooder.character.personal_read(report, brooder.pulse.noise_floor)
    assert "cynicism" in read
    assert brooder.character.top_dial() == "cynicism"


# ---------------------------------------------------------------------- #
# Acting changes the room; the deadband rings; the plot advances         #
# ---------------------------------------------------------------------- #
def _cellar_world() -> RPGWorld:
    world = RPGWorld()
    world.add_room("The Cellar", seed_messages=[
        Message(author="keeper", text="The cellar is quiet tonight. Dust and barrels, nothing more.", ts=100),
    ], description="a dusty cellar", deadband=Deadband({"panic": 0.3}))
    world.add_room("The Hall", description="the hall above")
    world.add_edge("The Cellar", "The Hall", "stairs")
    return world


def test_fight_action_spikes_panic_and_rings_the_deadband():
    """A fight action spikes panic; the room's field crosses its
    deadband; the ring advances the plot with a GM line."""
    world = _cellar_world()
    player = RPGPlayer("Marnie", archetype="comedian", world=world,
                       start="The Hall", goal="start a fight")
    player.enter("The Cellar")
    panic_before = world.rooms["The Cellar"].effective_field().readings["panic"]

    line = player.act("fight")
    room = world.rooms["The Cellar"]
    panic_after = room.effective_field().readings["panic"]

    assert "fire" in line.lower() or "run" in line.lower()   # the words do it
    assert panic_after > panic_before, "the fight must spike panic"
    assert panic_after >= 0.3

    # The full loop on a fresh cellar: the ring advances the plot.
    world2 = _cellar_world()
    player2 = RPGPlayer("Marnie", archetype="comedian", world=world2,
                        start="The Hall", goal="fight")
    engine = RPGEngine(world=world2, players=[player2], goal="survive the night")
    engine.plot_lines = ["GM beat one: the cellar answers."]
    engine.script = [(1, "Marnie", "move", "The Cellar"),
                     (2, "Marnie", "fight", "The Cellar")]
    log = engine.run(max_turns=3)
    assert engine.plot_stage >= 1
    assert any("GM beat one" in line for line in log.lines)
    assert any("⚡" in line for line in log.lines)


def test_joke_warming_changes_the_room():
    """A joke that lands warms the room — acting changes the terrain
    the players share."""
    world = RPGWorld()
    world.add_room("The Cellar", seed_messages=[
        Message(author="keeper", text="The cellar is quiet tonight. Dust and barrels, nothing more.", ts=100),
    ], description="a dusty cellar")
    world.add_room("The Hall", description="the hall above")
    world.add_edge("The Cellar", "The Hall", "stairs")
    player = _warm_sunshine(world)
    player.enter("The Cellar")
    before = world.rooms["The Cellar"].effective_field().warmth()
    player.act("joke")
    after = world.rooms["The Cellar"].effective_field().warmth()
    assert after > before, "a landing joke must warm the room"
    assert world.rooms["The Cellar"].effective_field().readings["joke_landing"] > 0.5


# ---------------------------------------------------------------------- #
# The engine runs a scenario to completion                               #
# ---------------------------------------------------------------------- #
def _mini_scenario() -> dict:
    return {
        "name": "The Hollow",
        "premise": "A small dungeon with one true room and a goal.",
        "goal": "light the brazier",
        "goal_room": "The Hollow",
        "max_turns": 5,
        "rooms": [
            {"name": "The Hollow",
             "description": "a round stone room with a cold brazier",
             "seed_messages": [
                 ("keeper", "The brazier has been cold for a hundred years.", 100),
             ],
             "deadband": {"panic": 0.4}},
            {"name": "The Threshold",
             "description": "a doorway with a draft",
             "seed_messages": [("keeper", "Mind the step.", 100)]},
        ],
        "edges": [("The Hollow", "The Threshold", "the door")],
        "players": [
            {"name": "Marnie", "archetype": "comedian", "start": "The Threshold",
             "goal": "light the brazier"},
            {"name": "Theo", "archetype": "wallflower", "start": "The Threshold",
             "goal": "not be seen doing it"},
        ],
        "script": [
            (1, "Marnie", "move", "The Hollow"),
            (1, "Theo", "move", "The Hollow"),
            (2, "Marnie", "fight", "The Hollow"),
            (2, "Theo", "investigate", "The Hollow"),
            (3, "Marnie", "resolve", "The Hollow"),
            (3, "Theo", "wait", "The Hollow"),
        ],
        "plot_lines": ["The brazier remembers the flame.",],
    }


def test_engine_runs_scenario_to_completion():
    """The engine runs a full scenario to completion — no infinite
    loops, terminates at max_turns, players land where the script puts
    them, and the goal resolves."""
    data = _mini_scenario()
    log = run_scenario(data)

    assert isinstance(log, RPGLog)
    assert log.turns <= 5
    assert len(log.lines) > 10
    assert log.goal_reached
    assert log.plot_stage >= 1
    positions = {p.name: p.position for p in log.players}
    assert positions["Marnie"] == "The Hollow"
    assert positions["Theo"] == "The Hollow"
    # The transcript reads like a session: narration lines exist.
    assert any(line.startswith("GM:") for line in log.lines)


def test_engine_never_runs_forever():
    """Even a scenario with no goal resolution terminates at max_turns."""
    data = _mini_scenario()
    data["goal_room"] = "The Threshold"   # no resolve ever happens there
    data["script"] = [
        (1, "Marnie", "move", "The Hollow"),
        (2, "Marnie", "wait", "The Hollow"),
        (3, "Marnie", "wait", "The Hollow"),
    ]
    log = run_scenario(data, max_turns=4)
    assert log.turns == 4
    assert not log.goal_reached
    assert "The night ends" in "\n".join(log.lines)


# ---------------------------------------------------------------------- #
# The narration is always a shadow                                       #
# ---------------------------------------------------------------------- #
def test_narration_is_shadow_never_terrain():
    """The narration is the elephant's tint + the perception reports in
    words — never the raw vectors."""
    log = run_scenario(_mini_scenario())
    for line in log.lines:
        assert "array(" not in line, line
        assert "np." not in line, line
        assert "[0." not in line, line
        # No raw dict dumps of the field.
        assert not line.startswith("{'mood'"), line

    # The tinted description contains the base text, dressed by the
    # room's field — the cave wall, not the cave.
    shadow = log.world.describe("The Hollow")
    assert "cold brazier" in shadow           # the base description is inside
    assert "night" in shadow or "light" in shadow or "lamps" in shadow


def test_enter_report_is_a_perception_not_a_vector():
    world = _warming_tavern()
    player = _warm_sunshine(world)
    report = player.enter("The Tap")
    assert report.agent_id == player.name
    assert report.warmth_direction > 0
    # The words of the report — shadow language, not numbers.
    words = report_words(report, player.pulse.noise_floor)
    assert "warm" in words or "cool" in words or "hold" in words
    assert "0.0" not in words.split("—")[0]


# ---------------------------------------------------------------------- #
# World, map, and character sheets                                       #
# ---------------------------------------------------------------------- #
def test_world_map_paths_and_neighbors():
    world = _warming_tavern()
    world.add_room("The Dock", description="planks and fog")
    world.add_edge("The Yard", "The Dock", "the pier")
    assert [e for e, _ in world.neighbors("The Tap")] == ["the gate"]
    assert world.path("The Tap", "The Dock") == ["The Tap", "The Yard", "The Dock"]
    assert world.path("The Dock", "The Dock") == ["The Dock"]
    assert world.path("The Dock", "Nowhere") == []


def test_character_sheet():
    world = _warming_tavern()
    player = _warm_sunshine(world)
    player.enter("The Tap")
    player.act("joke")
    sheet = player.character_sheet()
    assert sheet["name"] == "Sunny"
    assert sheet["position"] == "The Tap"
    assert sheet["elephant"]["charisma"] == pytest.approx(0.6)
    assert sheet["pulses"] >= 1
    assert any(v == "joke" for _t, v, _tg in sheet["acts"])
    assert "last_perception" in sheet


def test_archetype_presets_exist_for_the_demo_cast():
    for key in ("comedian", "brooder", "wallflower"):
        assert key in ARCHETYPES
        for verb in ("joke", "investigate", "comfort", "fight",
                     "wait", "resolve", "banter"):
            assert ARCHETYPES[key]["lines"].get(verb), (key, verb)


def test_player_wraps_a_learned_avatar():
    """The avatar seam: a Tap-raised round character (elephant/avatar.py)
    speaks and monologues for the player; the game mechanics (the
    perception roll, the ripple) keep running on its personal elephant."""
    from elephant.avatar import Avatar

    world = _warming_tavern()
    avatar = Avatar(
        "Marnie",
        "I am Marnie, and I laugh at funerals, meaning it as a kindness.",
        preset="comedian")
    player = RPGPlayer("Marnie", archetype="comedian", world=world,
                       start="The Yard", goal="find the warmth", avatar=avatar)

    report = player.enter("The Tap")
    assert report.warmth_direction > 0      # mechanics still roll

    line = player.act("joke")
    assert "Marnie" in line                 # the round character's own voice
    assert player.character.top_dial() == "joke_landing" or \
        player.character.top_dial() in avatar.dial_weights

    monologue = player.perceive()
    assert "Marnie" in monologue or "ear" in monologue

    sheet = player.character_sheet()
    assert sheet["avatar"]["nights_at_the_tap"] == 0
    assert "through_line" in sheet["avatar"]
