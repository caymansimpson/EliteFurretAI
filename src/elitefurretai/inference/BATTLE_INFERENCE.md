# Utils

This folder contains the classes that do Battle Inference for VGC -- their goal is to leverage knowledge of game mechanics to supplement AI's knowledge of the game at hand. It is a side project; since this module requires a fair amount of mechanics knowledge, it's quite manual and hard to maintain. If you plan to use these, I would encourage you to be cautious read this document in its entirety.

This module has gone through extensive Fuzz Testing, and so should work for most use-cases. However, it has many known bugs (noted below). I will not personally work on any issues related to the Inference classes I've implemented, but will take PR's if you want to fix them.

## **BattleInference**
This module stores inferences made by ItemInference and SpeedInference, and can be sent through the Embedder class to ameliorate an agent's understanding of the battle state. It is a glorified dicionary that has peresistent memory across both `SpeedInference` and `ItemInference` modules.

Ideally, we would combine `BattleInference`, `SpeedInference` and `ItemInference` into one module so we only store one battle state and go through logs once, but the complexity of such a task would make these modules even more difficult to maintain.

#### NOTES:
- All inference items are tested on doubles only, and some methods do have pity singles implementations, but they are untested!
- ItemInference and SpeedInference both independently track battle states because it is necessary to track battle states mid-turn to properly infer speed and items. This means that they triple the memory needed to play a battle! Just beware osf this fact.
- Each inference will assume a mon has an advantageous ability if that ability isn't known. For example, we will always assume Venasaur has Chlorophyll if it is in the sun.

## SpeedInference
This module infers an opponent's speed. Using it's undertanding of game mechanics and Showdown's message protocol, it:
1. Assembles orders of relevant events using encoded knowledge of (Gen 9) game mechanics
2. Uses those order of relevant to create a Linear Programming problem to solve for opponent's speeds. We need to use a LP because there are interactions with two unknown variables in VGC (your opponent's mons).
3. It will also infer choicescarf if the LP isn't solvable until we assume that an opponent mon can have 1.5x speed it normally does and it only has used one move while being out.

There are many edge-cases SpeedInference won't catch. Some examples:
- Iron Ball or Lagging Tail affecting move orders
- It won't infer choice scarf if we speedtie and always lose
- It will misinterpret speeds if your opponent has a mon that could possibly have a speed-based ability (e.g. Basculeigon's Swift Swim) in the appropriate conditions (e.g. Rain) but doesn't actually (e.g. instead has Adaptability).
- It doesnt take into account "intuition", so if a urushifu looks faster than a chien-pao, it will assume the chien-pao is slow instead of the urushifu being choicescarf (it will always try to solve the problem without a choicescarf), until it can confirm speeds with our own mons; in this way, **it will sometimes make the wrong inferences.**

## ItemInference
This module infers opponents' most common and inferrable items using it's undertanding of game mechanics and Showdown message protocol. Importantly, poke-env already updates with explicit mentions of items via Showdown protocol (e.g. via Frisk or Trick); this module infers items based on what _didn't_ happen. Right now, it supports inferring:
1. **Light Clay**: If a mon's screens last for more than 5 turns
2. **Heavy Duty Boots**: If a mon enters the battle with hazards and is unaffected, but should have been.
3. **Safety Goggles**: If a mon should have been affected by Rage Powder or Sandstorm but wasn't.
4. **Covert Cloak**: If a mon should have been affected by a move's secondary ability (with 100% chance of activating, like Salt Cure) but wasn't.
5. **Clear Amulet**: If a mon should have been silently affected by an Intimidate or a move's secondary ability (with 100% chance of hitting that affects boosts), but didn't.
6. **Choice Tracking**: Tracks whether a mon could be choice based on the moves it's used.
7. **Assault Vest Tracking**: Trakvs whether a mon used a Status move, which would invalidate whether it has an Assault Vest. This can be ameliorated when we have Damage Calculation implemented.

## DamageInference
This module will infer an opponent's item/EV spread. Right now, I have not prioritized its development.


## Usage:
```python
class InferencePlayer(RandomPlayer):

   def __init__(self, *):
      super().__init__(*)

      # Where I will store Inference objects (per battle)
      self._inferences = {}
      self._speed_inferences = {}
      self._item_inferences = {}
      self._damage_inferences = {}

   def teampreview(self, battle):

      # Initialize main Inference storage and store in Player
      inferences = BattleInference(battle)
      self._inferences[battle.battle_tag] = inferences

      # Initialize Speed and Item Inferences
      si = SpeedInference(battle, inferences)
      ii = ItemInference(battle, inferences)
      di = DamageInference(battle, inferences)

      # Store them in Player
      self._speed_inferences[battle.battle_tag] = si
      self._item_inferences[battle.battle_tag] = ii
      self._damage_inferences[battle.battle_tag] = di

      # Update them with what we've seen so far
      self._speed_inferences[battle.battle_tag].update(battle)
      self._item_inferences[battle.battle_tag].update(battle)
      self._damage_inferences[battle.battle_tag].update(battle)

      # Whatever algo to choose teampreview; hardcoded in this example
      return "/team 1234"

    def choose_move(self, battle):

      # Force Switch is not implemented yet due to a poke-env bug
      if not any(battle.force_switch):

         # Update Inferences
         self._speed_inferences[battle.battle_tag].update(battle)
         self._item_inferences[battle.battle_tag].update(battle)
         self._damage_inferences[battle.battle_tag].update(battle)

      # Whatever algo to choose a move; random in this example
      return self.choose_random_doubles_move(battle)
```
