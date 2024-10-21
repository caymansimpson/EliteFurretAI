# Utils

TODO: update how it works and what it covers

This folder contains the class that does Battle Inference for VGC. Development of this module requires a fair amount of mechanics studying, and thus is quite manual, and hard to maintain. It is WIP.

**BattleInference**: This module uses battle logs to infer various hidden information. This is WIP. It will make the following inferences:
   - Speed -- it will bound your opponent's mon's speed (including make inferences on choicescarf)
   - Attack/Defense -- it will do reverse damage calculation to infer opponent's spreads
   - Item -- it will infer opponent's items (right now: safetygoggles, heavydutyboots, lightclay and covertcloak) and keep track of whether choice/assaultvest are possible. It does not yet keep track of every item possibility (but can be easily extended to do so).

**SpeedInference**: This module infers an opponent's speed. Using it's undertanding of game mechanics and Showdown message protocol, it assembles orders of logs, creating a Linear Programming problem to solve for opponent's speeds. It will also infer choicescarf. Note of an edge-case: it will not infer choicescarf if a mon's speed is in the realm of possibility (e.g. the unlikely scenario where a mon has no speed investment and has choicescarf, such that a jolly + 252 speed EV has the same speed). It will correctly identify it though if the mon demonstrates it's not scarf.

**ItemInference**: This module infers an opponent's items (at least most). Using it's undertanding of game mechanics and Showdown message protocol, it infers based on what _didn't_ happen, what item a pokemon may have (since Showdown already exposes when something does happen, what item caused that event). Right now, it supports inferring lightclay, heavydutyboots, safetygoggles and covertcloak. It will also track whether assaultvest and choice items are plausible.

NOTES:
- All inference items are tested on doubles only, and some methods do have pity singles implementations, but they are untested -- they're neither robust nor thorough!
- ItemInference and SpeedInference both independently track battle states because it is necessary to infer speed and items. This means that they triple the memory needed to play a battle! Just beware of this fact.
- Each inference will assume a mon has an advantageous ability if that ability isn't known. For example, we will always assume Venasaur has Chlorophyll if it is in the sun.

**InfostateNetwork** -- a module that predicts the likelihood of opponent infostates. We will be using opponent’s past actions (e.g. our PBS/history) to infer infostates using our value network when created. Because Pokemon is a two-player zerosum game, our oppponent's predicted value can be trivially inferred by our predicted value -- they sum to 0. We can calculate the probability of an infostate based on the probability an opponent was likely to make previous moves (derived from the value they'd get from those moves). As an example, if I'm going to hydro pump an Incineroar, they probably let the Incineroar take it if the only mon they in back is a Heatran. This method is superior to MetaDB's because it can make inferences using opponent's actions and make guesses on unseen data. This has yet to be implemented.
