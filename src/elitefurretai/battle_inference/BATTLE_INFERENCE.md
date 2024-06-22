# Utils

This folder contains the class that does Battle Inference for VGC. Development of this module requires a fair amount of mechanics studying, and thus is quite manual, and hard to maintain. It is WIP.

**BattleInference**: This module uses battle logs to infer various hidden information. This is WIP. It will make the following inferences:
   - Speed -- it will bound your opponent's mon's speed
   - Attack/Defense -- it will do reverse damage calculation to infer opponent's spreads
   - Item -- it will infer opponent's items (eg if it doesn't take sandstorm damage), if it uses a status move, if it's damage calculator are off

**SpeedInference**: This module infers an opponent's speed. Using it's undertanding of game mechanics and Showdown message protocol, it assembles orders of logs, creating a Linear Programming problem to solve for opponent's speeds. This is WIP

**InfostateNetwork** -- a module that predicts the likelihood of opponent infostates. We will be using opponent’s past actions (e.g. our PBS/history) to infer infostates using our value network when created. Because Pokemon is a two-player zerosum game, our oppponent's predicted value can be trivially inferred by our predicted value -- they sum to 0. We can calculate the probability of an infostate based on the probability an opponent was likely to make previous moves (derived from the value they'd get from those moves). As an example, if I'm going to hydro pump an Incineroar, they probably let the Incineroar take it if the only mon they in back is a Heatran. This method is superior to MetaDB's because it can make inferences using opponent's actions and make guesses on unseen data. This has yet to be implemented.
