# Model Free
This folder contains modules to support model-free learning; in particular, R-NaD. It constitutes an architecture described [here](https://docs.google.com/document/d/14menCHw8z06KJWZ5F_K-MjgWVo_b7PESR7RlG-em4ic/edit?tab=t.0#heading=h.235tkdxlbz3z).

**ModelFreeActor**: This modules constitutes an actor which intakes a policy and acts on it. It then uses the battle finished callback to save data to a folder, which is just a `ReplayBuffer`. These actors can inherit any policy, so as R-NaD develops, it will intake different policies.

**ModelFreeLearner**: This module pulls in data from our `ReplayBuffer` and updates its understanding of the optimal policy. It can then save the policies to a folder system where they can be subsequently loaded by `ModelFreeActor`'s.
