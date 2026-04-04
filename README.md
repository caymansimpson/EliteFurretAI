# EliteFurretAI
**The goal of this project is to build a superhuman bot to play Pokemon VGC**. It is not to further research, nor is it to build a theoretically sound approach -- the goal is to be the best that no one ever was. We will only contribute to research or take sound approaches if it will help us towards our ultimate goal.

Table of Contents:
1. [Goals & Priorities](#goals-and-priorities)
2. [Summary of the VGC Problem Space](#summary-of-the-vgc-problem-space)
3. [Current Proposed Approach](#current-proposed-approach)
4. [Where I am noq](#where-im-now)
5. [Why the name?](#why-the-name-elitefurretai)
6. [Resources](#resources)
7. [Contributors & Acknowledgements](#contributors--acknowledgements)

![AI Pokeball](docs/images/aipokeball.png)

## Goals and Priorities
This project is (unnecessarily) big, and so there is a sequence of milestones:
1. **Basic Foundations**: Simple utilities extending off of poke-env to make it easier to build a VGC RL or supervised learning bot off-the-shelf for me and researchers. This includes generating, storing, inferring and embedding data.
2. **Build a VGC Bot**: We want to build a bot using the above utilities.
3. **Derive Teambuilding**: Once our bot gets to superhuman, we can use it and a sample of teams in the current Meta to derive an optimal team-building strategy via brute force.
4. **Create Furret-based teams**: With the above, we can contain our bot to force it to have and bring Furret in matchups to help derive the most optimal usage of this monster of a Pokemon. Imagine a world in which Furret dominates a VGC meta!
5. **Incorporate into games**: With a strong bot, playing Pokemon will become intensely challenging and strategic.

### Summary of the VGC Problem Space
- In the purest sense, **a VGC battle is an imperfect information zero-sum two player game with a very large simultaneous action space, complex game mechanics, a high degree of stochasticity and an effectively continuous state space**.
    - VGC is an incredibly difficult AI problem. The fact that there is a large pool of top players (and they’re hard to sort) demonstrates the difficulty of this problem even for humans.
- After reading a wide array of literature, **we suggest we should tackle VGC directly** (instead of through Singles) because of the 40x action space, 3000x branching factor and the additional importance given to game interactions. These factors necessitate that an agent more deeply understands game mechanics and be more computationally efficient.
- Given these properties of VGC and top existing bots, **we will attempt to use a model-based search algorithm with depth-limited + heavily pruned search and a Nash Equilibrium-based value function that does not assume unobservable information**. We plan to initialize our agent with human data and train using self-play.
    - There is still quite a lot we need to understand about specifically how VGC behaves in order to make more informed algorithmic choices, and so this approach is very likely to change as we learn more.
- Industry’s dominance in making State of the Art agents demonstrates that **with enough talent, capacity and infrastructure, virtually all problems with VGC’s nature can be solved**. However, assessing the current state of resources available to us, the current bottlenecks for developing a successful agent is (in order):
    - **Talent** – Very few agents have seen dedicated and organized support over a span more than 12 months; having a dedicated and organized team is crucial (and people who are better at this than me).
    - **Engine** – Faster pokemon engine with ability to simulate (where we can control RNG). This is already accomplished by [pmarglia](https://github.com/pmariglia/foul-play) for Singles with great results.
    - **Capacity** – CPU for generating training data, GPU for inference
    - **Human Training Data** – while not essential, this will accelerate training convergence by orders of magnitude, reduce capacity needs and accelerate our own internal learning speed tremendously. It will also help our bot transition to playing humans more easily.

## Current Proposed Approach
From our synthesis of [available literature](https://docs.google.com/document/d/14menCHw8z06KJWZ5F_K-MjgWVo_b7PESR7RlG-em4ic/edit#heading=h.p6dz1cv0mnpx), we’ve gleaned:
- Model-free alone is unlikely to produce superhuman performance without the capacity that we don’t have available
- Search is necessary for decision-time planning, and game abstractions are necessary to make search tractable
- The behavior of VGC from a game-theoretic perspective is still unknown, and theory might not help the practical purposes of making a superhuman bot.

We feel the best approach will likely be both of:
- **Policy-based** – based on Nash Equilibrium using Deep Learning to create the best policy/value networks that generalize to the game well. This allows for most flexibility for decision-time planning. These will likely have to be from a combination of classic self-play RL and imitation learning.
- **Search-based** – during decision-time planning, we should expore MCTS guided by our Policy and Value networks. This allows us to better deal with nuances of game mechanics that RL might not be able to fully grasp. We can use different types of game abstractions to speed up this process and make it more tractable. This will unequivocally be critical given the game mechanic complexity and high cost of mistakes in VGC; RL with our current resources will unlikely be sufficient.

Ultimately, we think that Search-based will be the quickest way to get to peak human levels, and that policy-based (or methods that combine policies with search) will get to superhuman performance. We see this playing out with Foul Play

There is quite a lot of complexity in the above, and we encourage you to check out [the doc linked above](https://docs.google.com/document/d/14menCHw8z06KJWZ5F_K-MjgWVo_b7PESR7RlG-em4ic/edit#heading=h.p6dz1cv0mnpx) to learn more.

### What I've Done
Currently, I've built a [supervised deep learning model](./src/elitefurretai/supervised/SUPERVISED.md) (`TransformerThreeHeadedModel`, ~125M parameters) that predicts a human's action:

*   **Overall Top-1/3/5 Action Accuracy**: 41% / 61% / 69%
*   **Move Top-1/3/5**: 26% / 51% / 62%
*   **Switch Top-1**: 99%
*   **Top-1/3/5 Teampreview Accuracy**: 54% / 82% / 99%
*   **Win Correlation**: 0.82

*Takeaway*: The transformer-based unified model coordinates joint turn actions (65% BOTH Top-3) while maintaining near-perfect switching fundamentals (99% Top-1). The distributional C51 value head provides reliable win probability estimates (0.82 correlation). The teampreview head is detached from the shared encoder to prevent harmful gradient interference.

**Primary Learnings**:
*   **Teampreview**: With the TP head detached from the shared encoder, TP accuracy is lower than earlier overfit models but the backbone produces better action/value representations. There is limited strategic variation in the dataset; for a given team composition, 88.6% of the time a player makes the same teampreview choice.
*   **Action**: Predicting the *exact* move a human makes is difficult due to playstyle variety and simultaneous decision-making. Models bias towards learning "easy" actions (switching) instead of harder actions like moves/targets. The BOTH action type (coordinating two Pokemon) at 65% Top-3 shows the model has learned meaningful joint action reasoning.
*   **Advantage**: The distributional C51 value head outperforms scalar regression. Predicting advantage over raw win probability is better due to the stochasticity of Pokemon.

### Where I'm now
I'm in the beginning phases of RL -- for now, I'm taking the following approach:
- RNaD based on the recent superhuman performance of the algorithm in Stratego, which is very similar to Pokemon
- League training: self, ghosts (past selves), exploiters, heuristics (SHP, MaxDamage), [VGCBench](https://github.com/cameronangliss/vgc-bench) and the above BC model -- to ensure our agent learns robust strategies with many teams (this will be key for learning how to play with Furret and future teambuilding applications)
- Portfolio-based regularization -- I'll be experimenting with regularizing against a portfolio of models, vs a single reference model.

I'm building an agent that will gain near-pro expertise with one team against 20+ teams with different strategies (eg rain, tailwind, sun, trickroom, etc). Then I will tackle improvements -- better information management and improved generalization (against more teams, and using more teams).

## Why the name EliteFurretAI?
As mentioned above, the penultimate goal of this work is to make Furret central to the VGC meta. Because Nintendo refuses to give Furret the Simple/Shadow Tag/Pure Power/Prankster buffs it desperately ~~needs~~ deserves, only a superhuman AI will be able to build around this dark horse to unleash its latent potential. This bot is the first step to doing so; once it can accurately value starting positions w/ decent generalizations, we can use it to start building teams.

Eventually, we hope that this AI can build and use a competitive team centered around Furret -- one that will be deserving of surpassing all Elite Fours, and even potentially replacing in-game AI -- hence the name "EliteFurret". We chose to stick with "AI" at the end of the name so players know they are being decimated by a robot that profoundly understands the capabilities of this monster.

![OG Furret](docs/images/furret.png)

## Resources
More details on this approach, thinking and understanding that led to everything in this README can be found [here](https://docs.google.com/document/d/14menCHw8z06KJWZ5F_K-MjgWVo_b7PESR7RlG-em4ic/edit).


## Contributors & Acknowledgements
It's definitely presumptuous to acknowledge people before EliteFurretAI amounts to anything, but I do have a couple of people I want to call out that have been instrumental to even getting this project off the ground.
- First and foremost, a huge shoutout to [hsahovic](https://github.com/hsahovic) both for building poke-env, but also teaching me quite a lot about how to code better
- Second, a shoutout to [attraylor](https://github.com/attraylor) who brought me into the Pokemon AI community
- Lastly, a shoutout to [pre](https://github.com/scheibo) for being the engine that keeps the community going, and inspiring in me a new round of motivation to build AI right.

![https://x.com/megapody6/status/1849056000969699480?s=46](docs/images/furret2.png)
