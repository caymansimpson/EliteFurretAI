# Supervised Models
Here, we train and test our supervised models (Stage II), documented in the original [README](../../README.md)

We are going to build three networks, to:
1. Power a FutureSightAI (Stage III in the original [README](../../README.md))
2. Help our future AI perform against humans by integrating human-like behaviors in our self-play
3. Help our future AI perform by pruning unlikely actions; this will both speed up training convergence and decision-time planning

Details on each of the models below

### **Information Network**
Infostate + history → E[Worldstate]
- This is most important as if this can get accuracy of up to 95%+, we can transition from imperfect to perfect information games and save computation.
- There is a dependency on game engine utilities
- Needs to optimize for probability of information
- Should also be evaluated by how effective it is based on the # of potential histories one can have, and how many it can eliminate

### **“Value” Network**

Infostate + E[Worldstate] → P(win)
- The higher the accuracy here, the less we need to search, meaning we will face less computational constraints
- This is not a truly valuable network because it isn’t sound (based on average policy across players), but we can try it for shits and giggles and see if it works like it did in FutreSightAI
- Note: this has a dependency on the Information Network accuracy


### **Human Policy Network**
PBS + E[Worldstate] → P(action) over all opponent legal actions
- The higher the accuracy here, the less we need to search because we can eliminate unlikely game-tree paths, meaning we will face less computational constraints
- Note: this has a dependency on the Information Network accuracy
