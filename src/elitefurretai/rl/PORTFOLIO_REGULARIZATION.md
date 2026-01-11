# Portfolio Regularization in RNaD: A Comprehensive Guide

This document provides an in-depth explanation of portfolio regularization for the RNaD (Regularized Nash Dynamics) algorithm. It's written for someone new to reinforcement learning, so we'll build up from first principles.

## Table of Contents

1. [Prerequisites: RL Basics You Need to Know](#1-prerequisites-rl-basics-you-need-to-know)
2. [The Problem: Why Do We Need Regularization?](#2-the-problem-why-do-we-need-regularization)
3. [Standard RNaD: Single Reference Model](#3-standard-rnad-single-reference-model)
4. [Portfolio Regularization: The Core Idea](#4-portfolio-regularization-the-core-idea)
5. [Implementation Approaches](#5-implementation-approaches)
6. [Theoretical Analysis: Will This Converge?](#6-theoretical-analysis-will-this-converge)
7. [Practical Considerations for VGC](#7-practical-considerations-for-vgc)
8. [Implementation Recommendations](#8-implementation-recommendations)
9. [Monitoring and Debugging](#9-monitoring-and-debugging)
10. [References](#10-references)

---

## 1. Prerequisites: RL Basics You Need to Know

Before diving into portfolio regularization, let's establish some foundational concepts.

### 1.1 What is a Policy?

A **policy** (denoted $\pi$) is simply a function that tells an agent what action to take in any given situation. In Pokémon VGC:

- **Input**: The current battle state (your team's HP, opponent's visible Pokémon, field conditions, etc.)
- **Output**: A probability distribution over possible actions (which move to use, whether to switch, etc.)

```python
# Conceptually, a policy looks like this:
def policy(battle_state):
    # Neural network magic happens here
    action_probabilities = model(battle_state)
    # Returns something like: {"Thunderbolt": 0.7, "Protect": 0.2, "Switch Incineroar": 0.1}
    return action_probabilities
```

### 1.2 What is a Probability Distribution?

When we say our policy outputs a "probability distribution," we mean it assigns a probability to each possible action, and all probabilities sum to 1.

**Example**: In a situation with 4 possible actions:
- Thunderbolt: 60% (0.6)
- Protect: 25% (0.25)
- Ice Beam: 10% (0.10)
- Switch: 5% (0.05)
- **Total**: 100% (1.0)

The policy is **probabilistic** because it doesn't always pick the highest-probability action. Instead, it samples from this distribution. This is important for **exploration** — trying new things to discover better strategies.

### 1.3 What is KL Divergence?

**KL Divergence** (Kullback-Leibler Divergence) is a way to measure how "different" two probability distributions are. Think of it as a "distance" between two policies (though technically it's not a true distance because it's asymmetric).

**Intuition**: If you have two policies:
- Policy A thinks Thunderbolt is 60% likely to be best
- Policy B thinks Thunderbolt is 10% likely to be best

These policies "disagree" a lot, so the KL divergence between them is high.

**Formula**:
$$D_{KL}(P \| Q) = \sum_x P(x) \cdot \log\left(\frac{P(x)}{Q(x)}\right)$$

Don't worry too much about the formula. The key intuition is:
- **KL = 0**: The two distributions are identical
- **KL is small**: The distributions are similar
- **KL is large**: The distributions are very different

**Important**: KL divergence is **asymmetric**: $D_{KL}(P \| Q) \neq D_{KL}(Q \| P)$

```python
import torch
import torch.nn.functional as F

def kl_divergence(p_logits, q_logits):
    """
    Compute KL divergence between two policies.

    Args:
        p_logits: Log-probabilities from policy P (the one we're measuring FROM)
        q_logits: Log-probabilities from policy Q (the one we're measuring TO)

    Returns:
        KL divergence value (scalar)
    """
    p_probs = F.softmax(p_logits, dim=-1)
    q_log_probs = F.log_softmax(q_logits, dim=-1)
    p_log_probs = F.log_softmax(p_logits, dim=-1)

    # KL = sum of p(x) * [log(p(x)) - log(q(x))]
    kl = (p_probs * (p_log_probs - q_log_probs)).sum(dim=-1)
    return kl.mean()
```

### 1.4 What is a Loss Function?

A **loss function** measures how "bad" our model is doing. During training, we try to **minimize** the loss. Lower loss = better model.

**Examples**:
- **Supervised learning**: Loss = how different our predictions are from the correct answers
- **RL (policy gradient)**: Loss = negative of expected reward (we minimize negative reward, which maximizes positive reward)

### 1.5 What is Regularization?

**Regularization** is a technique to prevent a model from doing something "too extreme." It adds a penalty term to the loss function.

**Analogy**: Imagine you're learning to play tennis. Without regularization, you might develop a very weird, specialized technique that works great against your practice partner but fails against anyone else. Regularization is like a coach saying "keep your fundamentals sound" — it prevents over-specialization.

In math:
$$\text{Total Loss} = \text{Main Loss} + \lambda \cdot \text{Regularization Penalty}$$

Where $\lambda$ (lambda) controls how strong the regularization is.

### 1.6 What is Nash Equilibrium?

In game theory, a **Nash Equilibrium** is a situation where no player can improve their outcome by changing their strategy alone. Both players are playing the "best response" to each other.

**Example**: In Rock-Paper-Scissors, the Nash Equilibrium is to play each option with 33% probability. If you deviate (say, playing Rock 50% of the time), your opponent can exploit you by playing Paper more often.

**For Pokémon VGC**: A Nash Equilibrium policy would be one that cannot be exploited by any opponent strategy. This is the "theoretically perfect" way to play.

---

## 2. The Problem: Why Do We Need Regularization?

### 2.1 The Naive Approach: Pure Self-Play

The simplest RL training approach is **self-play**: have the agent play against itself and learn from the outcomes.

```
Agent v1.0 plays against Agent v1.0
    → Agent learns, becomes v1.1
Agent v1.1 plays against Agent v1.1
    → Agent learns, becomes v1.2
... and so on
```

**The Problem**: This can lead to **unstable training** and **cyclic behavior**.

### 2.2 Cyclic Behavior Explained

Imagine your agent discovers three strategies:
- **Strategy A**: Heavy offense (always attack)
- **Strategy B**: Defensive stall (use Protect, recover HP)
- **Strategy C**: Prediction-based (read opponent's moves and counter)

These might form a "rock-paper-scissors" dynamic:
- A beats C (offense overwhelms prediction)
- B beats A (defense outlasts offense)
- C beats B (predictions beat predictable defense)

During self-play:
1. Agent learns Strategy A
2. Agent learns Strategy B to beat A
3. Agent learns Strategy C to beat B
4. Agent learns Strategy A to beat C (back to start!)
5. Cycle repeats forever...

**The agent never converges to a stable, robust strategy.**

### 2.3 Catastrophic Forgetting

Another problem is **catastrophic forgetting**. Neural networks can "forget" previously learned skills when learning new ones.

**Example**:
1. You train your agent on 1 million human games (Behavioral Cloning)
2. Agent learns sensible, human-like play
3. You start RL self-play
4. Agent discovers a "cheese" strategy that wins against itself
5. Agent over-optimizes for cheese strategy
6. Agent completely forgets all the human-like fundamentals

**Result**: Your agent might beat itself but lose to any reasonable opponent who doesn't fall for the cheese.

### 2.4 Why This Matters for VGC

VGC has:
- **Imperfect information** (you don't know opponent's moves, items, EVs)
- **Complex strategy space** (team preview, switching, mega/Z/tera/dynamax)
- **Diverse opponent strategies** (hyper offense, trick room, weather, balance)

A robust VGC agent needs to:
- Play solidly against unknown opponents
- Not over-fit to any single strategy
- Remember fundamentals while learning new tricks

**This is exactly what regularization helps with.**

---

## 3. Standard RNaD: Single Reference Model

### 3.1 The Core Idea

**RNaD** (Regularized Nash Dynamics) solves the cycling problem by adding a "anchor" to training. The agent is penalized for straying too far from a **reference model**.

Think of it like this:
- The **main model** ($\pi$) is actively learning
- The **reference model** ($\pi_{ref}$) is a frozen snapshot of a previous version
- The loss function says: "Learn to win, BUT don't become too different from the reference"

### 3.2 The RNaD Loss Function

$$L_{total} = L_{policy} + \beta \cdot L_{value} - \gamma \cdot H + \alpha \cdot D_{KL}(\pi \| \pi_{ref})$$

Let's break down each term:

#### Term 1: $L_{policy}$ (Policy Loss / PPO Loss)

This is the standard RL learning signal. It says: "Increase the probability of actions that led to good outcomes."

**Simplified intuition**:
- If you won and used Thunderbolt, increase P(Thunderbolt)
- If you lost and used Protect, decrease P(Protect)

The actual PPO formula is more sophisticated (uses "advantages" and clipping), but the intuition is the same.

#### Term 2: $\beta \cdot L_{value}$ (Value Loss)

Your model has a "value head" that predicts: "What's the probability of winning from this state?"

**Example**:
- State: You have 4 healthy Pokémon, opponent has 1 low HP Pokémon
- Value prediction: 95% win probability

The value loss measures how accurate these predictions are:
$$L_{value} = (V_{predicted} - V_{actual})^2$$

Where $V_{actual}$ is 1 if you won, 0 if you lost.

**Why this matters**: Good value predictions help the agent make better decisions.

#### Term 3: $-\gamma \cdot H$ (Entropy Bonus)

**Entropy** measures how "spread out" a probability distribution is.
- High entropy: Probabilities are spread evenly (lots of uncertainty)
- Low entropy: One action dominates (very confident)

We **subtract** entropy from the loss (equivalently, add it as a bonus) to encourage **exploration**. Without this, the agent might become overly confident too quickly and miss better strategies.

**Example**:
- Low entropy policy: "Always use Thunderbolt" (bad — doesn't explore)
- High entropy policy: "All moves are roughly equal" (explores more)

#### Term 4: $\alpha \cdot D_{KL}(\pi \| \pi_{ref})$ (RNaD Regularization)

**This is the key RNaD term.**

It measures how different the current policy is from the reference model. By adding this to the loss, we penalize the agent for changing too much.

**Intuition**: "You can learn new things, but don't forget your fundamentals."

**The $\alpha$ parameter** controls the strength:
- High $\alpha$: Agent changes very slowly (stable but slow learning)
- Low $\alpha$: Agent can change quickly (faster learning but risk of cycling)
- Typical values: 0.001 to 0.1

### 3.3 How the Reference Model Updates

The reference model isn't static forever. Every N training steps, we update it:

```python
# Every checkpoint_interval steps:
if step % checkpoint_interval == 0:
    ref_model.load_state_dict(main_model.state_dict())
```

This creates a "moving anchor" that tracks the main model, but with a delay.

**Analogy**: It's like having a coach who watches your games from last week. They keep you grounded in what worked before, while allowing you to improve.

### 3.4 Visualization

```
Training Step: 0        1000      2000      3000      4000      5000
                |---------|---------|---------|---------|---------|
Main Model:     A ------> B ------> C ------> D ------> E ------> F
                          ↑                   ↑                   ↑
Reference:      A --------|-------> B --------|-------> D --------|
                          |                   |                   |
                    (update ref)        (update ref)        (update ref)
```

The reference "lags behind" the main model, providing stability.

---

## 4. Portfolio Regularization: The Core Idea

### 4.1 The Limitation of Single Reference

Standard RNaD uses **one** reference model. This can still lead to problems:

**Scenario**:
1. Step 0: Reference = Policy A (balanced play)
2. Step 1000: Main model has learned Policy B (offensive)
3. Step 1000: Reference updates to Policy B
4. Step 2000: Main model has learned Policy C (defensive)
5. Step 2000: Reference updates to Policy C
6. **Problem**: The agent has completely "forgotten" Policy A!

If Policy A was actually important (maybe it beats Policy C), we've lost valuable knowledge.

### 4.2 The Portfolio Solution

Instead of one reference model, maintain a **collection** (portfolio) of past models:

$$\text{Portfolio} = \{\pi_1, \pi_2, \pi_3, ..., \pi_K\}$$

Now, regularize against **all of them** (or a weighted combination):

$$L_{portfolio} = \alpha \cdot \sum_{i=1}^{K} w_i \cdot D_{KL}(\pi \| \pi_i)$$

**Intuition**: "Don't stray too far from ANY of your past successful strategies." [TODO: call out that this is max, not min]

### 4.3 Visual Comparison

**Standard RNaD** (Single Reference):
```
Current Policy ←---(regularize)--- Reference Model
                                        ↓
                            (gets replaced each checkpoint)
```

**Portfolio RNaD** (Multiple References):
```
Current Policy ←---(regularize)--- Reference 1 (from step 0)
               ←---(regularize)--- Reference 2 (from step 1000)
               ←---(regularize)--- Reference 3 (from step 2000)
               ←---(regularize)--- Reference 4 (from step 3000)
                                        ↓
                            (old ones removed as new ones added)
```

### 4.4 Why This Helps

1. **Prevents Cycling**: Even if the current policy "forgets" Reference 2, References 1, 3, and 4 keep it grounded.

2. **Maintains Diversity**: The agent is encouraged to maintain strategies that work in different ways.

3. **Smoother Learning**: Instead of abrupt changes when the reference updates, the portfolio provides continuous, smooth regularization.

4. **Robustness**: The agent is less likely to be exploited because it maintains multiple viable strategies.

---

## 5. Implementation Approaches

There are many ways to implement portfolio regularization. Each has tradeoffs.

### 5.1 Uniform Weighting (Simplest)

**Idea**: All references are equally important.

$$L_{portfolio} = \frac{\alpha}{K} \sum_{i=1}^{K} D_{KL}(\pi \| \pi_i)$$

```python
def compute_portfolio_loss_uniform(current_logits, portfolio_models, states):
    """
    Compute KL divergence to all reference models, weighted equally.

    Args:
        current_logits: Action logits from current policy [batch, actions]
        portfolio_models: List of K reference models
        states: Current battle states [batch, features]

    Returns:
        Scalar loss value
    """
    total_kl = 0.0
    K = len(portfolio_models)

    for ref_model in portfolio_models:
        with torch.no_grad():
            ref_logits = ref_model(states)

        kl = kl_divergence(current_logits, ref_logits)
        total_kl += kl

    # Average across all references
    return total_kl / K
```

**Pros**:
- Simple to implement
- Stable — no reference is favored over others
- All past strategies are preserved equally

**Cons**:
- Very old, possibly irrelevant policies get same weight as recent good ones
- As portfolio grows, regularization might become too strong

**When to use**: Good starting point; use if you don't have strong intuitions about weighting.

### 5.2 Recency Weighting (Exponential Decay)

**Idea**: Recent policies matter more than old ones.

$$w_i = \frac{\lambda^{K-i}}{\sum_{j=1}^{K} \lambda^{K-j}}, \quad \lambda \in (0, 1)$$

This gives exponentially decaying weights. If $\lambda = 0.9$ and $K = 5$:
- Reference 5 (newest): weight ∝ $0.9^0 = 1.0$
- Reference 4: weight ∝ $0.9^1 = 0.9$
- Reference 3: weight ∝ $0.9^2 = 0.81$
- Reference 2: weight ∝ $0.9^3 = 0.73$
- Reference 1 (oldest): weight ∝ $0.9^4 = 0.66$

```python
def compute_portfolio_loss_recency(current_logits, portfolio_models, states, decay=0.9):
    """
    Weight references by recency — newer references get higher weight.

    Args:
        decay: Lambda value, typically 0.8-0.95
    """
    K = len(portfolio_models)

    # Compute weights: newest reference has highest weight
    raw_weights = [decay ** (K - 1 - i) for i in range(K)]
    total_weight = sum(raw_weights)
    weights = [w / total_weight for w in raw_weights]  # Normalize to sum to 1

    total_kl = 0.0
    for i, ref_model in enumerate(portfolio_models):
        with torch.no_grad():
            ref_logits = ref_model(states)

        kl = kl_divergence(current_logits, ref_logits)
        total_kl += weights[i] * kl

    return total_kl
```

**Pros**:
- Prioritizes recent learning (which is probably more relevant)
- Old strategies naturally fade away
- Prevents stale policies from dominating

**Cons**:
- Might forget important old strategies
- Lambda hyperparameter needs tuning

**When to use**: When you expect the game meta or optimal strategy to evolve over training.

### 5.3 Performance-Based Weighting

**Idea**: Weight references by how well they perform against the current policy.

$$w_i \propto \text{win\_rate}(\pi_i \text{ vs } \pi_{current})$$

References that beat the current policy get higher weight — these are strategies we don't want to forget!

```python
def compute_portfolio_loss_performance(
    current_logits,
    portfolio_models,
    portfolio_win_rates,  # Pre-computed win rates
    states
):
    """
    Weight references by their win rate against current policy.

    portfolio_win_rates[i] = win rate of reference i when playing against current policy
    Higher win rate = we need to remember this strategy more!
    """
    # Normalize win rates to sum to 1
    total_wr = sum(portfolio_win_rates)
    weights = [wr / total_wr for wr in portfolio_win_rates]

    total_kl = 0.0
    for i, ref_model in enumerate(portfolio_models):
        with torch.no_grad():
            ref_logits = ref_model(states)

        kl = kl_divergence(current_logits, ref_logits)
        total_kl += weights[i] * kl

    return total_kl
```

**Pros**:
- Directly addresses weaknesses — strategies that beat us get preserved
- Very principled from a game theory perspective

**Cons**:
- Expensive: Requires playing evaluation games to compute win rates
- Win rates change as current policy changes (stale data problem)
- Can over-fit to specific counter-strategies

**When to use**: When you have compute budget for evaluation games; for fine-tuning a strong agent.

### 5.4 Diversity-Based Sampling

**Idea**: Instead of computing KL to ALL references each step, sample ONE reference based on how often each has been sampled.

```python
def compute_portfolio_loss_diversity(
    current_logits,
    portfolio_models,
    selection_counts,  # How many times each reference has been selected
    states,
    temperature=1.0
):
    """
    Sample one reference, prioritizing under-sampled ones.

    This ensures all references get equal representation over time,
    while only computing one KL per update (efficient!).
    """
    # Convert selection counts to probabilities
    # Under-sampled references have HIGHER probability
    scores = [-count for count in selection_counts]  # Negate: low count = high score
    probs = F.softmax(torch.tensor(scores) / temperature, dim=0)

    # Sample one reference
    selected_idx = torch.multinomial(probs, num_samples=1).item()
    selected_model = portfolio_models[selected_idx]

    # Update selection count
    selection_counts[selected_idx] += 1

    # Compute KL to just this one reference
    with torch.no_grad():
        ref_logits = selected_model(states)

    return kl_divergence(current_logits, ref_logits)
```

**Pros**:
- Very efficient: Only one forward pass per reference per update
- Memory efficient: Don't need all references in GPU memory simultaneously
- Encourages diversity: All references get sampled equally over time

**Cons**:
- High variance: Gradient estimates are noisy (different reference each time)
- Might miss important references on any given update

**When to use**: When memory or compute is constrained; for larger portfolios.

### 5.5 Maximum KL Selection (Adversarial)

**Idea**: Always regularize against the reference we've diverged from the MOST.

```python
def compute_portfolio_loss_max_kl(current_logits, portfolio_models, states):
    """
    Find the reference we've drifted furthest from, regularize against it.

    Intuition: "What strategy have I forgotten the most? Remember that one!"
    """
    max_kl = 0.0
    max_ref_logits = None

    for ref_model in portfolio_models:
        with torch.no_grad():
            ref_logits = ref_model(states)

        kl = kl_divergence(current_logits, ref_logits)
        if kl > max_kl:
            max_kl = kl
            max_ref_logits = ref_logits

    # Return KL to the most-forgotten reference
    return kl_divergence(current_logits, max_ref_logits)
```

**Pros**:
- Directly prevents catastrophic forgetting
- Simple and principled

**Cons**:
- Can slow down learning by anchoring to very old/different policies
- Ignores references we're still close to (which might also be important)

**When to use**: When preventing forgetting is the top priority.

### 5.6 Minimum KL Selection (Implemented)

**Idea**: Regularize against the reference we're CLOSEST to (minimum KL divergence).

```python
def compute_portfolio_loss_min_kl(current_logits, portfolio_models, states):
    """
    Find the reference we're closest to, regularize against it.

    Intuition: "Stay close to at least ONE past strategy."
    """
    min_kl = None
    min_ref_logits = None

    for ref_model in portfolio_models:
        with torch.no_grad():
            ref_logits = ref_model(states)

        kl = kl_divergence(current_logits, ref_logits)
        if min_kl is None or kl < min_kl:
            min_kl = kl
            min_ref_logits = ref_logits

    # Return KL to the closest reference
    return kl_divergence(current_logits, min_ref_logits)
```

**Pros**:
- **Allows faster learning**: Only penalized for diverging from ALL strategies, not ANY strategy
- **Maintains diversity naturally**: Can specialize away from some refs as long as close to others
- **Prevents strategy collapse**: As long as one past strategy is maintained, avoids cycling
- **More permissive than averaging**: Lower regularization penalty = faster adaptation to new strategies

**Cons**:
- **Can still forget strategies**: If policy drifts to be close to ref A, it might be very far from refs B, C, D
- **May not prevent all cycling**: If minimum KL reference keeps changing, could still oscillate
- **Less conservative**: Allows more deviation from portfolio than uniform/max approaches

**Comparison to Maximum KL**:
- **Maximum KL** (section 5.5): "Don't forget ANYTHING" — very conservative, slow learning
- **Minimum KL** (this section): "Stay close to SOMETHING" — more permissive, faster learning
- **Trade-off**: Minimum KL allows specialization while maximum KL forces generalization

**When to use**: 
- When you want fast adaptation while maintaining some stability
- When portfolio contains diverse strategies and you want agent to pick compatible ones
- **Currently implemented in `PortfolioRNaDLearner`** — this is the default approach

**Mathematical intuition**: 
With K references, minimum KL creates a "union" of safe zones:
```
Policy Space:
           ○ ref_1        Minimum KL: stay in any circle
              ○ ref_2     (allows movement between them)
    ○ ref_3
         ○ ref_4

Current policy can be in ref_1's neighborhood OR ref_2's OR ref_3's OR ref_4's
```

With maximum/average KL, you'd need to stay close to ALL references simultaneously (intersection of neighborhoods), which is much more restrictive.

### 5.7 Hybrid Approach: Anchor + Sampling (Recommended)

**Idea**: Combine the best of multiple approaches:
1. **Always** regularize against a fixed "anchor" (e.g., the BC model)
2. **Also** sample from the portfolio for diversity

```python
def compute_portfolio_loss_hybrid(
    current_logits,
    anchor_model,          # The original BC model (never changes)
    portfolio_models,      # Dynamic portfolio of past checkpoints
    selection_counts,
    states,
    anchor_weight=0.5,     # How much weight to give the anchor
    training_progress=0.0  # 0.0 at start, 1.0 at end of training
):
    """
    Hybrid approach: Fixed anchor + sampled portfolio reference.

    The anchor ensures we never completely forget BC fundamentals.
    The portfolio ensures we maintain diverse RL-learned strategies.

    anchor_weight can decay over training (start high, end low).
    """
    # Decay anchor weight as training progresses
    # Early: rely heavily on BC anchor
    # Late: rely more on RL-learned portfolio
    effective_anchor_weight = anchor_weight * (1.0 - 0.5 * training_progress)

    # Compute KL to anchor
    with torch.no_grad():
        anchor_logits = anchor_model(states)
    anchor_kl = kl_divergence(current_logits, anchor_logits)

    # Sample one portfolio reference
    scores = [-count for count in selection_counts]
    probs = F.softmax(torch.tensor(scores), dim=0)
    selected_idx = torch.multinomial(probs, num_samples=1).item()
    selection_counts[selected_idx] += 1

    with torch.no_grad():
        portfolio_logits = portfolio_models[selected_idx](states)
    portfolio_kl = kl_divergence(current_logits, portfolio_logits)

    # Weighted combination
    total_kl = effective_anchor_weight * anchor_kl + (1 - effective_anchor_weight) * portfolio_kl

    return total_kl
```

**Pros**:
- Best of both worlds
- Guaranteed to maintain BC fundamentals (via anchor)
- Maintains RL diversity (via portfolio)
- Smooth transition during training

**Cons**:
- More hyperparameters to tune
- Slightly more complex

**When to use**: **Recommended for VGC** — this approach directly addresses our goal of building on BC while improving via RL.

---

## 6. Theoretical Analysis: Will This Converge?

This section explains WHY portfolio regularization works from a theoretical perspective. Don't worry if the math feels dense — the intuitions are what matter.

### 6.1 The Goal: Nash Equilibrium

In two-player zero-sum games (like Pokémon battles), the goal of training is to find a **Nash Equilibrium** — a strategy that cannot be exploited.

**Definition**: A policy $\pi^*$ is a Nash Equilibrium if:
$$\max_{\pi'} u(\pi', \pi^*) \leq u(\pi^*, \pi^*)$$

In English: No opponent strategy $\pi'$ can do better against $\pi^*$ than $\pi^*$ does against itself.

**Exploitability**: We measure how far a policy is from Nash equilibrium by its **exploitability**:
$$\epsilon(\pi) = \max_{\pi'} u(\pi', \pi) - u(\pi^{Nash}, \pi^{Nash})$$

A policy with exploitability $\epsilon$ can be beaten by at most $\epsilon$ more than the Nash equilibrium strategy.

### 6.2 Why Regularization Helps: Intuition

Imagine the space of all possible policies as a landscape:

```
          Policy Space (2D visualization)

    Bad ←                              → Good

         *  *           🏔️ Nash
        * *  *         *   *
       *   *  *       *     *
      *     * *      *       *    ← Local optima
     *       **     *         *      (exploitable strategies)
    *         *    *           *
   ──────────────────────────────────
           (many policies)
```

Without regularization, the learning algorithm might:
1. Find a local optimum (exploitable strategy)
2. Bounce between local optima (cycling)
3. Never reach the Nash equilibrium peak

Regularization helps by:
1. **Smoothing the landscape**: The KL penalty makes the effective landscape smoother
2. **Providing a gradient toward stability**: If you're too far from known-good policies, you get pulled back
3. **Preventing oscillation**: You can't bounce between extremes because the penalty increases

### 6.3 Convergence of Standard RNaD (Background)

The original DeepNash paper (Perolat et al., 2022) proves that RNaD converges. Here's the key insight:

**Define a "regularized game"** where the payoff is modified:
$$\tilde{u}(\pi, \pi') = u(\pi, \pi') - \alpha \cdot D_{KL}(\pi \| \pi_{ref})$$

The regularization term acts like a "cost" for deviating from the reference.

**Key Theorem** (informal): If:
1. The reference policy is updated slowly (averaging or periodic snapshots)
2. The learning rate is appropriately set
3. The game is two-player zero-sum

Then the **time-averaged policy** converges to a Nash equilibrium:
$$\bar{\pi}_T = \frac{1}{T} \sum_{t=1}^{T} \pi_t \xrightarrow{T \to \infty} \pi^{Nash}$$

The exploitability is bounded by:
$$\epsilon(\bar{\pi}_T) \leq O\left(\alpha + \frac{1}{\sqrt{T}}\right)$$

As $T \to \infty$ and $\alpha \to 0$ (slowly), exploitability goes to zero.

### 6.4 Extending to Portfolio Regularization

Now we prove (informally) that portfolio regularization also converges.

#### Step 1: Convexity Preservation

The portfolio KL term is a **weighted average** of KL divergences:
$$L_{portfolio} = \sum_{i=1}^{K} w_i \cdot D_{KL}(\pi \| \pi_i), \quad \text{where } \sum_i w_i = 1$$

**Key Property**: KL divergence is **convex** in its first argument. A weighted average of convex functions is still convex.

**Why this matters**: Convexity means there are no "bad" local minima. Any local minimum is also a global minimum. This is essential for convergence guarantees.

#### Step 2: Effective Reference Interpretation

We can interpret portfolio regularization as regularizing toward an "effective reference":
$$\pi_{eff} = \arg\min_{\pi'} \sum_i w_i \cdot D_{KL}(\pi' \| \pi_i)$$

This is the policy that minimizes total divergence from all portfolio members — it's like the "center" of the portfolio in KL-space.

**Interpretation**: Portfolio regularization ≈ regularizing toward the centroid of past policies.

#### Step 3: Tracking Condition

For convergence, the portfolio must "track" the learning policy — it shouldn't fall too far behind.

**Condition**: Add a new reference to the portfolio whenever:
$$\min_i D_{KL}(\pi_{current} \| \pi_i) > \epsilon_{threshold}$$

This ensures the portfolio always contains a policy "close" to the current one.

#### Step 4: Exploitability Bound for Portfolio RNaD

**Theorem** (informal): For portfolio RNaD with:
- Weights $w_i = 1/K$ (uniform)
- Portfolio size $K$
- Reference added every $M$ steps
- Regularization strength $\alpha$

The time-averaged policy has exploitability:
$$\epsilon(\bar{\pi}_T) \leq O\left(\alpha K + \frac{1}{\sqrt{T}} + \frac{K}{M}\right)$$

**What each term means**:
- $\alpha K$: More references = more regularization = slower movement toward Nash
- $1/\sqrt{T}$: Standard convergence rate (improves with more training)
- $K/M$: How well the portfolio tracks learning (add references often enough!)

**Key Insight**: As long as $\alpha$ is small enough and we update the portfolio frequently, convergence is guaranteed.

### 6.5 Conditions for Convergence

For portfolio regularization to work well, these conditions should hold:

1. **Bounded Portfolio Size**: $K$ shouldn't grow forever
   - If $K \to \infty$, regularization dominates and learning stops
   - Use a fixed-size portfolio with FIFO eviction

2. **Sufficient Update Frequency**: Add new references often enough
   - Rule of thumb: Every 1,000-10,000 training steps
   - Monitor divergence from portfolio to calibrate

3. **Weight Decay** (optional but helpful): Old references should lose influence over time
   - Prevents ancient, irrelevant policies from holding back learning

4. **Maintain Exploration**: Keep the entropy bonus ($\gamma$) high enough
   - Without exploration, the agent might converge to a suboptimal deterministic policy

5. **Decreasing Regularization**: Consider annealing $\alpha$ over training
   - High $\alpha$ early for stability
   - Low $\alpha$ late for fine-tuning

### 6.6 The Math of KL Divergence (Optional Deep Dive)

For those interested in the mathematical details:

**Forward vs Reverse KL**:

$$D_{KL}(\pi \| \pi_{ref}) = \mathbb{E}_{a \sim \pi}\left[\log \frac{\pi(a|s)}{\pi_{ref}(a|s)}\right] \quad \text{(Forward KL)}$$

$$D_{KL}(\pi_{ref} \| \pi) = \mathbb{E}_{a \sim \pi_{ref}}\left[\log \frac{\pi_{ref}(a|s)}{\pi(a|s)}\right] \quad \text{(Reverse KL)}$$

**RNaD uses Forward KL**. This has a specific effect:

- If $\pi_{ref}(a) \approx 0$ but $\pi(a) > 0$: Forward KL is large → penalizes the current policy for taking actions the reference never took
- If $\pi(a) \approx 0$ but $\pi_{ref}(a) > 0$: Forward KL is small → doesn't strongly penalize dropping actions the reference liked

**Intuition**: Forward KL is "mode-seeking" — it prevents the current policy from doing things the reference considered very bad, but allows dropping things the reference did.

This is what we want for RL: Don't make catastrophic mistakes, but feel free to become more focused/refined.

---

## 7. Practical Considerations for VGC

Now let's get practical. How should you actually implement this for VGC?

### 7.1 Portfolio Size

**How many references should you keep?**

| Size | Pros | Cons | Recommendation |
|------|------|------|----------------|
| 2-4 | Fast computation, low memory | May not capture enough diversity | Good for testing/debugging |
| 8-16 | Good coverage of strategy space | Moderate memory usage | **Recommended for most cases** |
| 32+ | Excellent diversity | Slow, high memory, over-regularization risk | Only if you have compute/memory to spare |

**Memory Estimate**:
- Your model is ~50-100 MB
- Portfolio of 8 models = 400-800 MB GPU memory
- This is usually fine for a 24GB GPU like RTX 3090

### 7.2 When to Add New References

**Option A: Fixed Interval**
```python
if training_step % 5000 == 0:
    portfolio.add(copy.deepcopy(current_model))
```

**Option B: Divergence-Based**
```python
min_kl = min(compute_kl(current_model, ref) for ref in portfolio)
if min_kl > divergence_threshold:  # e.g., 0.5
    portfolio.add(copy.deepcopy(current_model))
```

**Recommendation**: Start with fixed interval (every 5,000-10,000 steps). Monitor KL divergence to validate.

### 7.3 When to Remove Old References (Eviction)

**Option A: FIFO (First-In-First-Out)**
```python
if len(portfolio) > max_size:
    portfolio.pop(0)  # Remove oldest
```

**Option B: Keep the Anchor**
```python
if len(portfolio) > max_size:
    # Never remove the first reference (BC model)
    portfolio.pop(1)  # Remove second-oldest
```

**Option C: Performance-Based**
```python
# Remove the reference that the current policy beats most easily
win_rates = [evaluate_vs_current(ref) for ref in portfolio]
worst_idx = argmin(win_rates)
portfolio.pop(worst_idx)
```

**Recommendation**: Use Option B — always keep the BC anchor.

### 7.4 The Anchor Model

**Strongly Recommended**: Always keep the BC (Behavioral Cloning) model as a permanent anchor.

**Why**:
1. BC model learned from 1M+ human games — it knows sensible play
2. RL can go off the rails, but anchor keeps it grounded
3. Prevents complete catastrophic forgetting of human-like fundamentals

```python
class PortfolioLearner:
    def __init__(self, bc_model_path, ...):
        # Load and freeze the BC model
        self.anchor_model = load_model(bc_model_path)
        for param in self.anchor_model.parameters():
            param.requires_grad = False

        # This is NEVER modified or removed
        self.portfolio = []  # Additional references added during RL
```

### 7.5 Hyperparameter Recommendations

| Parameter | Typical Range | Recommended Start | Notes |
|-----------|---------------|-------------------|-------|
| Portfolio size $K$ | 4-16 | 8 | More = more memory |
| Regularization $\alpha$ | 0.001-0.1 | 0.01 | Higher = more conservative |
| Anchor weight | 0.3-0.7 | 0.5 | For hybrid approach |
| Add interval | 2,000-20,000 | 5,000 | Steps between new references |
| Diversity temperature | 0.5-2.0 | 1.0 | For diversity-based sampling |

### 7.6 Integration with Exploiter Training

Your system already has **exploiter training**. Here's how portfolio regularization interacts:

```
EXPLOITER TRAINING                    MAIN TRAINING
(find weaknesses)                     (improve policy)
       |                                    |
       |     discovers weakness             |
       |──────────────────────>             |
       |                          learns to defend
       |                                    |
       |                          portfolio prevents forgetting
       |                          other defensive strategies!
       |                                    |
```

**The portfolio ensures** that when the main agent learns to defend against Exploiter A, it doesn't forget how to defend against Exploiters B, C, D that were discovered earlier.

**Practical Integration**:
```python
# When an exploiter is successful, consider adding victim as a reference
if exploiter_win_rate > 0.6:
    # The victim was weak — this version of main policy should be "remembered to avoid"
    # Actually, we should keep the MAIN policy that beat the exploiter as a reference
    pass

# After main agent learns to beat an exploiter, add that checkpoint to portfolio
if main_agent_beats_exploiter:
    portfolio.add(copy.deepcopy(main_model))
```

---

## 8. Implementation Recommendations

### 8.1 Suggested Implementation for Your Codebase

Based on your existing `PortfolioRNaDLearner`, here's a recommended enhanced version:

```python
class EnhancedPortfolioLearner:
    """
    Portfolio regularization with fixed anchor + diversity sampling.

    This combines:
    1. Always regularize against BC anchor (prevents catastrophic forgetting)
    2. Sample from portfolio for diversity (prevents cycling)
    3. Decay anchor weight over training (allows more RL innovation late)
    """

    def __init__(
        self,
        main_model: nn.Module,
        bc_model_path: str,
        config: RNaDConfig,
        device: torch.device,
    ):
        self.main_model = main_model
        self.config = config
        self.device = device
        self.updates = 0

        # Load BC model as permanent anchor
        self.anchor_model = self._load_model(bc_model_path)
        self.anchor_model.eval()
        for p in self.anchor_model.parameters():
            p.requires_grad = False

        # Dynamic portfolio (starts empty, grows during training)
        self.portfolio: List[nn.Module] = []
        self.portfolio_add_interval = config.portfolio_add_interval  # e.g., 5000
        self.max_portfolio_size = config.max_portfolio_size  # e.g., 8

        # For diversity-based sampling
        self.selection_counts = []

        # Optimizer
        self.optimizer = torch.optim.Adam(
            main_model.parameters(),
            lr=config.learning_rate
        )

    def _load_model(self, path: str) -> nn.Module:
        """Load a model checkpoint."""
        model = FlexibleThreeHeadedModel(...)  # Use your model architecture
        state_dict = torch.load(path, map_location=self.device)
        model.load_state_dict(state_dict)
        model.to(self.device)
        return model

    def _add_portfolio_reference(self):
        """Add current main model to portfolio."""
        # Deep copy the main model
        new_ref = copy.deepcopy(self.main_model)
        new_ref.eval()
        for p in new_ref.parameters():
            p.requires_grad = False

        self.portfolio.append(new_ref)
        self.selection_counts.append(0)

        # Evict oldest (but never remove index 0 if we want to keep oldest)
        if len(self.portfolio) > self.max_portfolio_size:
            # Remove second-oldest (keep first as it might be important)
            self.portfolio.pop(0)
            self.selection_counts.pop(0)

    def _compute_anchor_kl(
        self,
        current_logits: torch.Tensor,
        states: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Compute KL divergence to the BC anchor model."""
        with torch.no_grad():
            anchor_output = self.anchor_model(states)
            anchor_logits = anchor_output["action_logits"]

        return self._kl_divergence(current_logits, anchor_logits)

    def _compute_portfolio_kl(
        self,
        current_logits: torch.Tensor,
        states: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Sample one portfolio reference and compute KL."""
        if len(self.portfolio) == 0:
            return torch.tensor(0.0, device=self.device)

        # Diversity-based sampling: prioritize under-sampled references
        scores = torch.tensor(
            [-c for c in self.selection_counts],
            dtype=torch.float32
        )
        probs = F.softmax(scores / self.config.diversity_temperature, dim=0)

        selected_idx = torch.multinomial(probs, num_samples=1).item()
        self.selection_counts[selected_idx] += 1

        selected_model = self.portfolio[selected_idx]

        with torch.no_grad():
            ref_output = selected_model(states)
            ref_logits = ref_output["action_logits"]

        return self._kl_divergence(current_logits, ref_logits)

    def _kl_divergence(
        self,
        p_logits: torch.Tensor,
        q_logits: torch.Tensor
    ) -> torch.Tensor:
        """Compute forward KL divergence: D_KL(P || Q)."""
        p_probs = F.softmax(p_logits, dim=-1)
        p_log_probs = F.log_softmax(p_logits, dim=-1)
        q_log_probs = F.log_softmax(q_logits, dim=-1)

        kl = (p_probs * (p_log_probs - q_log_probs)).sum(dim=-1)
        return kl.mean()

    def compute_loss(
        self,
        trajectories: List[Trajectory]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute the full RNaD loss with portfolio regularization.
        """
        # Prepare batch from trajectories
        states, actions, rewards, old_log_probs, old_values = self._prepare_batch(
            trajectories
        )

        # Forward pass through main model
        output = self.main_model(states)
        current_logits = output["action_logits"]
        values = output["value"]

        # PPO losses (standard)
        policy_loss = self._compute_policy_loss(
            current_logits, actions, rewards, old_log_probs
        )
        value_loss = self._compute_value_loss(values, rewards)
        entropy = self._compute_entropy(current_logits)

        # Portfolio regularization
        # Compute progress (0 at start, 1 at end of training)
        progress = min(1.0, self.updates / self.config.total_updates)

        # Anchor weight decays over training
        # Start: 70% anchor, 30% portfolio
        # End: 30% anchor, 70% portfolio
        anchor_weight = 0.7 - 0.4 * progress  # Decays from 0.7 to 0.3

        anchor_kl = self._compute_anchor_kl(current_logits, states)
        portfolio_kl = self._compute_portfolio_kl(current_logits, states)

        rnad_loss = (
            anchor_weight * anchor_kl +
            (1 - anchor_weight) * portfolio_kl
        )

        # Total loss
        total_loss = (
            policy_loss
            + self.config.value_coef * value_loss
            - self.config.entropy_coef * entropy
            + self.config.rnad_alpha * rnad_loss
        )

        return {
            "total_loss": total_loss,
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy": entropy,
            "anchor_kl": anchor_kl,
            "portfolio_kl": portfolio_kl,
            "rnad_loss": rnad_loss,
            "anchor_weight": anchor_weight,
        }

    def update(self, trajectories: List[Trajectory]) -> Dict[str, float]:
        """Perform one update step."""
        self.main_model.train()

        # Compute losses
        losses = self.compute_loss(trajectories)

        # Backprop
        self.optimizer.zero_grad()
        losses["total_loss"].backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.main_model.parameters(),
            self.config.max_grad_norm
        )

        self.optimizer.step()
        self.updates += 1

        # Periodically add new portfolio reference
        if self.updates % self.portfolio_add_interval == 0:
            self._add_portfolio_reference()

        # Return metrics for logging
        return {k: v.item() if torch.is_tensor(v) else v
                for k, v in losses.items()}
```

### 8.2 Configuration Updates

Add these to your `RNaDConfig`:

```python
@dataclass
class RNaDConfig:
    # ... existing fields ...

    # Portfolio regularization
    portfolio_add_interval: int = 5000  # Add reference every N updates
    max_portfolio_size: int = 8         # Maximum number of references
    diversity_temperature: float = 1.0  # Temperature for sampling (higher = more random)
    anchor_decay: bool = True           # Whether to decay anchor weight over training
    initial_anchor_weight: float = 0.7  # Starting anchor weight
    final_anchor_weight: float = 0.3    # Ending anchor weight
```

---

## 9. Monitoring and Debugging

### 9.1 W&B Metrics to Log

Add these metrics to track portfolio health:

```python
def log_portfolio_metrics(self, wandb):
    """Log portfolio-related metrics to W&B."""

    # KL divergence to anchor
    wandb.log({"portfolio/kl_to_anchor": self.last_anchor_kl})

    # KL divergence to each portfolio reference
    for i, ref in enumerate(self.portfolio):
        kl = self._compute_kl(self.main_model, ref)
        wandb.log({f"portfolio/kl_to_ref_{i}": kl})

    # Portfolio diversity (average pairwise KL between references)
    if len(self.portfolio) >= 2:
        pairwise_kls = []
        for i in range(len(self.portfolio)):
            for j in range(i + 1, len(self.portfolio)):
                kl = self._compute_kl(self.portfolio[i], self.portfolio[j])
                pairwise_kls.append(kl)
        wandb.log({"portfolio/avg_pairwise_kl": sum(pairwise_kls) / len(pairwise_kls)})

    # Selection counts (is sampling balanced?)
    wandb.log({
        "portfolio/selection_count_std": np.std(self.selection_counts),
        "portfolio/selection_count_max": max(self.selection_counts) if self.selection_counts else 0,
        "portfolio/selection_count_min": min(self.selection_counts) if self.selection_counts else 0,
    })

    # Max KL drift (are we forgetting any reference too much?)
    if len(self.portfolio) > 0:
        max_kl = max(self._compute_kl(self.main_model, ref) for ref in self.portfolio)
        wandb.log({"portfolio/max_kl_drift": max_kl})

    # Anchor weight (for hybrid approach)
    wandb.log({"portfolio/anchor_weight": self.current_anchor_weight})
```

### 9.2 What to Look For

**Healthy Training Signs**:
- `anchor_kl` stays moderate (0.1-1.0 typically)
- `portfolio/max_kl_drift` stays bounded (doesn't explode)
- `portfolio/avg_pairwise_kl` is non-zero (portfolio is diverse)
- `selection_count_std` is low (all references getting sampled)

**Warning Signs**:

| Symptom | Likely Cause | Solution |
|---------|--------------|----------|
| `anchor_kl` goes to 0 | Agent not learning | Decrease `rnad_alpha`, increase learning rate |
| `anchor_kl` explodes | Catastrophic forgetting | Increase `anchor_weight`, decrease learning rate |
| `max_kl_drift` explodes | Forgetting old strategies | Increase `rnad_alpha`, add references more often |
| All `selection_counts` same | Good! Balanced sampling | N/A |
| One count much higher | Sampling is stuck | Increase `diversity_temperature` |
| `avg_pairwise_kl` ≈ 0 | Portfolio not diverse | Add references less often (they're too similar) |

### 9.3 Debugging Tips

**Problem: Training is too slow / agent isn't improving**
- Decrease `rnad_alpha`
- Decrease `anchor_weight`
- Increase `entropy_coef` for more exploration

**Problem: Agent forgets how to play sensibly**
- Increase `rnad_alpha`
- Increase `anchor_weight`
- Check that BC model is loading correctly

**Problem: Cycling between strategies**
- Add more references to portfolio
- Increase `max_portfolio_size`
- Consider performance-based weighting

**Problem: Out of GPU memory**
- Decrease `max_portfolio_size`
- Use diversity sampling (one KL per update instead of all)
- Consider weight pruning/compression for references

---

## 10. References

### Papers
1. **DeepNash / RNaD Paper**: "Mastering Stratego, the classic game of imperfect information" - Perolat et al., 2022
   - Original RNaD algorithm and convergence proofs
   - https://www.science.org/doi/10.1126/science.add4679

2. **PPO**: "Proximal Policy Optimization Algorithms" - Schulman et al., 2017
   - Foundation for our policy gradient updates
   - https://arxiv.org/abs/1707.06347

3. **NFSP**: "Neural Fictitious Self-Play" - Heinrich & Silver, 2016
   - Alternative approach to learning in imperfect-information games
   - https://arxiv.org/abs/1603.01121

### Your Codebase
- [`rl/learner.py`](learner.py) - Standard RNaD learner
- [`rl/portfolio_learner.py`](portfolio_learner.py) - Current portfolio implementation
- [`rl/agent.py`](agent.py) - Agent wrapper for step-by-step inference
- [`rl/RL.md`](RL.md) - Comprehensive RL system documentation

### Further Reading
- Sutton & Barto, "Reinforcement Learning: An Introduction" (free online)
- Shoham & Leyton-Brown, "Multiagent Systems" (for game theory background)
- Deep RL course by Sergey Levine (UC Berkeley, YouTube)
