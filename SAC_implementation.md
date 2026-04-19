This pseudo-algorithm outlines the implementation of **Soft Actor-Critic (SAC)** as required by your assignment, incorporating automated temperature tuning, clipped double Q-learning, and the specific evaluation cycles requested.

### 1. Initialization Phase
* [cite_start]**Initialize Networks**: Create an actor network $\pi_\phi$ and two critic networks $Q_{\theta_1}, Q_{\theta_2}$[cite: 21, 22].
* [cite_start]**Target Networks**: Create target critic networks $Q_{\bar{\theta}_1}, Q_{\bar{\theta}_2}$ with weights copied from the critics[cite: 21].
* [cite_start]**Hyperparameters**: Set $\gamma = 0.99$[cite: 19].
* [cite_start]**Temperature**: Initialize the temperature parameter $\alpha$ and set a target entropy $H_{target} = -\text{dim}(\mathcal{A})$ for automated tuning[cite: 44, 59].
* [cite_start]**Buffer**: Initialize a Replay Buffer $\mathcal{D}$ to store transitions[cite: 105].

---

### 2. Phase 1: Initial Random Exploration (Steps 0 to 10,000)
* [cite_start]For each environment timestep $t$ until $10,000$[cite: 20]:
    * [cite_start]**Action**: Select an action $a_t$ by sampling uniformly from the environment's action space[cite: 20].
    * **Step**: Execute $a_t$ in the environment; observe $s_{t+1}$, reward $r_t$, and termination flag $d_t$.
    * **Store**: Save the transition $(s_t, a_t, r_t, s_{t+1}, d_t)$ in the Replay Buffer $\mathcal{D}$.
    * [cite_start]**Reset**: If the episode ends ($d_t$ is true), reset the environment to a new starting state[cite: 40].

---

### 3. Phase 2: Online Training Loop (Steps 10,001 to End)
* [cite_start]For each environment timestep $t$[cite: 20]:
    * **Action Selection**:
        * Input current state $s_t$ into Actor $\pi_\phi$.
        * Actor outputs mean $\mu$ and standard deviation $\sigma$.
        * [cite_start]Sample noise $\epsilon \sim \mathcal{N}(0, 1)$ and compute $a = \tanh(\mu + \sigma \odot \epsilon)$ (Reparameterization Trick)[cite: 21].
    * **Interaction**: Execute $a_t$, observe $s_{t+1}, r_t, d_t$, and store in $\mathcal{D}$.
    * **Update Critics (Clipped Double Q-learning)**:
        * Sample a batch from $\mathcal{D}$.
        * Compute target Q-value: 
          $$y = r + \gamma(1-d) \left( \min_{i=1,2} Q_{\bar{\theta}_i}(s', a') - \alpha \log \pi_\phi(a'|s') \right)$$
          [cite_start]*(where $a'$ is sampled from the current policy for state $s'$)*[cite: 19, 21].
        * [cite_start]Update $\theta_1, \theta_2$ by minimizing the Mean Squared Error against $y$[cite: 21].
    * **Update Actor**:
        * [cite_start]Update $\phi$ to minimize the loss: $J_\pi(\phi) = \mathbb{E} [\alpha \log \pi_\phi(a|s) - \min_{i=1,2} Q_{\theta_i}(s, a)]$[cite: 21, 22].
    * **Update Temperature ($\alpha$)**:
        * [cite_start]Update $\alpha$ by minimizing the entropy loss against $H_{target}$[cite: 44, 59].
    * **Target Update**: Perform a soft update of target networks: $\bar{\theta}_i \leftarrow \tau \theta_i + (1-\tau)\bar{\theta}_i$.

---

### 4. Phase 3: Periodic Offline Evaluation
* [cite_start]**Trigger**: Every $10,000$ environment timesteps[cite: 105].
* **Procedure**:
    * Pause training updates.
    * [cite_start]Run exactly **20 episodes**[cite: 105].
    * [cite_start]**Deterministic Actions**: In these episodes, choose actions using the mean of the distribution: $a = \tanh(\mu)$ (no sampling/noise)[cite: 20].
    * [cite_start]**Logging**: Calculate the **average undiscounted return** across these 20 episodes[cite: 1].
    * [cite_start]**Plotting**: Record this average return on the y-axis against the current total environment timesteps on the x-axis[cite: 14, 108].

---

### 5. Multi-Seed Requirement
* Repeat the entire process (Phase 1 through 3) for **15 random seeds**.
* Calculate the mean performance and confidence intervals across these 15 runs for the final plot.