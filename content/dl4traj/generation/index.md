+++
title = "Trajectory Generation"
date = 2026-02-04
description = ""
weight = 7

[extra]
chapter = "Chapter 7"
+++

Trajectory generation, or trajectory synthesis, aims to generate trajectories that are not actually recorded in the available trajectory data, but are still realistic and follow a target distribution.
It is an important task for expanding the scale of trajectory datasets, when the availability of real-world trajectory data is limited.
It is also a unique task that does not fit neatly into the categories of [end-to-end learning](../end-to-end) or [self-supervised learning](../self-supervised).

In this post, we will focus on the trajectory generation task, and discuss how the problem is defined, how to evaluate the quality and diversity of the generated trajectories, and introduce a general framework for solving the problem.

{{ toc() }}


## Problem Formulation

Trajectory generation differs from tasks like prediction or classification in that there is no single correct output. Instead, the goal is to produce trajectories that are plausible under the distribution of real trajectories. This section formalizes the generation objective and distinguishes between unconditioned and conditioned settings.

In _unconditioned trajectory generation_, the goal is to sample trajectories from a distribution that matches the real trajectory distribution.
Given a dataset $\mathcal{D} = \{\mathcal{T}^{(1)}, \mathcal{T}^{(2)}, \ldots, \mathcal{T}^{(N)}\}$ of $N$ real trajectories, we want to learn a generative model that can produce new trajectories $\hat{\mathcal{T}}$ following the same distribution:

$$\hat{\mathcal{T}} \sim p_\theta(\mathcal{T}) \approx p_{\text{data}}(\mathcal{T})$$

where $p_{\text{data}}(\mathcal{T})$ is the true data distribution and $p_\theta(\mathcal{T})$ is the distribution defined by the generative model.

Each generated trajectory $\hat{\mathcal{T}} = \langle (\hat{l}_1, \hat{t}_1), (\hat{l}_2, \hat{t}_2), \ldots, (\hat{l}_m, \hat{t}_m) \rangle$ should exhibit the same statistical properties as trajectories in $\mathcal{D}$. These include realistic movement patterns, plausible speeds and accelerations, and coherent spatial structure. The length $m$ of generated trajectories may be fixed or variable depending on the model design.

_Conditioned trajectory generation_ extends the unconditioned setting by incorporating constraints that the generated trajectory should satisfy. The generative model now takes a condition $\mathbf{c}$ as input:

$$\hat{\mathcal{T}} \sim p_\theta(\mathcal{T} \mid \mathbf{c})$$

An example form of conditioning is origin-destination (OD) constraints, where the generated trajectory should start at a specified origin and end at a specified destination:

{% math() %}
\mathbf{c} = (l_{\text{start}}, l_{\text{end}}), \quad \hat{l}_1 = l_{\text{start}}, \quad \hat{l}_m = l_{\text{end}}
{% end %}

Other forms of conditioning include geographical constraints (generating trajectories within a specific city or region), temporal constraints (requiring specific start times or trip durations), waypoint constraints (passing through intermediate locations), and attribute constraints (generating trajectories for a specific transportation mode or user profile).

## Evaluation Metrics

### Distributional Consistency

Evaluation of trajectory generation compares the distribution of generated trajectories against that of real trajectories. This involves choosing a divergence measure and deciding which mobility characteristics to compare.

Let $P$ denote the distribution computed from generated data and $Q$ denote the distribution computed from real data, both over a discrete set $\mathcal{X}$.

_Kullback-Leibler divergence_ (KLD) measures information loss when $P$ approximates $Q$:

$$D_{\text{KL}}(P \| Q) = \sum_{x \in \mathcal{X}} P(x) \log \frac{P(x)}{Q(x)}$$

KLD is asymmetric and undefined when $Q(x) = 0$ for any $x$ where $P(x) > 0$.

_Jensen-Shannon divergence_ (JSD) symmetrizes KLD and is always defined:

$$D_{\text{JS}}(P, Q) = \frac{1}{2} D_{\text{KL}}(P \| M) + \frac{1}{2} D_{\text{KL}}(Q \| M)$$

where $M = \frac{1}{2}(P + Q)$. JSD is bounded between 0 and 1 when using base-2 logarithms.

_Earth mover's distance_ (EMD), or Wasserstein distance, incorporates the geometry of $\mathcal{X}$ by measuring the minimum cost to transform $P$ into $Q$:

{% math() %}
\text{EMD}(P, Q) = \min_{\gamma \in \Gamma(P, Q)} \sum_{x, x' \in \mathcal{X}} \gamma(x, x') \cdot d(x, x')
{% end %}

where $d(x, x')$ is the ground distance between elements. EMD penalizes shifts to nearby bins less than shifts to distant bins, but has no fixed range.

These divergence measures can be applied to various mobility characteristics, each capturing a different aspect of trajectory data.

Spatial distribution compares the frequency of visits across locations. The set $\mathcal{X}$ is the set of discretized locations (grid cells, POIs, or road segments), and $P(x)$, $Q(x)$ are the proportions of visits to location $x$. A related metric is the location popularity rank (G-rank), which compares the rank ordering of most frequently visited locations to evaluate whether the model captures the heavy-tailed nature of human mobility.

Trip length distribution compares how far trajectories travel. Trip length can be the origin-destination distance, total path length, or number of points. The set $\mathcal{X}$ consists of distance bins. A related measure is the radius of gyration, which captures the spatial spread of a trajectory as the root mean square distance of all locations from their centroid.

Temporal distribution compares when trips occur. This includes start time distribution (with $\mathcal{X}$ as hours of day) and dwell time distribution at locations.

Origin-destination flows compare the frequency of trips between location pairs. The set $\mathcal{X}$ is the set of all (origin, destination) pairs. Travel patterns compare the distribution of location subsequences. The set $\mathcal{X}$ consists of the most frequent $n$-grams (e.g., 3-location sequences) and their counts or rankings.

Motion dynamics compare kinematic properties along trajectories. The set $\mathcal{X}$ consists of binned values for quantities such as segment distance (distance between consecutive points), acceleration, or jerk (rate of change of acceleration).

### Trajectory Validity

The divergence measures above evaluate whether the aggregate statistics of generated trajectories match those of real data, but do not verify whether individual trajectories are physically plausible.
For that, we can use trajectory validity metrics to check structural and kinematic constraints at the individual trajectory level, and report the proportion of trajectories (or trajectory segments) that satisfy the constraints.

For example, spatial connectivity measures whether consecutive locations are reachable from one another; for road network-constrained trajectories, this means checking if adjacent road segments are connected in the network graph. Speed validity checks whether the implied velocity between consecutive points falls within realistic bounds for the transportation mode. Similarly, acceleration and turn angle constraints can detect trajectories with implausible sudden changes in speed or direction. For vehicle trajectories, road compliance verifies that generated points lie on the road network rather than in buildings or open water.

### Diversity

Beyond matching real data distributions and producing valid trajectories, a good generative model should also produce diverse outputs rather than repeatedly generating identical or near-identical trajectories.

A simple measure of diversity is the uniqueness ratio, defined as the proportion of distinct trajectories in the generated set: $|\text{Unique}(\hat{\mathcal{D}})| / |\hat{\mathcal{D}}|$. There are multiple ways to determine whether two trajectories are identical.

For road network-constrained trajectories where the locations are road segments, two trajectories can be considered identical if they share the exact same sequence.
For continuous GPS trajectories, defining uniqueness is less straightforward since exact coordinate matches are unlikely; one approach is to discretize trajectories onto a grid or road network before comparison, or to define a similarity threshold below which two trajectories are considered duplicates.

### Downstream Performance

Since the motivation behind trajectory generation is typically to provide more data for training models and performing downstream tasks, one can directly assess the effectiveness of the generated trajectories by measuring downstream task performance of models trained on them.
If models trained on generated data achieve similar or better results than those trained on real data, the generated trajectories are useful for that application.

## General Framework

Three families of generative models have been widely applied to data generation: variational auto-encoders (VAEs), generative adversarial networks (GANs), and diffusion models. Each can be adapted to generate trajectories.
Each offers a different mechanism for learning and sampling from the trajectory distribution $p_\theta(\mathcal{T})$.

### Variational Auto-encoder

Recall that we touched on the VAE framework in the [self-supervised learning discussion](../self-supervised/).
VAE learns a generative model by training an encoder-decoder pair with a probabilistic latent space. For trajectory generation, the encoder maps an input trajectory $\mathcal{T}$ to parameters of a latent distribution, typically a Gaussian:

{% math() %}
q_\phi(\mathbf{z} \mid \mathcal{T}) = \mathcal{N}(\boldsymbol{\mu}_\phi(\mathcal{T}), \text{diag}(\boldsymbol{\sigma}^2_\phi(\mathcal{T})))
{% end %}

The decoder reconstructs the trajectory from a sampled latent vector:

$$p_\theta(\mathcal{T} \mid \mathbf{z})$$

Training maximizes the evidence lower bound (ELBO):

{% math() %}
\mathcal{L}_{\text{VAE}} = \mathbb{E}_{q_\phi(\mathbf{z} \mid \mathcal{T})}[\log p_\theta(\mathcal{T} \mid \mathbf{z})] - D_{\text{KL}}(q_\phi(\mathbf{z} \mid \mathcal{T}) \| p(\mathbf{z}))
{% end %}

where the first term is the reconstruction likelihood and the second term regularizes the posterior toward a standard Gaussian prior $p(\mathbf{z}) = \mathcal{N}(\mathbf{0}, \mathbf{I})$.

To generate new trajectories, the decoder samples from the prior and decodes:

$$\mathbf{z} \sim p(\mathbf{z}), \quad \hat{\mathcal{T}} \sim p_\theta(\mathcal{T} \mid \mathbf{z})$$

### Generative Adversarial Network

The GAN framework learns a generator that transforms random noise into realistic samples, trained through an adversarial game with a discriminator. The generator $G_\theta$ maps a noise vector $\mathbf{z} \sim p(\mathbf{z})$ to a trajectory:

$$\hat{\mathcal{T}} = G_\theta(\mathbf{z})$$

The discriminator $D_\psi$ attempts to distinguish generated trajectories from real ones. Training optimizes:

{% math() %}
\min_\theta \max_\psi \mathbb{E}_{\mathcal{T} \sim p_{\text{data}}}[\log D_\psi(\mathcal{T})] + \mathbb{E}_{\mathbf{z} \sim p(\mathbf{z})}[\log(1 - D_\psi(G_\theta(\mathbf{z})))]
{% end %}

The generator learns to produce trajectories that the discriminator cannot distinguish from real data.

### Diffusion Model

Diffusion models learn to generate data by reversing a gradual noising process. The forward process adds Gaussian noise to a trajectory over $T$ timesteps:

{% math() %}
q(\mathcal{T}_t \mid \mathcal{T}_{t-1}) = \mathcal{N}(\mathcal{T}_t; \sqrt{1-\beta_t} \mathcal{T}_{t-1}, \beta_t \mathbf{I})
{% end %}

where $\beta_t$ is a noise schedule. After sufficient steps, $\mathcal{T}_T$ becomes approximately standard Gaussian noise. The generative model learns the reverse process:

$$p_\theta(\mathcal{T}_{t-1} \mid \mathcal{T}_t)$$

which iteratively denoises a sample starting from pure noise $\mathcal{T}_T \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$.

A popular reparameterization technique for training a diffusion model is to let the model predict the noise $\boldsymbol{\epsilon}$ added at each step. Given a noisy trajectory $\mathcal{T}_t$ at diffusion timestep $t$, the denoising network {% m() %}\boldsymbol{\epsilon}_\theta(\mathcal{T}_t, t){% end %} predicts the noise component, and the training objective is:

{% math() %}
\mathcal{L}_{\text{diffusion}} = \mathbb{E}_{t, \mathcal{T}, \boldsymbol{\epsilon}}[\|\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\mathcal{T}_t, t)\|^2]
{% end %}

## Adapting to Trajectories

The generative frameworks above were developed primarily for images or text, and applying them to trajectories introduces several practical challenges.

### Handling Mixed Feature Types

Trajectories contain both continuous features (coordinates, timestamps) and discrete features (road segment IDs, POI identifiers, transportation modes).
For VAEs and GANs, handling mixed feature types should be straightforward. The VAE decoder or the GAN generator can use separate output heads for each feature type.

Diffusion models require more consideration since the iterative denoising process is designed to operate on continuous values. Three main strategies address this.

The first strategy encodes discrete features into continuous representations before diffusion. Each discrete feature is converted to a continuous form such as one-hot vectors or learned embeddings. The diffusion process then operates entirely in continuous space, adding noise to both the original continuous features and the encoded discrete features. After the final denoising step, the continuous representations of discrete features are decoded back to discrete values, for example by taking the argmax of the denoised one-hot vector or finding the nearest embedding in the vocabulary.

The second strategy uses separate diffusion processes for continuous and discrete features. Discrete diffusion models define forward processes that corrupt discrete tokens through random substitutions rather than Gaussian noise, and learn reverse processes that recover the original tokens.
One can combine a standard continuous diffusion branch for coordinates and timestamps with a discrete diffusion branch for categorical features, with the two branches sharing information through cross-attention or joint conditioning.

The third strategy is to use latent diffusion models that operate in a learned latent space. A pre-trained encoder maps the full trajectory, including both continuous and discrete features, into a unified continuous latent representation. The diffusion process operates entirely in this latent space. After denoising, a pre-trained decoder maps the latent representation back to the original mixed-type trajectory.
The encoder and decoder can be trained separately with standard auto-encoding objectives.

### Handling Variable Length

Trajectories usually vary in length, but many generative architectures assume fixed-size outputs. Three main strategies address this mismatch.

_Padding and masking_ fixes a maximum length and pads shorter trajectories with a constant value, typically zero for all features. During training, the model implicitly learns that zero-valued points represent padding rather than actual locations. At generation time, the model produces fixed-length outputs, and the actual trajectory is extracted by discarding points whose features are zero or close to zero. This is straightforward but wasteful for datasets with high length variance, and the model must learn to produce clean transitions between real content and padding.

_Autoregressive generation_ produces points one at a time until a termination condition. The model learns to output a special end token or a termination probability at each step:

$$P(\text{stop} \mid \mathbf{z}, \hat{l}_1, \ldots, \hat{l}_i)$$

Generation continues until the model signals completion. This naturally handles variable length but requires sequential decoding.

_Length prediction_ first predicts the trajectory length, then generates that many points. This separates the length decision from content generation and allows parallel decoding, but errors in length prediction can propagate to the generated trajectory.

### Incorporating Conditions

For all three generative frameworks, a straightforward way to incorporate conditions is to include them as additional input during training. The model learns to correlate conditions with the corresponding trajectory patterns, and at generation time, providing a condition steers the output accordingly. In VAEs, the condition $\mathbf{c}$ is fed to both encoder and decoder. In GANs, both generator and discriminator receive the condition. In diffusion models, the denoising network takes the condition as input: $\boldsymbol{\epsilon}_\theta(\mathcal{T}_t, t, \mathbf{c})$.

Diffusion models offer additional techniques for conditional generation. _Classifier guidance_ enables conditioning an unconditionally-trained diffusion model by using a separate classifier $p_\phi({\mathbf{c}} \mid \mathcal{T}_t)$ that predicts the condition from noisy trajectories. During generation, the gradient of the classifier with respect to the noisy trajectory is used to guide the denoising direction toward samples that satisfy the condition. This allows adding new conditions without retraining the diffusion model, as long as a classifier can be trained for the desired condition. By extension, for continuous-valued conditions, one can train a regressor instead and use its gradient to derive "regressor guidance".

_Classifier-free guidance_ avoids training a separate classifier by jointly training the diffusion model on both conditioned and unconditioned objectives. During training, the condition is randomly dropped with some probability, so the model learns both
{% m() %}\boldsymbol{\epsilon}_\theta(\mathcal{T}_t, t, \mathbf{c}){% end %}
and
{% m() %}\boldsymbol{\epsilon}_\theta(\mathcal{T}_t, t){% end %}.
At generation time, the two predictions are combined:

{% math() %}
\tilde{\boldsymbol{\epsilon}} = \boldsymbol{\epsilon}_\theta(\mathcal{T}_t, t) + w \cdot (\boldsymbol{\epsilon}_\theta(\mathcal{T}_t, t, \mathbf{c}) - \boldsymbol{\epsilon}_\theta(\mathcal{T}_t, t))
{% end %}

where $w > 1$ amplifies the effect of the condition. This provides explicit control over condition strength without requiring an external classifier.

For hard constraints like origin-destination requirements, post-processing or constrained decoding can enforce that generated trajectories satisfy the constraints exactly. In autoregressive generation, the first point can be fixed to the origin, and generation can be guided or truncated to reach the destination.
Diffusion models pose more difficulty for hard constraints due to their multi-step nature. Intermediate noisy states do not directly correspond to valid trajectories, so constraints applied during denoising may not translate accurately to the final output.

> 1. Kingma, Diederik P., and Max Welling. "Auto-Encoding Variational Bayes."
> 2. Goodfellow, Ian J., Jean Pouget-Abadie, Mehdi Mirza, et al. "Generative Adversarial Nets."
> 3. Ho, Jonathan, Ajay Jain, and Pieter Abbeel. "Denoising Diffusion Probabilistic Models."
> 4. Dhariwal, Prafulla, and Alexander Nichol. "Diffusion Models Beat GANs on Image Synthesis."
> 5. Ho, Jonathan, and Tim Salimans. "Classifier-Free Diffusion Guidance."
> 6. Kapp, Alexandra, Julia Hansmeyer, and Helena Mihaljević. "Generative Models for Synthetic Urban Mobility Data: A Systematic Literature Review."
> 7. Jiang, Wenjun, Wayne Xin Zhao, Jingyuan Wang, and Jiawei Jiang. "Continuous Trajectory Generation Based on Two-Stage GAN."
> 8. Wei, Tonglong, Youfang Lin, Shengnan Guo, et al. "Diff-RNTraj: A Structure-Aware Diffusion Model for Road Network-Constrained Trajectory Generation."



