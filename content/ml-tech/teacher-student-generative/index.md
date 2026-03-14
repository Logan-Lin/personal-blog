+++
title = "Teacher-Student Philosophy Behind One-Step Generative Models"
date = 2026-03-04
description = ""
draft = true
+++

There have been lots of efforts at creating one-step generative models that, as their name suggests, can perform generation in one step, instead of needing multiple unparallelizable steps like in a conventional diffusion model. Previously, I have touched on the topic of one-step generative models in [this preliminary post](../one-step-diffusion-models) and [this post about shortcut models](../ode-sde).
Intuitively, such models can have generation speed superiority, as long as we have decent hardware to parallelize the running of the one-step model at hand.

Obviously, the idea of one-step (or end-to-end) generative models is nothing new. Earlier generative modeling frameworks including variational auto-encoders (VAEs) and generative adversarial networks (GANs) are both natively one-step generative models.
Yet their capabilities at generating high-quality data, like realistic high-resolution images, are limited; especially compared to diffusion models that now dominate generative modeling.

So, why do recent efforts at developing one-step generative models seem to have been rather successful, reported to result in models with comparable or even better performance compared to their multi-step counterparts?
In my observation, they all share one similar philosophy: training the one-step generative model as a **student**, by teaching it with a more powerful (usually multi-step) **teacher** generative model. I call it the teacher-student philosophy behind one-step generative models. And this might be the core distinction that made their generative capabilities emerge compared to VAEs and GANs.

## The Teacher

We will start by designing a performant generative model as the teacher. The teacher should be designed to have a large model capacity and can generate realistic data with enough computing power and time, with the computational efficiency not the primary concern here.
As of right now, there are three main paradigms for designing the teacher: diffusion models (general term, including both discrete-time DDPM and continuous-time score/flow matching here), auto-regressive, and normalizing flows.

Diffusion models generate data by starting from random noise (usually Gaussian noise) and gradually removing a certain scale of noise by predicting the noise to remove with a denoiser network, until the clean generated data is obtained.
In the case of discrete-time DDPM, the number of noise removing steps is fixed; and in the case of continuous-time score/flow matching, the noise removing process is equivalent to solving a differential equation under the hood, and the solution can be estimated with a finite number of steps (essentially, the Euler method).
Either way, the generation capacity of diffusion models comes from multiple generation steps; intuitively, the more steps used, the higher generation capacity the model has.

Auto-regressive models generate data by arranging the data into a certain order of steps, and always condition the generation of the current step on already generated steps. In each step, a sequential model is given the generated steps as input, and it predicts the next step. The generation executes recursively, until the maximum number of steps is reached, or the model produces a termination signal. 
Due to the sequential nature of auto-regressive, it is suitable to generate data that is naturally ordered (like language sequences or time series). Nonetheless, for unordered data like images, one can also break down the data with certain order (e.g., scan the pixels row-by-row) and make it work with auto-regressive models.

Normalizing flows (NFs) generate data by turning clean data into noise step-by-step, and the real generation process is the inverted operations of the steps. While it sounds similar to diffusion models, NFs are unique in that each operation is strictly invertible, which should make them mathematically more robust.
However, the invertibility requirement also imposes restrictions on how we could design an NF, and it can be challenging to scale NFs to complex data.
Recent efforts that successfully make NFs generate complex data mostly follow the idea that, since each operation has to be invertible and thus relatively simple, we can increase NFs' generation capacity by scaling the number of operations to thousands.

One common pattern behind all three paradigms is breaking down the generation process into multiple Markov steps.
This can be seen as an important source of generative capacity that enable them to generate high-quality, complex data.
The downside is that these Markov steps cannot be parallelized, make the generation process relatively slow.

## The Student
