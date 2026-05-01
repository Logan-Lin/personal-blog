+++
title = "Spatiotemporal AI Agent Dilemma"
date = 2025-10-29
description = ""
+++
Moving beyond large language models (LLMs), agents are the new hot topic in AI research, and needless to say, lots of researchers in spatiotemporal data mining (ST-DM), like us, are trying to propose novel ideas about AI agents for spatiotemporal data.

First of all, what is an agent and how it differentiates from an LLM? As Anthropic stated in [this post](https://www.anthropic.com/engineering/building-effective-agents):

> **Quotes:**
>
> "Agent" can be defined in several ways. Some customers define agents as fully autonomous systems that operate independently over extended periods, using various tools to accomplish complex tasks. Others use the term to describe more prescriptive implementations that follow predefined workflows. At Anthropic, we categorize all these variations as agentic systems, but draw an important architectural distinction between workflows and agents:
> - Workflows are systems where LLMs and tools are orchestrated through predefined code paths.
> - Agents, on the other hand, are systems where LLMs dynamically direct their own processes and tool usage, maintaining control over how they accomplish tasks.

From my point of view, the primary source of "intelligence" of an agent system still comes from LLMs, but the specific engineering of an agent system allows it to interact with the outside world in more ways than natural language chats with human, and more or less involves automated decision making without human intervention.

Before we go ahead and integrate LLMs into the spatiotemporal domain and build agents just because this is a hyped topic, I want to first discuss a few problems of building AI agents for spatiotemporal data that we should think about.

## Problems with ST Agents

### The Limitation of LLMs

LLMs are not the savior of everything and they have no magic. As its name suggests, they are language models that at best are good at comprehending the logic behind languages, human or computer ones. From a technical standpoint, they are (as of 2025) no more than probabilistic models based on Transformers that predict the next token with highest probability given the context. Their "intelligence" has a high reliance on data mining, e.g., pre-training on large-scale language data scrapped from the internet.

Thus the idea of using their foundational Transformer model to process spatiotemporal data doesn't make a lot of sense. Such data is highly sparse among human conversations and thus the training data of most LLMs. Even if someone includes all publicly available spatiotemporal data into the training data of LLMs, the scale of spatiotemporal data is just pathetic compared to language data. When you don't have enough scale of data to support such a big model, you naturally encounter overfitting: a basic knowledge of machine learning.

And in reality, LLMs are infamous for being insensitive to numbers, let alone spatiotemporal data. Many research papers also questioned the rational and actual effectiveness of using LLMs on spatiotemporal data and time series. I personally have also worked on a few papers adopting LLMs to spatiotemporal data, and despite the results reported in the paper, I will be honest here and say that some clever "tricks" contributes a lot to those results, and in reality such practice will probably result in 1\% performance improvement (at best) at the cost of 10000\% model size increase.

> Tan, Mingtian, et al. "Are Language Models Actually Useful for Time Series Forecasting?"

### The Necessity of Spatiotemporal Agents

Intuitively, AI agents based on LLMs are a far more reasonable use case of LLMs compared to applying them to data other than languages and images, because they actually use the language processing capabilities of LLMs. In most agent systems, languages serve as the media for interacting with humans, and a protocol for LLMs to interact with other computer tools.

Problem is, building an agent for ST-DM in many cases is hard to justify. If we were talking about classical tasks in ST-DM, like traffic flow forecasting, trajectory classification, and next POI recommendation, these are tasks with clear formal problem definition and highly quantifiable performance metrics. You build your loss function based on these metrics, perform back propagation on a neural network, and you naturally get optimum performance (at least on the dataset at hand), as long as your network design is good. There are no interaction with human or intelligent decision making needed here to begin with.

Another idea would be building agents that perform tasks we as human usually do in ST-DM research, like data analysis. Some works already explored this idea. This direction might be more promising, but will also be largely engineering. Also depending who you ask, the usefulness of such agents can still be questionable, seeing that: the procedure of such tasks is highly matured and might make more sense to hard-code instead of letting an LLM to decide; and an experienced researcher would probably do a better and faster job at such tasks compared to an AI agent.

> Hong, Sirui, et al. "Data Interpreter: An LLM Agent for Data Science."

## When to Build ST Agents

Nevertheless, I know the impulse of trying to build AI agents for ST-DM is unstoppable, and I am not against it, no matter if you just want to publish a trash paper or actually want to build something cool. But I think we should think deeper about when it makes sense to build agents, instead of awkwardly jamming the concepts of AI agents and ST-DM together and call it a day.

As Anthropic said themselves:

> **Quotes:**
>
> When building applications with LLMs, we recommend finding the simplest solution possible, and only increasing complexity when needed. This might mean not building agentic systems at all. Agentic systems often trade latency and cost for better task performance, and you should consider when this tradeoff makes sense.
> 
> When more complexity is warranted, workflows offer predictability and consistency for well-defined tasks, whereas agents are the better option when flexibility and model-driven decision-making are needed at scale. For many applications, however, optimizing single LLM calls with retrieval and in-context examples is usually enough.

Translates to the ST-DM domain, if we are to build something useful and purely from an engineering standpoint, we should always aim for the more straightforward approach, and in many cases that means not using agents or LLMs and stick to "classical" methods.

Problem is, such approach might be unappealing to the academic community, in other words, you will find it difficult to publish papers, since publishing papers nowadays often involves increasing complexity for no practical reason. But even just from the standpoint of increasing possibility of getting accepted, I think we should make it clear how to build spatiotemporal agents so that it makes sense.

## How to Build ST Agents

This is something we need to explore further and I also don't have the definite answer. But I can give some vague, personal suggestions for directions.

### Use LLMs for Their Strength

Corresponding to [The Limitation of LLMs](#the-limitation-of-llms), the use of LLMs in ST-DM only makes sense if we are actually utilizing the language (or plus images and audio) processing capabilities of them. Examples include describing a complex scenario when doing POI recommendation in natural language and asking LLMs to sort out the logic problem of which restaurant would the user prefer. Scenarios where "classic" machine learning methods indeed cannot solve effectively.

### Build Agents where Automation is Needed

Corresponding to [The Necessity of Spatiotemporal Agents](#the-necessity-of-spatiotemporal-agents), building an AI agent should start with coming up with new problem definitions or real-world scenarios that are complex enough to demand automated decision making and agentic interaction with the environment. This will probably involve jumping out of existing problem definitions of ST-DM. If we limit ourselves to traditional ST-DM problems, no matter how complex the solution we came up with, it will be very hard to justify since we are just solving a simple enough problem.

### Think About both Engineering and Academic Aspects

We can take a lot of inspiration of successful implementation of AI agents from the industry, for example Cursor and Claude Code. How they are designed to improve the successful rate of task execution. How they optimize external tool calling and context fetching. There are also lots of existing tools that can streamline the implementation of an AI agent. Nowadays you don't really need to code the interaction between LLMs and external tools/resources yourself.

And as academic researchers, we can also focus our attention in the academic aspects of AI agents, and there are surely still lots of missing pieces in building AI agents for ST-DM that worth exploring. For example, proper feedback mechanism is critical for the robustness of an AI agent: the LLMs need to know how each task is executed and how to improve the execution if not satisfactory. Yet due to [The Limitation of LLMs](#the-limitation-of-llms), they cannot fully comprehend spatiotemporal data and thus the feedback loop is not fully closed without proprietary design.
