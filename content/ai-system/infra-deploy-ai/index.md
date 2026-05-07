+++
title = "Infrastructure & Deployment of AI"
date = 2026-05-08
description = ""
weight = 20
aliases = [ "/ai-system/infrastructure-deployment/" ]

[extra]
chapter = "Part B"
+++

With [Part A](@/ai-system/interact-ais/index.md) complete, we have built a system that, while small, is no doubt a complete API server and a functional AI system: an AI model behind a versioned, validated, authenticated, logged, and rate-limited HTTP API, with a Python client that uses it like any commercial AI service.

But during the whole experience, you might have already noticed some of the limitations of our system.
It is relatively slow, especially if you choose to integrate heavier AI models into your API server. Even though we tried to improve the responsiveness of our server through practices like Async, it still takes some time for the server to respond to a request, and its capability of simultaneously responding to multiple requests is ultimately limited by the total computing resources of the computer you run the server on. Bringing back the restaurant analogy in [Module A.4](@/ai-system/ai-api-server/index.md#async-endpoints), there is a limit of how many cooks we can hire, so we cannot expect to infinitely increase the capability of your server simply by putting the endpoints behind Async.
And if you run your server on a computer with a cooling fan, chances are you will start to hear it once your server starts to accept constant incoming requests.

Meanwhile, most AI models behind the APIs served by companies like OpenAI, Anthropic, and Google are technically too big to fit on consumer-level computers you can just walk into Elgiganten or Power to buy.
And they are serving these AI models to millions of people around the globe every day. How did they manage to do that? And how do these companies single-handedly drive the global memory and storage prices crazy?
In this part of the course, we will be answering these questions.

We will start with the basics of computer architecture in [Module B.1](@/ai-system/ai-hardware/index.md), as well as hardware made for AI computing, how they are different from conventional computing hardware while fundamentally still fitting into a computer architecture that is more than half-century old.
In [Module B.2](@/ai-system/container-fundamentals/index.md) and [Module B.3](@/ai-system/ai-container/index.md) we will discuss containerization, one of the most important and widely-adopted techniques to streamline deployment and management of AI systems on computers.
Finally, we will discuss two types of common hardware infrastructure (other than your own personal computers) you will expect to deploy AI systems on: the cloud in [Module B.4](@/ai-system/cloud-deployment/index.md), and the edge in [Module B.5](@/ai-system/edge-deployment/index.md).

At the end of this part, we will have all the important knowledge needed for us to replicate industry-standard practices used by these companies to scale up their AI systems.
Of course the physical hardware and the absurd funding needed for actually doing so are not included.
