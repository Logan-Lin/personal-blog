+++
title = "Mini Project"
date = 2026-05-28
description = ""
weight = 31

[extra]
chapter = "Info"
+++

Leveraging the knowledge from this course, we will develop and deploy an AI system in our mini project.

## Outcome

The outcome of the project is an AI API server, similar to the one we implemented in [Module A.4](@/ai-system/ai-api-server/index.md), deployed as a container running on a computer.
A client program, running on another computer, should be able to interact with this server.

### Necessary Requirements

The **API server** should have more than one endpoint. You have the flexibility to plan the functionality of these endpoints.
Examples include:

- One route `<domain>/v1/image_classify` for image classification and another route `<domain>/v1/conversation` for LLM-powered natural language conversation
- Multiple versions `<domain>/v1/image_classify` and `<domain>/v2/image_classify` providing different sets of functionality
- Utility routes like `<domain>/v1/model` for listing available AI models

Note that at least one of them should incorporate "AI functionality", for example, an image classifier, a large language model, or an image generator.
For the AI model powering the AI functionality, you can totally use off-the-shelf models from HuggingFace, the model you prepared for other courses, or elsewhere; this is not the focus of this mini project.

> Nevertheless, it is suggested that these AI-enabled endpoints actually compute AI models under the hood.
> In other words, you probably shouldn't just route API calls to the endpoints to other AI service providers, for example to OpenAI or Google, since this kinda makes the API server not qualifiable as an "AI API server", but more of a broker, or a proxy, to existing AI services.

The server should be deployed using the containerization techniques we learned in [Module B.2](@/ai-system/container-fundamentals/index.md) and [Module B.3](@/ai-system/ai-container/index.md), rather than running directly on the computer.

You should also prepare a simple **client program** for validating that the server is functioning correctly. There is no strict requirement for any aspect of this client program, as long as it can demonstrate the functionality of the API server you deployed.

> For implementation and deployment of the server, you are free to use Python libraries, ask AI for help, or even reference the code I included in the blog post.
> But you shouldn't directly copy and paste any code (implemented by others or AI) without any modification, especially if you have no idea what the code means. 
> In other words, I am not against reuse of existing tools and code since it is a common practice in software development, but you have to ensure that you understand your implementation.

### Bonus Points

These are not strict requirements, but are aspects that you can consider to demonstrate your understanding of the knowledge covered in this course:
- Integration with databases for API key management, per-API-key usage tracking, and rate-limiting-protected endpoints ([Module A.4](@/ai-system/ai-api-server/index.md#protecting-and-tracking-the-endpoints))
- Leverage specialized AI computing hardware to accelerate the AI model used in the server, if the machine running the server has any ([Module B.1](@/ai-system/ai-hardware/index.md#specialized-hardware-for-ai))
- Make the server publicly accessible and enable HTTPS ([Module B.4](@/ai-system/cloud-deployment/index.md#going-public-domains-and-https) and [Module B.5](@/ai-system/edge-deployment/index.md#network-getting-past-nat))

Do note that incorporating these optional achievements in the project does not directly grant you a higher score than those who don't.
The purpose of the project is to reflect your understanding of the knowledge we covered in this course, and the course is scored based on the oral exam, not the project outcome directly.

## Report

A report for this mini project should be submitted.
The report should be 3-8 pages, excluding references, containing the following aspects:
- Title and all authors
- Introduction: a short problem analysis or presentation of the background
- Implementation: explain important design and implementation choices of the API server, and its deployment
- Results: evaluation of the API server's functionality
- Conclusion: your reflections on the mini project

Note that the report does not need to have these exact sections.

> The report can provide a concise explanation of your key design choices, the rationale behind them, and the deployment process you followed.
> Focus on demonstrating your understanding of the system you built and showing that it achieves its intended objectives.
> Think of it as walking the reader through your thought process and the outcomes of your work, rather than providing exhaustive code documentation.

## Submission

In the end, you should submit one `.zip` or `.tar.gz` (or other open file bundle formats) file containing:
- The report in PDF format
- All source code necessary to build the API server container, including the source code of the API server, the `Dockerfile`, and others if needed (e.g., `requirements.txt`)
- Source code of the client program

All tests can be done internally. The API server does not have to be up and running after the report and code are submitted.
