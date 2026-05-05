+++
title = "Introduction to AI Systems & Infrastructure"
date = 2026-04-11
description = ""
weight = 1

[extra]
chapter = "Intro"
+++

What is AI composed of, and how are companies like OpenAI, Anthropic, Google, and Microsoft pushing AI for millions of people around the globe to use (whether people want it or not)?
From courses you've taken in previous semesters and the Deep Learning course this semester, you know that one of the essential components of AI is what we call AI models, usually neural network or deep learning models these days, trained on large amounts of data. 
AI models are also essentially a type of software, predominantly implemented using the Python programming language and Python libraries like PyTorch and TensorFlow.

However, just the AI models themselves are far from being able to serve people all over the world with different styles and habits for using AI.
To help inflate the AI bubble even more and integrate AI into every piece of our digital life, all kinds of software, such as search engines, messaging apps, text editors, social media apps, and video games, need to interact with AI models.
However, not all software is coded with Python, and even if it is, you cannot expect all software to be Python functions within a giant Python project that can call each other for communication. Therefore, there must be a more standardized, language-agnostic way to enable communication between AI models and other parts of the digital world, but what is it, and can we implement it ourselves?

As a type of software, AI models also need hardware to run. And not all hardware is created equal. Why do top-of-the-line AI models like OpenAI's ChatGPT and Anthropic's Claude need to run in those data centers each with the same power draw of a small town, rather than on your smartphone?
If you have ever installed and set up Windows from scratch, you can tell that it is impossible for OpenAI and Google to repeat the process of manually setting up each computer in their data centers 10,000 times. But then, what kind of techniques do they use to streamline this process?
And is it even possible for you to run an AI model yourself at home so that you can finally stop paying those corrupt and monopolistic companies?

In this course, we will answer all those questions. Even better, you will learn industry-standard techniques to replicate what AI companies have been doing (well, at least the parts that do not involve burning millions of EURs on supercomputers).
Specifically, we will discuss techniques that enable streamlined interaction between AI systems and all kinds of software in [Part A](@/ai-system/interact-ais/index.md) of the course.
Then we will learn how to deploy AI systems on common real-world infrastructure in [Part B](@/ai-system/infrastructure-and-deployment-of-ai/index.md) of the course.
One day you might become the one who is able to charge everyone hundreds of EURs every month just to have a conversation with AI models.

## How Each Post is Structured

Each blog post under the [AI Systems section](@/ai-system/_index.md) is the corresponding literature for a module, or it serves as an introduction like this one, or it provides information on mini projects and exams of this course.
When you are on any page of this blog site, you can always click the "AIsys" link on the top navigation bar to go to the root of the AI Systems section.

The literature for each module is generally structured as follows: it starts with an introduction to bring out the topic of the module, followed by a "Table of Contents" block that makes navigating the sections of the literature easier, then the main content of the literature split into sections, and finally the exercise for this module.

Throughout the literature, you will find internal links that will jump to other literature in this course for the convenience of reference, like the "[Part A](@/ai-system/interact-ais/index.md)" link above.
You will also find external links, usually leading you to other materials that will explain the corresponding terminologies or concepts in finer detail. For example, the "[IP address](https://www.geeksforgeeks.org/computer-science-fundamentals/what-is-an-ip-address/)" link will lead you to a blog post discussing this concept in detail. If you are not familiar with the concept or you are interested in learning more, you can read the materials behind these links; it is optional.

There will also be extended reading blocks in the form of quote blocks, like the one below.
They include additional learning materials to suit different learning styles, as well as discussions of related topics that are not mandatory to learn for the course, so check them out as you wish.

> Here is an example extended reading block, with a link to video material explaining computer network concepts to aid your study:
> - [IP address explained](https://www.youtube.com/watch?v=7_-qWlvQQtY)

Each module comes with an exercise, giving you a chance to solve a practical problem with the knowledge learned in this module. They also usually build on top of exercises of previous modules, so you do not need to implement everything from scratch every time.
