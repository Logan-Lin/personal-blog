+++
title = "Introduction to AI Systems & Infrastructure"
date = 2026-09-01
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
However, not all software is coded with Python, and even if it is, you cannot expect each software to be a Python function within a giant Python project that can call each other for communication. Therefore, there must be a more standardized, language-agnostic way to enable communication between AI models and other parts of the digital world, but what is it, and can we implement it ourselves?

As a type of software, AI models also need hardware to run. And not all hardware is created equal. Why do top-of-the-line AI models like OpenAI's ChatGPT and Anthropic's Claude need to run in those data centers each with the same power draw of a small town, rather than on your smartphone?
If you have ever installed and set up Windows from scratch, you can tell that it is impossible for OpenAI and Google to repeat the process of manually setting up each computer in their data centers 10,000 times. But then, what kind of techniques do they use to streamline this process?
And is it even possible for you to run an AI model yourself at home so that you can finally stop paying those corrupt and monopolistic companies?

In this course, we will answer all those questions. Even better, you will learn industry standard techniques to replicate what AI companies have been doing (well, at least the parts that do not involve burning millions of EURs on supercomputers).
Specifically, we will discuss techniques that enable streamlined interaction between AI systems and all kinds of software in [Part A](@/ai-system/interacting-with-ai-systems/index.md) of the course.
Then we will learn how to deploy AI systems on common real-world infrastructure in [Part B](@/ai-system/infrastructure-and-deployment-of-ai/index.md) of the course.
One day you might become the one who is able to charge everyone hundreds of EURs every month just to have a conversation with AI models.
