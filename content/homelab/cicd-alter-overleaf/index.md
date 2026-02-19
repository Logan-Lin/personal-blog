+++
title = "A CI/CD Alternative to Overleaf"
date = 2026-02-20
description = ""
draft = true
+++

... or any online collaborative LaTeX editor in general (yeah, I am talking about OpenAI's Prism, or any future platform that might pop up).

I want to demonstrate that with the good old Git and the industry standard CI/CD pipeline, everything you need from Overleaf can be easily replaced for free, with much better practices for version control and data security.

## Why?

I have touched on the general topic of replacing cloud-based tools with local file synchronization in [this post](../replace-cloud-w-sync).
To repeat the point, I believe pure cloud-based tools like Overleaf is simply unusable and should be avoided at all cost: your data only have one copy on their server, meaning there is zero guarantee that you can always access them or they will never lost them; even the slightest network instability will interrupt your workflow; and why would you trust them to keep your classified work in progress a secret?
Specific to Overleaf, their price for premium accounts is also absurdly high, and the set of functionalities at free tier is very limited.

While it is understandable that Overleaf provide two core functionalities that make it user-friendly:

1. Collaborative editing, which can become complicated when multiple team members are working the same part of a project at the same time
2. LaTeX compiling, given that TeX runtime like TeXLive isn't the easiest thing to install on the world and is pretty heavy on both storage and computation

But both functionalities are very limited in the free tier. There can only be a total of 2 editors on one project, LaTeX compile is slower than paid tier and timeout easier when dealing with large documents.
