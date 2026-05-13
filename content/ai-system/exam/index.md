+++
title = "Exam Information"
date = 2026-05-11
description = ""
weight = 32

[extra]
chapter = "Info"
+++

## Modality and Duration

**Individual oral exam based on submitted project.**
The duration will be 15 minutes, followed by 5 minutes of deliberation.

The agenda for each student is as follows.

- The student gives a 5-minute presentation of their completed mini-project
- We have a 10-minute discussion around topics and questions about the course and the mini-project
  - Start by randomly drawing a topic from the 6 topics listed below, using the [random number generator](https://www.random.org/)
  - Should the student wish to, they may discuss their understanding of that topic. Otherwise, the examiner will start the conversation with questions related to that topic
  - The examiner and censor ask follow-up questions, which may relate to the student's mini-project report or other topics of the course
- Deliberation will take place within 5 minutes, followed by the delivery of the assessment
  - The student will be asked to leave the exam room while the examiner and censor reach an agreement on the assessment
  - After that, the student will return to the room, where the examiner and censor will deliver the final assessment and explanation, plus personal comments and suggestions
  - Should the student have any doubts or questions, we can discuss them briefly

Students are not required to write code during the exam, nor to remember any specific commands or code syntax.
Students may be asked to draw diagrams or solve small tasks manually.
The grade will be based on an overall assessment of both the oral performance and the mini-project, following the 7-point grading scale.

Notes, for example, a concise outline of the literature, important terminology abbreviations and their full names, or documentation of the mini-project, are permitted.
Do note that if a student reads directly from notes or copies them verbatim during the exam, they may be asked to put the notes away. Answers based solely on reading from notes will result in failure.

Timely hand-in of the mini-project is the prerequisite for participation in the exam.

> The core spirit of the exam is to assess students' level of understanding of the knowledge covered in this course.
> Specific to the context of this course, "understanding" means being able to look at a real-world AI system engineering or deployment scenario, recognize the pieces involved and how they fit together, and reason about what to do and why.
> The goal is to see whether students have built up enough such understanding, within the scope indicated by this literature or the example questions below, to confidently engage with these scenarios in practice.
> 
> So, the exam is not about reproducing exact command flags, library function signatures, or code syntax from memory, since we will have access to documentation when actually doing the work.
> Rather, it is about whether students know what to do in a scenario covered in this course, understand the rationale behind it, and can reason about how the techniques across modules relate to each other.
> That said, the core terminology and concepts of the course are still part of what we expect students to know. For example, a strong answer would be able to tell HTTPS from HTTP, a container from a virtual machine, or cloud from edge deployment, and explain why each distinction matters.
> 
> If any of this still feels abstract, the example questions listed under each topic below should give a more concrete sense of what kinds of questions to expect at the exam.

## Topics and Example Questions

Note that the questions listed below may be formulated differently during the exam.
Students are not expected to answer all questions under each topic.
Ideally, students should be able to discuss every topic covered in the course, but this is not required to pass the exam.

While this literature, the blog post for each module excluding the extended reading blocks, is itself the best indicator of the scope of this course, the questions under each topic also indicate the scope of each topic.
Additionally, the number and formulation of questions indicate the importance and expected level of knowledge for each topic.


### 1. Interacting with AI Systems (Module [A.1](@/ai-system/api-fundamentals/index.md)-[A.2](@/ai-system/interact-api-python/index.md))

- What is an API, and what problem does it solve when different applications need to communicate?
- What is an IP address, and what role does it play in network communication?
- What is a domain name, and why do we use one instead of the underlying IP address?
- What does a URL contain on top of the domain name?
- What is a port, and why is it needed when each device already has an IP address?
- Explain the request-response model of HTTP.
- What are the main components of an HTTP request?
  - What kind of information typically goes into the request headers when calling an AI API?
  - What does the body of a POST request to an AI API usually contain, and in what format?
- What are the main components of an HTTP response?
  - How can a client tell from a response whether the request succeeded?
- What is the difference between GET and POST methods?
  - When would you use each one with an AI API?
- Why is it useful to learn API interaction with a library like `requests` rather than starting with a provider's official Python SDK?
- What kinds of failures can occur when calling an AI API from a Python program?
- Why is it important to set a timeout when sending a request to an AI API?
- Why should API keys not be hardcoded in source code?
  - What is the recommended way to keep API keys out of source code?
- How is an image typically sent inside the body of an HTTP request to an AI API?
  - Why can't an image just be placed in the body as raw bytes?
- What is streaming, and why is it the preferred way to receive responses from conversational AI models?
- How does Server-Sent Events (SSE) differ from a regular request-response interaction?

### 2. Building AI Servers (Module [A.3](@/ai-system/api-server/index.md)-[A.4](@/ai-system/ai-api-server/index.md))

- What is an API server, and how does it relate to the APIs we used as clients in the previous topic?
- What role does a framework like FastAPI play in building an API server?
- How do you define a GET endpoint in FastAPI?
- How do you define a POST endpoint in FastAPI?
- What are URL path parameters in FastAPI, and when are they useful?
- What are URL query parameters, and how do they differ from path parameters?
- Why might an API need versioning?
- What problem does Pydantic solve when handling POST request bodies?
  - What happens if a client sends a body that does not match the declared Pydantic model?
- How would you accept an image in the body of a POST endpoint?
  - What does the server have to do with the encoded image before passing it to a model?
- How can you attach custom headers to a response?
- How would you implement a streaming response from a FastAPI endpoint?
- Why should an AI model be loaded once at server startup rather than per request?
  - How does FastAPI's `lifespan` help with that?
- What does it mean for an endpoint to be `async`?
  - When does marking an endpoint `async def` actually help the server?
  - For a CPU-heavy inference endpoint, what additional step lets the server keep accepting other requests while one is being processed?
- Why does an AI API server need authentication?
  - How is API key authentication typically implemented in FastAPI?
- What are the limitations of a hardcoded list of valid keys?
  - How does moving keys into a database address those limitations?
- Why is it useful to log every request in a database?
  - What information would you typically record per request?
- What is rate limiting, and why is it especially important for AI APIs?
- What is fixed-window rate limiting?
- What is sliding-window rate limiting?
  - What are the trade-offs between fixed-window and sliding-window approaches?

### 3. AI Hardware Infrastructure (Module [B.1](@/ai-system/ai-hardware/index.md))

- What is the Von Neumann architecture?
  - Why is it still relevant to today's "AI computers"?
- What are the main components of the Von Neumann architecture?
- What is the difference between instructions and data in a computer?
  - Why does the Von Neumann architecture store both in the same memory?
- What is the role of the CPU in the architecture?
  - What is the Control Unit responsible for inside a CPU?
  - What is the Arithmetic Logic Unit responsible for?
  - How do the Control Unit and the ALU work together?
- What is the role of memory in the architecture?
- What is the role of the I/O system in the architecture?
- What is the purpose of the bus system in a computer?
  - What are the three logical parts of the bus, and what does each one carry?
- Why are CPUs designed primarily for sequential execution?
  - What makes this design less suitable for AI workloads?
  - Why do neural network computations benefit from parallel processing?
- What is the difference between memory latency and memory bandwidth?
  - Which one matters more for AI workloads, and why?
- What makes GPUs well-suited for AI computing?
  - How do GPU cores differ from CPU cores?
  - How does GPU memory differ from CPU memory?
- What is a TPU?
  - How does a TPU's design differ from a GPU's?
  - What does that specialization cost in flexibility?
- What is an NPU?
  - Where would you typically find an NPU?
  - Why are NPUs usually targeted at inference rather than training?
- Do specialized accelerators like GPUs, TPUs, and NPUs replace the Von Neumann architecture?
  - How do they fit into the overall system?
- How would your hardware choice differ between training and inference?
- How would your hardware choice differ between a data center deployment and an edge device?

### 4. Containers (Module [B.2](@/ai-system/container-fundamentals/index.md)-[B.3](@/ai-system/ai-container/index.md))

- What is the "it works on my machine" syndrome, and how do containers help with it?
- What is the difference between a container image and a container?
- Explain the layered filesystem used by container images.
  - Why does this layering matter for efficiency and reusability?
  - What happens when a running container writes to a file that lives in a read-only image layer?
- What is the OCI standard, and why does it matter for the container ecosystem?
- What is the Docker Engine, and what is it responsible for?
- What is the Docker CLI, and how does it relate to the Docker Engine?
- What is Docker Hub, and what role does it play?
- What is a container registry?
  - How does it enable distributing images between machines?
- What is port mapping, and why is it needed for a containerized API server?
- What is a volume, and why is it important for AI applications that need to persist data or hold large model files?
- How would you pass configuration or secrets like API keys to a container without baking them into the image?
- What is a Dockerfile?
  - Why is it preferred over building images interactively with `docker commit`?
- How does each line in a Dockerfile relate to the layered structure of the resulting image?
- Why does the order of instructions in a Dockerfile matter?
  - How does reordering improve build speed?
- When containerizing an AI API server, what would you typically bake into the image?
- What would you typically mount as a volume instead of putting in the container image?
  - Why is the application code usually baked into the image, while the database file is mounted as a volume?
  - Where would you put the AI model weights, and why?
- What is GPU passthrough, and why is it necessary for an AI container to use the host's GPU?

### 5. Cloud Deployment (Module [B.4](@/ai-system/cloud-deployment/index.md))

- What is "the cloud" in practical terms?
- What is the economic argument for the existence of cloud computing?
- What is virtualization?
  - How does it make cloud infrastructure possible?
- How is a virtual machine (VM) different from a container?
  - Why do cloud providers rely on virtualization?
- What is a virtual machine service in the cloud, and what level of control does it give you?
- What is a container service in the cloud, and how does it differ from running your own VM?
- What is a GPU instance, and when would you need one?
- Why might you choose a plain VM over a managed container service for deploying a small AI API server?
  - What are the risks of usage-based pricing compared to a fixed-price VM?
  - What is vendor lock-in, and why is it worth thinking about when choosing cloud services?
- When creating a cloud VM for an AI server, what specifications would you pay attention to?
- Why do firewall rules matter on a cloud VM?
  - What ports would you typically need to open for a public AI API with HTTPS?
- Why is it not enough to just use the VM's public IP address in client programs?
- What is DNS?
  - What is an A record, and what does it do?
- When would you buy a real domain name instead of using a free service like DuckDNS?
- Why is HTTPS necessary for a public AI API?
  - What problems does plain HTTP have?
- What is a certificate authority?
  - What role does Let's Encrypt play in HTTPS?
- What is a reverse proxy?
  - Why do we put Nginx in front of our container instead of teaching the container itself to speak HTTPS?
  - What other benefits does a reverse proxy bring beyond HTTPS?

### 6. Edge Deployment (Module [B.5](@/ai-system/edge-deployment/index.md))

- What is edge computing?
- Why would you prefer edge deployment over cloud deployment?
  - Why might latency be a motivation for deploying AI at the edge?
  - Why might bandwidth use be a motivation for edge deployment?
  - Why might privacy be a motivation for edge deployment?
  - Why might offline operation be a reason to deploy at the edge?
- What is self-hosting?
  - How does it differ from cloud deployment?
  - What typically motivates choosing self-hosting?
- How do edge computing and self-hosting relate to each other?
- What kinds of edge or self-hosted workloads is a Raspberry Pi a good fit for?
- When would you choose an NVIDIA Jetson over a Raspberry Pi?
- What makes a repurposed old laptop an attractive option for self-hosting?
- When does it make sense to build a purpose-built home server instead of using a small single-board computer?
- What is the difference between x86-64 and ARM64 CPU architectures?
  - Why does CPU architecture matter when running containers on edge devices?
- What is a multi-architecture image?
  - How can you build one from a single Dockerfile?
- Why can't a device on the public internet reach your edge device by default?
  - Explain how NAT works in a home network.
- What is CGNAT, and why does it make self-hosting harder?
- If your ISP gives you a real public IP, how can you make your edge device publicly reachable from the internet?
  - What is port forwarding, and what does it do on the router?
  - What is dynamic DNS, and why is it usually needed together with port forwarding?
- If you cannot use port forwarding, how can a small cloud VM combined with a VPN tunnel like WireGuard expose your edge device to the public?
- Why are backups more important for self-hosted services than for cloud VMs?
- Explain the 3-2-1 backup rule.
  - What kinds of failures does it protect against?
