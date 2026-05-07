+++
title = "Building AI-enabled Containers"
date = 2026-05-10
description = ""
weight = 23

[extra]
chapter = "Module B.3"
+++

In [Module B.2](@/ai-system/container-fundamentals/index.md) we used Docker to run programs from images that someone else had already built and published.
That covers most off-the-shelf software, but our AI server from [Module A.4](@/ai-system/ai-api-server/index.md) is not on Docker Hub. Nobody is going to package it for us.
If we want to run our server in a container on another machine, we have to build the image ourselves.

Building our own image is a different exercise from using one.
We have to decide what files go inside, install the Python dependencies our server needs, and pick a default command for the container to run.
On top of that, AI servers come with their own challenges that a typical web app does not have.
Model weights are large and have to live somewhere; inference often wants a GPU, which the container needs explicit access to; and the data our server collects (users, request logs) needs to be stored somewhere that does not get wiped out when a container is removed.

In this module we will look at the tools used to build container images, walk through containerizing the API server we built in [Module A.4](@/ai-system/ai-api-server/index.md), and cover the AI-specific parts as we run into them.
We will end with how to publish the image so others can pull it down with a single command.

{{ toc() }}


## Building Images

Docker offers two ways to build a custom image: an interactive one with `docker commit`, and a declarative one with a `Dockerfile`.
The interactive approach is rarely the right answer in practice, but it is worth seeing once because it makes the layered structure of images concrete.


### The Interactive Approach (Why Not)

In [Module B.2](@/ai-system/container-fundamentals/index.md#using-off-the-shelf-containers) we saw that `docker run -it python:3.13 bash` drops us into a fresh container.
Inside, we can install packages, copy files in, and configure the environment, all of which write to the container's writable layer.
The [`docker commit`](https://docs.docker.com/reference/cli/docker/container/commit/) command then snapshots that writable layer into a brand new image:
```bash
# Start an interactive container
docker run -it --name builder python:3.13 bash

# Inside the container, install dependencies
pip install fastapi uvicorn
exit

# On the host, snapshot the modified container into a new image
docker commit builder my-fastapi-image:dev
```
After this, `my-fastapi-image:dev` exists locally and can be run like any other image.

This works, but it is rarely a good idea.
There is no record of what we actually installed; if the image breaks in six months, we cannot reproduce it without remembering every command we typed.
There is also nothing for a teammate, or our future selves, to read, so the image is essentially a black box.
Every change also requires going through the same manual steps again, which is slow and easy to get wrong.

The interactive approach is sometimes useful for one-off snapshots, such as capturing a debugging session, but it should not be the way we build images we plan to use repeatedly.
And most fundamentally, it throws away the [layered structure](@/ai-system/container-fundamentals/index.md#layered-filesystems) that makes images efficient in the first place. Every change we made inside the container, big or small, gets squashed into a single opaque layer on top of the base image, defeating the per-layer caching and sharing we covered in the previous module.


### Dockerfile (The Recommended Way)

The fix is to describe the image as code instead of building it interactively.
A [**Dockerfile**](https://docs.docker.com/reference/dockerfile/) is a plain-text file that lists, step by step, what should go into the image, much like a recipe lists the steps to prepare a dish.
Docker reads the file from top to bottom and produces an image whose contents are the result of applying each step.

Putting the build steps in a file gives us everything that interactive building lacks.
The steps are documented (the file _is_ the documentation), versioned (we can commit it to Git like any other source file), and reproducible: anyone with the Dockerfile can rebuild the same image with `docker build`, the same way anyone with the recipe can cook the same dish (maybe not exactly the same in real life, but in the context of containers it will be truely identical).

A minimal Dockerfile looks like this:
```dockerfile
FROM python:3.13-slim
WORKDIR /app
COPY hello.py .
CMD ["python", "hello.py"]
```
This says: start from the official slim Python 3.13 image, set the working directory to `/app`, copy `hello.py` from the host into the working directory, and when a container is started from this image, run `python hello.py` as the default command.

We build the image with `docker build`, pointing it at the directory containing the Dockerfile (called the **build context**):
```bash
docker build -t my-hello:1.0 .
```
The `-t` flag tags the resulting image as `my-hello:1.0`. The `.` at the end tells Docker that the build context is the current directory, which is the set of files Docker has access to during the build.
Anything we want to `COPY` into the image has to live somewhere in this context.

Once built, the image runs like any other:
```bash
docker run my-hello:1.0
```

> The build context is sent to the Docker daemon as a tarball at the start of every build.
> If we accidentally include a `.git` directory or hundreds of megabytes of model weights we did not need, we will pay for it on every build.
> A [`.dockerignore` file](https://docs.docker.com/build/concepts/context/#dockerignore-files) (similar in spirit to `.gitignore`) lists patterns that should be excluded from the context, keeping our builds fast.


### Common Dockerfile Instructions

A Dockerfile has a small but expressive vocabulary.
The instructions we will use the most are listed below.

- `FROM <image>` picks the base image to build on. This is always the first instruction. The choice of base influences the size of the final image and the libraries already available; `python:3.13-slim` is a common starting point for Python projects.
- `WORKDIR <path>` sets the working directory inside the image. Subsequent instructions like `COPY` and `RUN` happen with this as their current directory.
- `COPY <src> <dest>` copies files or directories from the build context into the image. The most common use is to bring our application code in.
- `RUN <command>` executes a shell command during the build, in the most recent layer. Typical uses are installing system packages (`RUN apt-get install ...`) or Python dependencies (`RUN pip install ...`).
- `ENV <KEY>=<value>` sets an environment variable that will be visible inside the container at runtime.
- `EXPOSE <port>` documents that the container listens on a particular port. It does not actually publish the port; it is just a hint to the reader and to tools.
- `CMD ["program", "arg1", ...]` sets the default command that runs when a container is started from the image. It can be overridden with `docker run <image> <other-command>`.
- `ENTRYPOINT ["program"]` sets a fixed command that always runs, with `CMD` providing default arguments. Useful for images that wrap one specific program; we will not need it for our server.

The full list of instructions is in the [Dockerfile reference](https://docs.docker.com/reference/dockerfile/), but these are the ones we will rely on for the rest of the module.


### Layer Order Matters

Recall from [Module B.2](@/ai-system/container-fundamentals/index.md#layered-filesystems) that a Dockerfile maps closely to the layers of the resulting image: roughly one layer per instruction.
Docker caches layers between builds, so an instruction whose inputs have not changed reuses the cached layer instead of running again.
This is what makes rebuilding an image fast.

The cache rule has one important consequence: as soon as one layer's input changes, every layer below it has to be rebuilt as well, even if their own inputs did not change.
So we want layers that change rarely (like `pip install` for dependencies that we do not touch every day) above layers that change often (like our own application code).

A common newcomer mistake looks like this:
```dockerfile
FROM python:3.13-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
CMD ["python", "main.py"]
```
This builds correctly the first time, but every change to `main.py` invalidates the `COPY . .` layer, which in turn forces the `pip install` layer to rebuild from scratch on every iteration.
For a project with PyTorch in its dependencies, that means a few minutes of wasted time every time we save the file.

The fix is to copy just the dependency manifest first, install the dependencies, and only then copy the rest of the source code:
```dockerfile
FROM python:3.13-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["python", "main.py"]
```
Now `pip install` only reruns when `requirements.txt` changes.
Edits to `main.py` only invalidate the final `COPY . .` layer, which is fast.
The general rule is to put rarely changing instructions earlier and frequently changing ones later.

> Below are some additional resources for writing better Dockerfiles.
> - [Dockerfile best practices (docs)](https://docs.docker.com/build/building/best-practices/), the canonical list from Docker
> - [Optimize the build cache (docs)](https://docs.docker.com/build/cache/optimize/), a focused guide on cache invalidation
>
> Two more advanced techniques you will run into:
> - [**Multi-stage builds**](https://docs.docker.com/build/building/multi-stage/) let us use one base image to compile or download artifacts and a smaller base for the final image, copying only the artifacts across. This often shaves hundreds of megabytes off the final image.
> - [**Buildx**](https://docs.docker.com/reference/cli/docker/buildx/) is Docker's modern builder, which adds features like building for multiple CPU architectures (e.g., an `amd64` server and an `arm64` Raspberry Pi) from the same Dockerfile.


## Containerizing the AI API Server

We can now apply this to the API server we built in [Module A.4](@/ai-system/ai-api-server/index.md).
The end goal is a single image that bundles our code, its Python dependencies, and the model weights, so that anyone with our image can bring up the server with a single `docker run`.


### Project Layout and Dependencies

We will assume the server lives in a project directory `my-ai-server/` containing Python files like `main.py` and `db.py` we wrote in [Module A.4](@/ai-system/ai-api-server/index.md), a `requirements.txt` that lists the Python packages we need, and the `Dockerfile` we are about to write.

### A First Dockerfile

Putting together the patterns above:
```dockerfile
# Start from a small Python base image
FROM python:3.13-slim

# Set the working directory; later COPY and RUN happen here
WORKDIR /app

# Install dependencies first, before copying the application code
# so that pip install only re-runs when requirements.txt changes
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Now copy the application code
COPY main.py db.py ./

# Document the port the server listens on
EXPOSE 8000

# Default command: start the FastAPI app with uvicorn on all interfaces
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```
A few details worth pointing out.

We use `python:3.13-slim` as the base.
The full `python:3.13` image works too but is several hundred megabytes larger because it includes a wider range of OS-level packages we do not need.

The `--host 0.0.0.0` argument to `uvicorn` matters.
By default Uvicorn binds to `127.0.0.1`, which inside a container only accepts traffic from that container.
Binding to `0.0.0.0` makes it listen on all interfaces, including the one that connects the container to the host, so port mapping can actually reach it.

We do not bake the ResNet-18 weights into the image with this Dockerfile.
The first time the server starts, `torchvision` will download them from the internet into the container's writable layer.
That is convenient for development but not great for fresh containers, since each one has to download the same weights again.
We will fix this shortly.


### Building and Running the Image

Build the image from the project directory:
```bash
docker build -t my-ai-server:1.0 .
```
The first build takes a while because `pip install` has to download PyTorch and torchvision, both of which are large packages.
Subsequent builds are much faster as long as `requirements.txt` does not change, since the `RUN pip install` layer is cached.

Once built, run the container with port mapping so we can reach the server from the host:
```bash
docker run -d --name ai-server -p 8000:8000 my-ai-server:1.0
```
Now `http://127.0.0.1:8000/v1/classify` is reachable from our Module A.2 client (or from `curl` or a browser), exactly as in Module A.4.
The difference is that everything the server needs lives inside the container.

If we follow the logs with `docker logs -f ai-server`, we can see the model weights being downloaded the first time a classification request comes in.


### What Belongs in Layers, and What Belongs in Volumes

This brings us to a recurring design choice when containerizing an AI server: which files should live inside the image as layers, and which should be mounted in from the outside as volumes.
A useful rule of thumb is to keep things that should be the same across every running instance inside the image, and things that should differ between instances or persist beyond a single instance outside the image as a volume.

The application code and the Python packages are the easy case.
They go inside the image, since every instance should run the same code with the same dependencies, and that is what our Dockerfile already does.

Model weights usually go inside the image too.
The model is part of what defines the server's behavior, and we want every instance to behave identically.
The downside is image size; `torchvision`'s ResNet-18 weights are about 50 MB, but bigger models can easily push images past several gigabytes.
For our small server, baking them in is fine. We can update the Dockerfile to download them at build time:
```dockerfile
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the ResNet-18 weights so they are baked into a layer
RUN python -c "from torchvision.models import resnet18, ResNet18_Weights; \
               resnet18(weights=ResNet18_Weights.DEFAULT)"

COPY main.py db.py ./
```
Now the weights are part of the image, and a fresh container starts up immediately without a runtime download.
For very large models that would make the image too big to push and pull conveniently, the alternative is to download them at startup into a mounted volume; the first container pays the download cost, and the rest reuse the same volume.

The SQLite database file (`ai_api.db`) is the opposite case.
It stores our users, API keys, and request logs, and it must survive container removal.
If the database lives inside the writable layer, removing the container deletes all our data.
Instead, we point the database file at a host-mounted directory:
```bash
mkdir -p data    # on the host, the directory the database will live in

docker run -d --name ai-server \
    -p 8000:8000 \
    -v $(pwd)/data:/app/data \
    -e DATABASE_URL=sqlite:////app/data/ai_api.db \
    my-ai-server:1.0
```
We expose a `DATABASE_URL` environment variable that the server reads to find the database file (a small change to `db.py`'s `create_engine` call).
Now the database survives container removal, and we can even replace the running container with a newer image while keeping the data intact.

Configuration and secrets such as API keys for upstream services follow the same logic as the database.
Anything that differs between development, staging, and production should live outside the image so the same image can be reused across all of them, typically passed in as environment variables at runtime.


### GPU Passthrough

If we run our server on a machine with an NVIDIA GPU and we want the AI model to use it, the container needs explicit access to the GPU device.
This is not the default. Containers are isolated from host hardware, and the GPU has to be passed through.

The general mechanism here is the `--device <host>` flag, which grants a container access to one specific device file on the host.
The same pattern works for any hardware Linux exposes as a device file, including webcams at `/dev/video0`, serial ports at `/dev/ttyUSB0`, USB devices, and sound cards.
There is also a blunter [`--privileged`](https://docs.docker.com/reference/cli/docker/container/run/#privileged) flag that drops all device restrictions at once.
It shows up in tutorials as a quick fix but strips away most of the isolation containers exist to provide, and is best avoided unless we genuinely need it.

GPUs need more than raw device files, though.
A container that talks to a GPU also needs userspace driver libraries that match the kernel module on the host, and these are tedious to wire up by hand.
For NVIDIA GPUs, the [**NVIDIA Container Toolkit**](https://github.com/NVIDIA/nvidia-container-toolkit) is the standard tool that handles this and bridges the host driver to the container.
Once installed on the host (the toolkit's docs cover Linux, Docker Desktop, and WSL), we can run containers with the `--gpus` flag:
```bash
docker run -d --name ai-server \
    --gpus all \
    -p 8000:8000 \
    my-ai-server:1.0
```
The `--gpus all` flag exposes every GPU on the host to the container.
We can also pass `--gpus "device=0"` to expose only a specific one, or `--gpus 2` to limit how many GPUs the container can use.

The image itself does not need any special tweaks beyond ensuring its Python packages have GPU support.
The PyTorch wheels installed via `pip install torch` already include CUDA support on Linux.
For larger models, basing the image on NVIDIA's pre-built [`nvidia/cuda`](https://hub.docker.com/r/nvidia/cuda) or [`pytorch/pytorch`](https://hub.docker.com/r/pytorch/pytorch) images is sometimes more convenient, since they come with the CUDA libraries preconfigured.

> The `--gpus` flag described above only covers NVIDIA cards, since the toolkit is NVIDIA-specific.
> Other AI hardware we covered in [Module B.1](@/ai-system/ai-hardware/index.md) needs different plumbing, and the level of support varies a lot.
> - AMD GPUs (the ROCm stack) are accessed through `/dev/kfd` and `/dev/dri` device files passed in with `--device` flags. AMD has both an official [ROCm Docker guide](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/how-to/docker.html) and a newer [container toolkit](https://github.com/ROCm/container-toolkit) that brings the ergonomics closer to NVIDIA's `--gpus`.
> - Google's TPUs are not pluggable into a local machine; they live on Google Cloud and are accessed through a TPU VM that we then run our container on top of. Google's docs on [running TPU workloads in a Docker container](https://docs.cloud.google.com/tpu/docs/run-in-container) walk through this in detail.
> - NPUs in consumer devices are mostly not exposed to containers. For example, on macOS, Docker Desktop runs its daemon inside a Linux VM that has no access to the host's GPU or Neural Engine, so containers running there fall back to CPU. The usual workaround when targeting Apple Silicon or other consumer NPUs is to run inference natively on the host (often through frameworks like Core ML or ONNX Runtime) and only containerize the surrounding application code.
>
> If GPU or NPU support is the deciding factor for your project, it is worth checking compatibility before building anything else.


## Distributing Images

A built image lives only on the machine where it was built until we push it to a registry.
Like Git repositories, container images are pushed and pulled to and from registries; the registry is what makes a built image truly portable.

The basic workflow is the same for every registry. We log in, tag the image with the registry's path, and push:
```bash
# 1. Log in (Docker Hub example)
docker login

# 2. Tag the local image with the registry path
docker tag my-ai-server:1.0 yourusername/my-ai-server:1.0

# 3. Push it
docker push yourusername/my-ai-server:1.0
```
The tag follows the format `[<registry>/]<namespace>/<repository>:<tag>`. For Docker Hub the registry part is omitted and `yourusername` becomes the namespace.
After pushing, anyone with `docker pull yourusername/my-ai-server:1.0` can fetch and run the image, no source code or Dockerfile required on their end.

For the choice of registry, refer back to the list we covered in [Module B.2](@/ai-system/container-fundamentals/index.md#docker-hub).
The same options apply on the push side, with one extra consideration: pushing private images, going beyond a small free tier, or hitting Docker Hub's [pull rate limits](https://docs.docker.com/docker-hub/usage/pulls/) for anonymous traffic usually requires a paid plan or a registry tied to whichever platform we already use (GitHub or one of the cloud providers).

> Below are a few additional resources on the topics covered in this module.
> - [What is a Container Image? (video)](https://www.youtube.com/watch?v=wr4gpKBO3ug) by Sysdig, a short walkthrough of the layer-and-image model
> - [The Only Docker Tutorial You Need To Get Started (video)](https://www.youtube.com/watch?v=DQdB7wFEygo) by The Coding Sloth, a fast hands-on tour that includes writing a Dockerfile
> - [The official Dockerfile reference (docs)](https://docs.docker.com/reference/dockerfile/), the canonical list of every Dockerfile instruction with examples
>
> Once we have several containers and the lifecycle gets non-trivial (rolling updates, autoscaling, restarting on failure), we typically move from running individual `docker run` commands to a container orchestration platform.
> [**Kubernetes**](https://kubernetes.io/) is the most popular such platform: a system for declaring how a set of containers should behave (how many replicas, how to update them, where to schedule them) and letting it handle the rest.
> Cloud providers all offer hosted Kubernetes services that take care of running the underlying infrastructure for us.
> We will not use Kubernetes in this course, but it is a natural step beyond what this module covers.


## Exercise: Containerize Your AI API Server

Take the API server you built in the [Module A.4 exercise](@/ai-system/ai-api-server/index.md#exercise-bring-your-ai-api-server-to-life) and turn it into a container image.
Then point your [Module A.2 client](@/ai-system/interact-api-python/index.md#exercise-a-command-line-chatbot) at the containerized server and confirm that it still works end to end.

A reasonable starting point:

1. In the directory containing your Module A.4 server, write a `Dockerfile` along the lines of the one in this module: a slim Python base, dependency install before copying code, an `EXPOSE` for the server's port, and a `CMD` that starts uvicorn bound to all interfaces.
2. Add a `.dockerignore` file that excludes things like `__pycache__`, `.git`, `.venv`, and your local database file. This keeps your build context lean and avoids leaking development files into the image.
3. Build the image with `docker build` and give it a tag you can remember. Watch the layers being created in the build log.
4. Run the container in detached mode with the server's port mapped to your host, and verify that your Module A.2 client can still send a request and receive the same response as before.

Once the basic version works, try the following extensions:

1. Pre-download the model weights at build time so the first request after startup does not have to wait for them. Rebuild the image and confirm with `docker logs` that no download happens at runtime.
2. Mount a host directory as a volume for the SQLite database so the users and request logs survive container removal. Run the server, register a user (or log a request), stop and remove the container, start a new container with the same volume, and verify that the data is still there.
3. If you have access to an NVIDIA GPU, install the NVIDIA Container Toolkit on the host, run the server with `--gpus all`, and modify your inference code to move the model and inputs to `cuda`. Time a few requests to feel the difference between CPU and GPU inference, similar to what we did in [Module B.1](@/ai-system/ai-hardware/index.md#exercise-run-a-model-on-different-hardware).
4. Push the image to a registry of your choice (Docker Hub or GHCR are easiest). Pull it from a different machine if you have one, and confirm that a single `docker run` is enough to bring up a working AI API server with no other setup.

If you finish all the extensions, you will have a fully containerized AI API server with image distribution working, and a feel for how the parts fit together.
From here on, what changes is the infrastructure we deploy to.
The container itself stays the same.
