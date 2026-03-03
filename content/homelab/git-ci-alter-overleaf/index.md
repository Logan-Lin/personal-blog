+++
title = "A Git + CI Alternative to Overleaf"
date = 2026-03-03
description = ""
+++

... or any online collaborative LaTeX editor in general (yeah, I am talking about OpenAI's Prism, or any future platform that might pop up).

I want to demonstrate that with the good old Git and the industry standard CI (continuous integration) pipeline, everything you need from Overleaf can be easily replaced for free, with much better practices for version control and data security.

## Why?

I have touched on the general topic of replacing cloud-based tools with local file synchronization in [this post](../replace-cloud-w-sync).
To repeat the point, I believe pure cloud-based tools like Overleaf are simply unusable and should be avoided at all costs: your data only have one copy on their server, meaning there is zero guarantee that you can always access them or they will never lose them; even the slightest network instability will interrupt your workflow; and why would you trust them to keep your work in progress a secret?
Specific to Overleaf, their price for premium accounts is also absurdly high, and the set of functionalities at free tier is very limited.

It is understandable that Overleaf provides two core functionalities that make it user-friendly:

1. **Collaborative editing**, which can become complicated when multiple team members are working the same part of a project at the same time
2. **LaTeX compiling**, given that TeX runtime like TeXLive isn't the easiest thing to install in the world and is pretty heavy on both storage and computation

However, both functionalities are very limited in the free tier. There can only be a total of 2 editors on one project, LaTeX compilation is slower than the paid tier and times out easier when dealing with large documents.

## Git for Collaborative Editing

To be honest, I think why Git is one of the best collaboration platforms shouldn't need explanation.
Many of Git's components are built to accommodate collaborative coding (and by extension, any text-based file editing) across a big team. Git's handling of all the circumstances that might happen during collaborative editing (e.g., file conflict, lost track of historical edits) is also very robust and fool-proof.

To spell it out, Git's branching and merging let each collaborator work on their own copy without getting in each other's way.
Changes only get merged when they are ready, not because two people happened to edit the same file at the same time. When edits do overlap, Git clearly shows you where the conflict is and makes you resolve it yourself, rather than quietly dropping someone's work like real-time editors sometimes do.

Git also keeps a full record of every edit, who made it, when, and why.
This is way more useful than Overleaf's version history slider. You can compare any two versions, undo specific changes, or pick out individual edits whenever you want. Pull/merge requests also let you review and discuss changes before they go in, which is great for academic writing where you actually want feedback on drafts. Overleaf simply doesn't have anything like this.

Finally, since the whole repo lives on your machine, you can work, commit, and look through history without any internet connection, and just sync up later.
And of course there is no limit on how many people can contribute.

LaTeX files are just plain text, so everything Git already does well for collaborative coding works just as well for writing documents together.
And since collaboration happens through a Git remote, you are free to choose where to host it. It can be GitHub, GitLab, a self-hosted Gitea or Forgejo instance, or even just any server with an SSH connection.

## CI for LaTeX Compiling

Another obstacle stopping many people from working with LaTeX documents locally is the complexity of installing and using TeX runtime.
But once you start to use Git to manage your LaTeX source files, and your Git remote is hosted on a proper Git server like GitHub or self-hosted Forgejo instances, you (and your collaborators) can easily compile LaTeX documents without ever dealing with TeX runtime locally.
The solution is to use a CI pipeline to automate the compile process and run it all on the Git server.

If your Git remote is on GitHub, this can be a GitHub Actions workflow in `.github/workflows/build.yml`:

```yaml
name: Build and Release Documents

on:
  push:
    tags: ['v*']

permissions:
  contents: write

jobs:
  release:
    runs-on: ubuntu-latest
    container:
      image: ghcr.io/xu-cheng/texlive-full:latest
    steps:
      - uses: actions/checkout@v4
      - name: Build document
        run: |
          latexmk -pdf -bibtex -output-directory="./out" paper.tex
      - name: Create GitHub Release
        uses: softprops/action-gh-release@v2
        with:
          files: |
            out/paper.pdf
          body: "Compiled LaTeX documents"
```

The above CI pipeline essentially:
- Run on push of a commit tag starting with `v`, e.g., `v1.0.0`
- Use docker container with full TeX runtime
- Compile the LaTeX document with `latexmk`
- Create a GitHub release with the commit tag, so the compiled document can be accessed under the releases section of your GitHub remote repo

A running example of such workflow is [this repo containing LuaLaTeX song lyrics](https://github.com/Logan-Lin/jp-lyrics).

If you do not trust GitHub for hosting your classified LaTeX documents, self-hosted Git server options including Gitea and Forgejo also support CI pipelines, with similar workflow syntax as GitHub Actions.
As long as one of your team members is nerdy enough to deal with hosting the server and action runners, other members can just use it without caring too much about the implementation.
