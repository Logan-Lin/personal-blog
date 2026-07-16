+++
title = "One CLAUDE.md for an Agentic Second Brain"
date = 2026-06-14
description = "A single context file lets command-line AI agents navigate the Obsidian vault, and the on-disk artifacts behind it."
+++

In [Obsidian and DataView for Personal Project Management](../obsidian-dataview-pm), I turned my Obsidian vault into a live dashboard with the DataView plugin.
That dashboard renders inside Obsidian and serves as my entry point to any project I am working on.

With most modern command-line AI agents like Claude Code able to navigate the terminal environment the way humans do, we can add one of them as a second reader to the same vault, with a straightforward context file.

The vault already has a consistent structure, the one I described in that earlier post.
DataView relies on that structure to build the dashboard, and an agent can use the same structure to navigate the vault on its own.
So the context file mostly transcribes conventions the vault already follows, and nothing about the vault itself changes.

## The Context File

Since I use Claude Code, I keep one `CLAUDE.md` file at the vault root, which it reads on startup.

```markdown
This is a personal Obsidian vault used for project management. It is plain-text, local-first, and synced with Syncthing. The DataView plugin turns the whole vault into one queryable dataset, and `Dashboard.md` is the live entry point.

This context file is the current and authoritative convention for the topics it covers. Some legacy notes might follow deprecated conventions, so prefer this file over patterns found in existing notes.

## Layout

- `Dashboard.md` is the human entry point. It is a set of live `dataviewjs` queries that render inside Obsidian for the user
- `Projects/` holds one note per project, one thing the user works on. Filenames are the project title in human-readable Title Case. See [[#Frontmatter schemas]] for frontmatter fields
- `Programs/` holds one note per program. A program is something a project belongs to, such as a paper venue, a funding call, or a course. Filenames are the program name. See [[#Frontmatter schemas]] for frontmatter fields
- `Schedule/` holds the work log. It has daily notes named `YYYY-MM-DD.md` and weekly notes named `YYYY-Www.md` with an ISO week number padded with a leading zero
- `Drafts/` holds free-form working documents, for example, paper drafts, submission and review notes, rebuttals, job application drafts, lecture notes, how-to and setup notes, brainstorms, abstracts
- `People/` holds one note per person. Filenames are the person's full name
- `Papers/` holds the library of academic papers, one PDF per paper, organized in a topic tree of subfolders. See [[#Papers]]
- `Archive/` holds the user's personal life-admin documents, one file per document in a single flat folder. See [[#Archive]]
- `_templates/` holds the note templates
- `_attachments/` is the attachment sink for images
- `_unsorted/` is the new file inbox. It is configured as the default location for new notes and notes in progress
- `.obsidian/` holds config and plugins
- `.trash/` is Obsidian's local trash

The vault is a git repository, but `.gitignore` excludes every content folder, so git tracks only the vault's core configuration.

## Frontmatter schemas

### Project note

Field order `tags`, `code`, `program`, `people`, `deadline`.

`tags`: list of project tags, see [[#Tag taxonomy]].

`code`: the project's codename, a kebab-case slug. Keep it lower case and separate every word with a hyphen. It maps to a working directory on disk at `~/Documents/Projects/<code>/` that holds the project's non-note files. The note is usually created before that directory, so it is normal for the directory not to exist yet. Inside that directory the files are usually grouped under a purpose subfolder such as `code/` for source or `paper/` for the manuscript repository. That working directory may also carry its own `CLAUDE.md` files, in its root or in any subfolder, holding context specific to the project, so read them when working there.

`program`: the project's current target program when applicable, a quoted wikilink to a program note, for example `program: "[[Program Name]]"`.

`people`: a quoted wikilink to a person note, or a YAML list of them, for example `people: "[[Name]]"` or a block list of `- "[[Name]]"` lines, when applicable. It records the people related to the project.

`deadline`: a date `YYYY-MM-DD`. Rarely set on projects. By default, the dashboard inherits the deadline from the linked program in the `program` field, so set this field only when the project's own deadline differs from its program's.

### Program note

Field order `tags`, `code`, `deadline`.

`tags`: program tag, see [[#Tag taxonomy]].

`code`: the program's codename, a kebab-case slug. Like a project's `code`, it maps to a working directory on disk at `~/Documents/Programs/<code>/`, and that directory likewise may not exist yet and can carry its own `CLAUDE.md` files holding context specific to the program.

`deadline`: a date `YYYY-MM-DD`.

### Other notes

Schedule, Draft, and Person notes all have no YAML frontmatter.

## Tag taxonomy

Tags are the primary metadata.
Do not invent tag values outside the sets defined as follows.

### Project note

Carries exactly three tags, one from each axis.

- type:
  - `project/research` for work that produces new findings through models and experiments, usually aimed at a paper venue
  - `project/writing` for work whose deliverable is the document itself, such as applications, proposals, resource requests, the dissertation, lecture material, and ongoing logs
  - `project/development` for building software, tools, sites, or infrastructure
  - `project/teaching` for courses taught, supervision of student projects, and teaching training
- priority:
  - `priority/main` for projects the owner leads and drives himself as the primary person responsible
  - `priority/collaborate` for projects where the owner supports as a co-author or supervisor rather than the primary driver
- status:
  - `status/ongoing` for projects under active work, with open tasks or recent activity toward a live target
  - `status/on-hold` for projects paused for now but meant to resume later, whether barely started or parked after a submission
  - `status/done` for finished projects whose tasks are all complete, often tied to an accepted venue or a delivered document
  - `status/discarded` for projects dropped and not meant to continue, including superseded directions and abandoned collaborations

### Program note

Carries exactly one tag.

- `program/venue` for conferences and journals
- `program/grant` for funding calls and research awards
- `program/education` for teaching, courses, and supervision
- `program/position` for job openings and position calls

## Authoring rules

When creating or editing notes, follow the existing conventions.
The templates already lay out each note type's section headings and fixed blocks.
The conventions below cover only what the templates do not show.

### Shared conventions

Write tasks as markdown checkboxes, where the symbol inside the brackets sets the status.
The configured statuses are `- [ ]` todo, `- [x]` done, `- [/]` in progress, and `- [-]` cancelled.
The Tasks plugin is installed, so write a task's dates with its date emoji instead of in the task text.
The date emoji are `🛫` start, `📅` due, `✅` done, and `❌` cancelled, each written as the emoji followed by a `YYYY-MM-DD` date, for example `📅 2026-06-24`.

In a note, use `##` and lower for every section heading, matching the templates.
Never use a `#` heading in a note.

Internal links use Obsidian wikilinks, whether they point to another note, a file, or a section heading, for example `[[Name]]` or `[[#Title]]`.
Write every reference to a note or file as a wikilink, including each later mention of it, so a reference never falls back to plain text.
A wikilink names only the target note or file, not its full relative path, so to read or edit the target you first need to search for that file under the vault.
In the unlikely case that two files share the same name, a wikilink disambiguates by prepending enough of the parent folders to the name, such as `[[Folder/Note Name]]`.
External links to a web URL use standard Markdown link syntax instead.

Obsidian supports the rendering of special markdown blocks, listed as follows.
Use one when the user asks for it.
- Callout blocks, written as a blockquote whose first line is a `> [!type]` marker
- Tables, written as a markdown table with a header row and a delimiter row
- Mermaid diagrams, written inside a fenced mermaid code block

When a note needs an asset that markdown itself cannot express, such as a data chart, a complex diagram with SVG elements, or any other figure, produce it with whatever toolset fits the asset and source data, and render the result to PNG.
Do that work in a directory outside the vault, such as the project's working directory or a scratch directory, so the vault never holds the source, the toolchain, or the intermediate files.
Copy only the final PNG into `_attachments/` and embed it in the note with a wikilink such as `![[Image Name.png]]`.

A standard blockquote, one without a `> [!type]` callout marker, is a means of interaction in this vault.
The user writes a comment inside it to request an action or a revision to the note.

### Typed note conventions

Project note: to create a new one, copy `_templates/Project.md` and edit that copy in place, for example `cp "_templates/Project.md" "Projects/<Title>.md"`.
Set the frontmatter fields per [[#Frontmatter schemas]], including a `code` slug if the project has or needs to have a working directory.
Put actionable tasks under `## TODO`.
Below the `---` rule, add free-form sections of content and wikilinks under topical headings.

Program note: to create a new one, copy `_templates/Program.md` the same way, for example `cp "_templates/Program.md" "Programs/<Name>.md"`.
Set the frontmatter fields per [[#Frontmatter schemas]].
Free-form notes follow the `---` rule.

Person note: to create a new one, copy `_templates/People.md` the same way, for example `cp "_templates/People.md" "People/<Name>.md"`, and name the file with the person's name.
Because a person's information tends to change over time, prefer linking to an authoritative source for the person, such as their homepage or profile, rather than copying detailed information into the note, where it can later go out of date.
Below the `---` rule, the note is free-form and can contain any information related to the person.

Schedule note: to record a day's work, edit or create `Schedule/YYYY-MM-DD.md`, keep the template's `##` sections, and place the day's entries under whichever ones fit.
Reference a related project, person, program, or draft with a plain wikilink such as `[[Name]]`, `[[Project Name]]`, or `[[Person Name]]`.
These wikilinks power the referenced note's calendar heatmap.

Draft note: drafts have no schema.
Write the draft as plain markdown, and connect it to the vault by referencing a relevant entry with a plain wikilink.
A program, project, or people note also lists its direct child draft note with a plain wikilink in its free-form sections.

## Context about the user

The vault owner is Yan Lin. Refer to the following for the more professional side of him.
His person note [[Yan Lin]] is the entry point for facts about him. Read it when you need such a fact.
It points to the authoritative sources for his employment, education, research, and other activities.

Refer to the following for the more personal side of him.
He keeps a daily journal in the `## Notes` section of his `Schedule/` daily notes, written more candidly and reflectively than the rest of the vault, where he works through how his projects, plans, and career are going.
His [[Personal Blog]] project also partially reflects his personal opinions on certain topics, through the posts it plans and publishes.
```

For another level of laziness, adding the following sentences to Claude Code's global context means I do not need to manually point to the Obsidian vault in a conversation.

```markdown
The user's personal Obsidian vault is at `~/Documents/app-state/obsidian`.
It tracks his projects, their programs, his work log, and drafts, and is the authoritative source for facts about him.
Read its `CLAUDE.md` for the layout and where specific information is.
A wikilink like `[[Name]]` in a user prompt typically refers to a note in the Obsidian vault.
```

## From Second Brain to Doppelgänger

That `code` field is what makes this more than note navigation, both for me and for AI agents.
While I prefer to keep notes and drafts related to each project inside the Obsidian vault, Obsidian is not very suitable for holding a project's artifacts, such as the source code and the paper manuscript.
Back when I built the vault, I set up a convention where a project note's `code` field points to the project's on-disk artifacts in `~/Documents/Projects/<code>/`.
With the above context file, an AI agent can also quickly locate these artifacts when needed.

The vault and those artifacts together are more or less every file I have gathered over the years.
An agent that can navigate the vault, and the artifacts behind it, therefore has access to almost every part of my professional life.
People often call Obsidian, or a similar system like Notion, a second brain.
Once an agent can read that brain and act on it by the same conventions I do, the result is very close to a Doppelgänger of my professional working self.

Pointing an agent at the root of my documents folder would technically give it the same access.
But without the vault as an index, it is far less efficient, and it tends to either take in too much noise or fail to find the full set of relevant files.
The vault and the context file hand it a structured map instead, so it can go from any project note straight to the files behind it.

