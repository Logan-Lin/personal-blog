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
This is a personal Obsidian vault used for project management.
It is plain-text, local-first, and synced with Syncthing.
The DataView plugin turns the whole vault into one queryable dataset, and `Dashboard.md` is the live entry point.

This context file is the current and authoritative convention for the topics it covers.
Some legacy notes might follow deprecated conventions, so prefer this file over patterns found in existing notes.

## Layout

- `Dashboard.md` is the human entry point.
  It is a set of live `dataviewjs` queries that render inside Obsidian for the user.
- `Projects/` holds one note per project, one thing the user works on.
  Filenames are the project title in human-readable Title Case.
  See [[#Frontmatter schemas]] for frontmatter fields.
- `Programs/` holds one note per program.
  A program is something a project belongs to, such as a paper venue, a funding call, or a course.
  Filenames are the program name.
  See [[#Frontmatter schemas]] for frontmatter fields.
- `Schedule/` holds the work log: daily notes named `YYYY-MM-DD.md` and weekly notes named `YYYY-Www.md` with a zero-padded ISO week number.
- `Drafts/` holds free-form working documents, for example, paper drafts, submission and review notes, rebuttals, job-application drafts, lecture notes, how-to and setup notes, brainstorms, abstracts.
- `People/` holds one note per person.
  Filenames are the person's full name.
- `_templates/` holds the note templates.
- `_attachments/` is the attachment sink for images.
- `_unsorted/` is the new-file inbox.
  It is configured as the default new-file location and is normally empty.
- `.obsidian/` holds config and plugins.
- `.trash/` is Obsidian's local trash.

## Tag taxonomy

Tags are the primary metadata.

Project notes carry exactly three tags, one from each axis:

- type:
  - `project/research` for work that produces new findings through models and experiments, usually aimed at a paper venue
  - `project/writing` for work whose deliverable is the document itself, such as applications, proposals, resource requests, the dissertation, lecture material, and ongoing logs
  - `project/development` for building software, tools, sites, or infrastructure
  - `project/teaching` for courses taught, student-project supervision, and teaching training
- priority:
  - `priority/main` for projects the owner leads and drives himself as the primary person responsible
  - `priority/collaborate` for projects where the owner supports as a co-author or supervisor rather than the primary driver
- status:
  - `status/ongoing` for projects under active work, with open milestones or recent activity toward a live target
  - `status/on-hold` for projects paused for now but meant to resume later, whether barely started or parked after a submission
  - `status/done` for finished projects whose milestones are all complete, often tied to an accepted venue or a delivered document
  - `status/discarded` for projects dropped and not meant to continue, including superseded directions and abandoned collaborations

Program notes carry exactly one tag:

- `program/venue` for conferences and journals
- `program/grant` for funding calls and research awards
- `program/education` for teaching, courses, and supervision
- `program/position` for job openings and position calls

Do not invent tag values outside these sets.

## Frontmatter schemas

Project note, field order `tags`, `code`, `program`, `people`, `deadline`:

- `code`: the project's codename, a kebab-case slug.
  Keep it lower case and separate every word with a hyphen.
  It maps to an on-disk working directory at `~/Documents/Projects/<code>/` that holds the project's non-note files, separate from the vault's own `Projects/` note folder.
  The note is usually created before that directory, so it is normal for the directory not to exist yet.
  Inside that directory the files are usually grouped under a purpose subfolder such as `code/` for source or `paper/` for the manuscript repository.
- `program`: the project's current target program when applicable, a quoted wikilink to a program note, for example `program: "[[Program Name]]"`.
- `people`: a quoted wikilink to a person note, or a YAML list of them, for example `people: "[[Name]]"` or a block list of `- "[[Name]]"` lines, when applicable.
  It records the people related to the project.
- `deadline`: a date `YYYY-MM-DD`.
  Rarely set on projects.
  By default, the dashboard inherits the deadline from the linked program in the `program` field, so set this field only when the project's own deadline differs from its program's.

Program note, field order `tags`, `code`, `deadline`:

- `code`: the program's codename, a kebab-case slug.
  Like a project's `code`, it maps to an on-disk working directory at `~/Documents/Programs/<code>/`, and that directory likewise may not exist yet.
- `deadline`: a date `YYYY-MM-DD`.

Schedule, Draft, and Person notes all have no YAML frontmatter.
A Schedule note references a project with a plain wikilink.
A Draft note is connected to a program or project by a wikilink to it in the program or the project note.

## Body conventions

The templates already lay out each note type's section headings.
The conventions below cover only what the templates do not show.

Project note: the `## Schedule` calendar block is a heatmap of the days worked on the project, built only from daily notes that link to it.
See the Schedule note convention below for the link that powers it.
Below the `---` rule, add free-form sections of content and wikilinks under topical headings.

Program note: the `## Related Projects` section is a reverse DataView query that lists every project linking to this program, whether through its `program` field or a wikilink in free-form content, so a project targeted at several programs over time appears under each.
Free-form notes follow the `---` rule.

Schedule note: it references related entries with plain wikilinks such as `[[Project Name]]`, and a wikilink up to a project is what powers that project's calendar heatmap.

Person note: the `## Related Projects` section is a reverse DataView query that lists every project linking to this person, mirroring the Program note's Related Projects query.
Below the `---` rule, the note is free-form and can contain any information related to the person.

In a note, use `##` and lower for every section heading, matching the templates.

## Dashboard mechanics

`Dashboard.md` is rendered for the end user, and its three `dataviewjs` blocks only produce results inside Obsidian.
Reading the file gives you the query code, not the answers.
The logic below is documented so you can reconstruct the same information by reading the source notes yourself when the user asks for it.

- TODO collects every incomplete task from the `TODO` section of `Schedule/` notes, the `Milestones` section of ongoing projects, and the `Milestones` section of programs.
- Deadline ranks ongoing projects by how soon they are due, inheriting a program's deadline when the project has none, and also lists programs with a future deadline.
- Projects lists all projects grouped by status with emoji headers, sorted by type then priority.

## Authoring rules

When creating or editing notes, follow the existing conventions.

- New project: copy the template file and edit that copy in place, for example `cp "_templates/Project.md" "Projects/<Title>.md"`.
  Set the three tags per [[#Tag taxonomy]] and the frontmatter fields per [[#Frontmatter schemas]], including a `code` slug if the project has or needs to have a working directory.
  Put actionable tasks under `## Milestones`.
- New program: copy `_templates/Program.md` the same way, for example `cp "_templates/Program.md" "Programs/<Name>.md"`.
  Set the one tag per [[#Tag taxonomy]] and the frontmatter fields per [[#Frontmatter schemas]].
- New person: copy `_templates/People.md` the same way, for example `cp "_templates/People.md" "People/<Name>.md"`, and name the file with the person's name.
  Because a person's information tends to change over time, prefer linking to an authoritative source for the person, such as their homepage or profile, rather than copying detailed information into the note, where it can later go out of date.
- Recording a day's work: edit or create `Schedule/YYYY-MM-DD.md`, keep the template's `##` sections, and place the day's entries under whichever ones fit, linking related project, person, program, or draft with a plain wikilink such as `[[Name]]`.
- Use the date format `YYYY-MM-DD` for any `deadline` and for daily note names.
- Write tasks and milestones as markdown checkboxes, `- [ ]` for open and `- [x]` for done.
  The Tasks plugin is installed, so annotate them with its inline metadata where useful, such as a `📅 YYYY-MM-DD` due date and a `✅ YYYY-MM-DD` completion date on a finished task.
- Drafts have no schema.
  Write the draft as plain markdown, and connect it to the vault as described for Draft notes in [[#Frontmatter schemas]].

## Plugins and config

Community plugins in `.obsidian/`: `dataview` (with JS queries enabled), `calendar` (provides the weekly notes), `obsidian-tasks-plugin` (the Tasks plugin, for checkbox task metadata), `obsidian-minimal-settings`, and `obsidian-vimrc-support`.
The theme is Minimal and vim mode is on.

Templates come from the core Templates plugin pointed at `_templates/`.
Daily notes come from the core Daily Notes plugin, configured with folder `Schedule` and template `_templates/Schedule`.
Weekly notes come from the Calendar plugin, also writing into `Schedule` with the same template.

## Context about the user

The vault owner is Yan Lin.
His person note `People/Yan Lin.md`, a.k.a. the [[Yan Lin]] note, is the entry point for facts about him.
Read it when you need such a fact.
It points on to the authoritative sources for his employment, education, research, and other activities.
```

For another level of laziness, adding the following sentences to Claude Code's global context means I do not need to manually point to the Obsidian vault in a conversation.

```markdown
A personal Obsidian vault at `~/Documents/app-state/obsidian` is the user's project-management vault.
It tracks his projects, their programs, his work log, and drafts, and is the authoritative source for facts about him.
Read its `CLAUDE.md` for the layout and where specific information is.
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

