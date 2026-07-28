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
When a vault task needs temporary or intermediate files that are not themselves vault content, such as helper scripts, source files, and tool output, keep them outside the vault in a scratch directory or a project working directory.

This context file is the current and authoritative convention for the topics it covers. Some legacy notes might follow deprecated conventions, so prefer this file over patterns found in existing notes.

## Layout

- `Dashboard.md` is the human entry point. It is a set of live `dataviewjs` queries that render inside Obsidian for the user
- `Projects/` holds one note per project, one thing the user works on. Filenames are the project title in human-readable Title Case. See [[#Project note]]
- `Programs/` holds one note per program. A program is something a project belongs to, such as a paper venue, a funding call, or a course. Filenames are the program name. See [[#Program note]]
- `Schedule/` holds the work log. It has daily notes named `YYYY-MM-DD.md` and weekly notes named `YYYY-Www.md` with an ISO week number padded with a leading zero. See [[#Schedule note]]
- `Drafts/` holds free-form working documents, for example, paper drafts, submission and review notes, rebuttals, job application drafts, lecture notes, how-to and setup notes, brainstorms, abstracts. See [[#Draft note]]
- `People/` holds one note per person. Filenames are the person's full name. See [[#Person note]]
- `Papers/` holds the library of academic papers, one PDF per paper, organized in a topic tree of subfolders. See [[#Papers library]]
- `Archive/` holds the user's personal life-admin documents, one file per document in a single flat folder. See [[#Archive library]]
- `_templates/` holds the note templates
- `_attachments/` is the attachment sink for images
- `_unsorted/` is the new file inbox. It is configured as the default location for new notes and notes in progress
- `.obsidian/` holds config and plugins
- `.trash/` is Obsidian's local trash

Git in this vault uses `.gitignore` to ignore Obsidian workspace state and all vault content, and only tracks vault configuration, note templates, and root metadata files.
When searching vault content, configure the chosen search tool to include files ignored by `.gitignore`.

## Shared convention

### Tags

Tags are the primary metadata.
Only project notes and program notes carry tags, and each type's section below defines the set it draws from.
Do not invent tag values outside those sets.

### TODO items

Write TODO items as markdown checkboxes, where the symbol inside the brackets sets the status.
The configured statuses are `- [ ]` todo, `- [x]` done, `- [/]` in progress, and `- [-]` cancelled.
The Tasks plugin is installed, so write a TODO's dates with its date emoji, including `🛫` start, `📅` due, `✅` done, and `❌` cancelled, each written as the emoji followed by a `YYYY-MM-DD` date, for example `📅 2026-06-24`.
When a TODO needs more detail, follow its checkbox with a `> [!todo]` callout block containing the detailed task description.

### Section heading

In a note, use `##` and lower for every section heading, matching the templates.
Never use a `#` heading in a note.

### Links

Internal links use Obsidian wikilinks, whether they point to another note, a file, or a section heading, for example `[[Name]]` or `[[#Title]]`.
Write every reference to a note or file as a wikilink, including each later mention of it, so a reference never falls back to plain text.
A wikilink names only the target note or file, not its full relative path, so to read or edit the target you first need to search for that file under the vault.
In the unlikely case that two files share the same name, a wikilink disambiguates by prepending enough of the parent folders to the name, such as `[[Folder/Note Name]]`.

To embed an image from `_attachments/` so it renders inline in the note, prefix its file wikilink with `!`, for example `![[Image Name.png]]`.

External links to a web URL use standard Markdown link syntax instead.

### Multimedia assets

Embed the following assets in a note when the user asks for them.

Obsidian supports the following text-based assets.

- Callout blocks, written as a blockquote whose first line is a `> [!type]` marker
- Tables, written as a markdown table with a header row and a delimiter row
- Mermaid diagrams, written inside a fenced mermaid code block
- Furigana pronunciation annotations for Chinese characters, written as `{漢字|かんじ}` for a whole word or `{漢字|かん|じ}` per character. The 漢字 part before the first `|` can only be Chinese characters, and the annotation parts after it can be kana, Zhuyin, Pinyin, or Hangul

When a note needs an asset that markdown itself cannot express, such as a data chart, a complex diagram with SVG elements, or any other figure, produce it with whatever toolset fits the asset and source data, and render the result to PNG.

### User interaction

A standard blockquote and a `> [!todo]` callout are the two means of interaction in this vault.
The user writes a comment in a standard blockquote, one without a `> [!type]` callout marker, to request a revision to the note.
The user writes a comment in a `> [!todo]` callout to request an action.

## Project note

To create a new project note, copy `_templates/Project.md` and edit that copy in place, for example `cp "_templates/Project.md" "Projects/<Title>.md"`.
Set the frontmatter fields per [[#Project frontmatter]], including a `code` slug if the project has or needs to have a working directory.
Put actionable tasks under `## TODO`.
Below the `---` rule, add free-form sections of content and wikilinks under topical headings.

### Project frontmatter

Field order `tags`, `code`, `program`, `people`, `deadline`.

`tags`: list of project tags, see [[#Project tags]].

`code`: the project's codename, a kebab-case slug. Keep it lower case and separate every word with a hyphen. It maps to a working directory on disk at `~/Documents/Projects/<code>/` that holds the project's non-note files. The note is usually created before that directory, so it is normal for the directory not to exist yet.

`program`: the project's current target program when applicable, a quoted wikilink to a program note, for example `program: "[[Program Name]]"`.

`people`: a quoted wikilink to a person note, or a YAML list of them, for example `people: "[[Name]]"` or a block list of `- "[[Name]]"` lines, when applicable. It records the people related to the project.

`deadline`: a date `YYYY-MM-DD`. Rarely set on projects. By default, the dashboard inherits the deadline from the linked program in the `program` field, so set this field only when the project's own deadline differs from its program's.

### Project tags

A project note carries exactly three tags, one from each axis.

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

## Program note

To create a new program note, copy `_templates/Program.md` and edit that copy in place, for example `cp "_templates/Program.md" "Programs/<Name>.md"`.
Set the frontmatter fields per [[#Program frontmatter]].
Free-form notes follow the `---` rule.

### Program frontmatter

Field order `tags`, `code`, `deadline`.

`tags`: program tag, see [[#Program tags]].

`code`: the program's codename, a kebab-case slug. Like a project's `code`, it maps to a working directory on disk at `~/Documents/Programs/<code>/`, and that directory likewise may not exist yet.

`deadline`: a date `YYYY-MM-DD`.

### Program tags

A program note carries exactly one tag.

- `program/venue` for conferences and journals
- `program/grant` for funding calls and research awards
- `program/education` for teaching, courses, and supervision
- `program/position` for job openings and position calls

## Schedule note

Schedule notes are usually left for the user to edit on his own, so do not edit one unless specifically instructed by the user.
Schedule notes have no YAML frontmatter and no tags.

A day's work is recorded in `Schedule/YYYY-MM-DD.md`.
Reference to a related project, person, program, or draft with a wikilink, which powers the referenced note's calendar heatmap.

## Draft note

Draft notes have no schema, YAML frontmatter, or tags.

Write the draft as plain markdown, and connect it to the vault by referencing a relevant entry with a plain wikilink.
A program, project, or person note also lists its direct child draft note with a plain wikilink in its free-form sections.

## Person note

Person notes have no YAML frontmatter and no tags.

To create a new person note, copy `_templates/People.md` and edit that copy in place, for example `cp "_templates/People.md" "People/<Name>.md"`, and name the file with the person's name.
Because a person's information tends to change over time, prefer linking to an authoritative source for the person, such as their homepage or profile, rather than copying detailed information into the note, where it can later go out of date.
Below the `---` rule, the note is free-form and can contain any information related to the person.

## Papers library

`Papers/` is a library of academic papers, separate from the note system.
Each paper is one PDF file, so, unlike the note folders, this folder holds no markdown and no frontmatter.
It is organized as a topic tree, and every paper sits in the deepest topic folder that fits it, for example `Papers/NLP/Agent/Memory/Long-Term Memory/`.

A paper's filename is `Author et al. - Year - Title.pdf`.
Use the first author's surname, drop the "et al." for a single author, and shorten an overlong title where it reads naturally.

When researching a topic to author a note in the vault, download the representative papers and file them into the library.
The note can then refer to them with a wikilink.

To file an incoming paper PDF, read its first few pages to get the title, author, and year, then rename it following the above filename convention.
Search the full topic tree from the top down, move the file into the deepest folder that fits, and create a new leaf folder only when nothing fits.

## Archive library

`Archive/` is a flat, chronological store of the user's personal life-admin documents, the records he keeps for reference, such as receipts and invoices, tax and bank statements, identity and immigration papers, housing and lease documents, travel tickets and bookings, employment and education records, insurance, and product manuals and warranties.

Every file is named `YYYY-MM-DD <description>.<ext>`.
The name starts with the document's own date in ISO form, then a single space, then a short human-readable description, and it keeps the file's real extension in lower case.
The leading date is the date the document itself is about, which is its issue or transaction date for a receipt, invoice, letter, or certificate, and its travel or event date for a ticket, boarding pass, or booking. The file system's created and modified dates can serve as a secondary reference.
When part of the date is genuinely unknown, use a placeholder for the unknown part instead of inventing a number.

The description says plainly what the document is, for example `Residence Permit`, `Tax Assessment 2024`, or `Anthropic Receipt 2231-3352`.
Keep a distinguishing detail when similar documents would otherwise collide, such as an invoice number, a travel route, or a property.

A note refers to an archived document with a wikilink to the file name, for example `[[2025-07-18 Anthropic Receipt 2231-3352.pdf]]`.

To file an incoming archive document, read the document to learn what it is and to find the proper date, then rename it following the above filename conventions and keep it directly in the flat `Archive/` folder.

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

