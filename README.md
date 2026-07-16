# Personal Blog of Yan Lin

Static site built with [Zola](https://www.getzola.org/), served at `blog.yanlincs.com`.
Content is licensed CC BY-SA 4.0.

## Project Structure

- `config.toml`: site config
- `content/` holds posts grouped into three sections, each with an `_index.md`. A section sets its navbar abbreviation with `[extra] abbr`
- `templates/` holds Zola templates. `base.html` is the shared shell with head, nav, and footer. `page.html`, `section.html`, `index.html`, and `404.html` extend it
- `templates/shortcodes/` holds the custom shortcodes described below
- `sass/style.scss` is the single global stylesheet, compiled to `style.css`. It uses CSS variables with light and dark themes via `prefers-color-scheme`
- `static/` holds favicons and the web manifest, served at the site root
- `public/`: built output

## Stack and build

- The dev runtime is a Nix flake at `runtime/`
- `config.toml` holds the site config: base URL, gruvbox syntax highlighting, RSS feed, and Sass compilation
- CI in `.github/workflows/deploy.yml` runs `zola build` and deploys `public/` to Cloudflare Pages on every push to `main`

## Shortcodes

Image with max-width constraint. `width` defaults to `500px`:

```md
{{ img(src="./diagram.png", alt="Architecture", width="600px") }}
```

Figure caption:

```md
{% cap() %}The *architecture* diagram{% end %}
```

Block math:
 
```md
{% math() %}
\nabla \cdot \mathbf{E} = \frac{\rho}{\epsilon_0}
{% end %}
```

Inline math:

```md
The loss {% m() %}\mathcal{L}{% end %} is minimized.
```

Inserts table of contents where placed:

```md
{{ toc() }}
```

