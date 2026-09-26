# Personal Blog of Yan Lin

Static site built with [Zola](https://www.getzola.org/), served at `blog.yanlincs.com`. Content is licensed [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).

## Project Structure

- `config.toml`: site config
- `content/`: posts grouped into three sections, each with an `_index.md`
- `templates/`: Zola templates
- `templates/shortcodes/`: the custom shortcodes described below
- `sass/style.scss`: the single global stylesheet, compiled to `style.css`
- `static/`: favicons and the web manifest
- `runtime/`: development runtime defined by a Nix flake 
- `public/`: built output
- `.github/workflows/deploy.yml`: CI for running `zola build` and deploying `public/` to Cloudflare Pages on every push to `main`

## Shortcodes

Image with a max-width constraint. `width` defaults to `500px`:

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

Inserts a table of contents where placed:

```md
{{ toc() }}
```

