## Project Structure

- `config.toml`: site config (base_url, title, markdown settings)
- `content/`: blog posts organized by category (ml-tech, homelab, ai-system, dl4traj)
- `templates/`: Zola templates + shortcodes
- `sass/style.scss`: styles with light/dark mode via CSS variables
- `static/`: favicons, web manifest
- `public/`: built output

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

## Cloudflare Pages Deployment

To use a specific version of Zola, handle Cloudflare's distinction between preview and production builds, and include draft pages in preview builds:

```bash
curl -sL https://github.com/getzola/zola/releases/download/v0.22.1/zola-v0.22.1-x86_64-unknown-linux-gnu.tar.gz | tar xz && if [ "$CF_PAGES_BRANCH" = "main" ]; then ./zola build; else ./zola build --drafts --base-url $CF_PAGES_URL; fi
```

