## Project Structure

- `config.toml`: site config
- `content/`: blog posts organized by category
- `templates/`: Zola templates and shortcodes
- `sass/style.scss`: global styles
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

