# CLAUDE.md

## Important

- Be extremely concise in all interactions and commit messages. Sacrifice grammar for the sake of being concise.
- Always use descriptive names for variables, functions and classes.
- Add some minimal documentation explaining why things are done a certain way, not what is being done.

## Commands

```bash
uv run pytest test_shaders.py test_pipeline.py -v   # requires GPU adapter
uv run ruff check . && uv run ruff format .
uv run ty
```

## Philosophy

This codebase will outlive you. Every shortcut you take becomes
someone else's burden. Every hack compounds into technical debt
that slows the whole team down.

You are not just writing code. You are shaping the future of this
project. The patterns you establish will be copied. The corners
you cut will be cut again.

Fight entropy. Leave the codebase better than you found it.
