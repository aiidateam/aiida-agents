[![Templated from python-copier](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/mbercx/python-copier/refs/heads/main/docs/img/badge.json)](https://github.com/mbercx/python-copier)

# `aiida-agents`

> ⚠️ **Alpha — work in progress.**
> This is an early, exploratory project: there is **no functional release yet**, the architecture is still being decided, and APIs may change.
> Not production-ready; do not depend on it.

A natural-language, multi-agent interface to [AiiDA](https://www.aiida.net).

The goal is to let researchers create, run, inspect, and diagnose AiiDA workflows by describing what they want in plain language.
Specialized agents call a controlled set of typed tools (via the Model Context Protocol) over a real AiiDA profile, grounded by retrieval over AiiDA documentation.
Local models are a first-class target, not an afterthought.

This started as a [Google Summer of Code 2026](https://summerofcode.withgoogle.com/) project under NumFOCUS / AiiDA.
It is exploratory — we're finding out what works first — but the goal is to grow it into a production-quality tool, not to stop at a prototype.

## Status

**Alpha / work in progress.**
Engineering scaffolding (tests, CI, linting, typing) is in place; feature work has not started.
The architecture is being decided incrementally — read the design records before relying on anything here.

## Where to look

- **Project plan:** [`docs/gsoc/project-timeline.md`](/docs/gsoc/project-timeline.md) — phases and milestones.
- **Architecture decisions:** [`docs/adr/`](/docs/adr/) — the ADR log; start with [ADR-01](/docs/adr/01-package-scaffolding.md) (scaffolding).
- **Contributing / local setup:** [`CONTRIBUTING.md`](/CONTRIBUTING.md) — the `uv`-based development workflow.

## License

MIT. See [`LICENSE`](/LICENSE).
