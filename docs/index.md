# aiida-agents

A natural-language, multi-agent interface to [AiiDA](https://www.aiida.net): describe what you want in plain language, and specialized agents call a controlled set of typed tools (via the Model Context Protocol) over a real AiiDA profile, grounded by retrieval over AiiDA documentation, with local models as a first-class target.

This started as a Google Summer of Code 2026 project under NumFOCUS / AiiDA.
It is **exploratory** — we're finding out what works first — but the goal is to grow it into a production-quality tool, not to stop at a prototype.

```{warning}
Early development: engineering scaffolding is in place and there is no functional release yet.
```

## Where to start

- **[Architecture](/docs/architecture.md)** — how a request travels through the system, and why the pieces are arranged as they are.
- **[Extending](/docs/extending.md)** — adding a tool, a documentation corpus, or a whole specialist; and how an AiiDA plugin contributes to the agents without either package depending on the other.
- **[Architecture Decision Records](/docs/adr/README.md)** — the reasoning behind each individual decision, including the ones that were later revised.
