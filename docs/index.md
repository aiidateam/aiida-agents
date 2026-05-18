# aiida-agents

A natural-language, multi-agent interface to [AiiDA](https://www.aiida.net): describe what you want in plain language, and specialized agents call a controlled set of typed tools (via the Model Context Protocol) over a real AiiDA profile, grounded by retrieval over AiiDA documentation, with local models as a first-class target.

This started as a Google Summer of Code 2026 project under NumFOCUS / AiiDA.
It is **exploratory** — we're finding out what works first — but the goal is to grow it into a production-quality tool, not to stop at a prototype.

```{warning}
Early development: engineering scaffolding is in place, there is no functional release yet, and the architecture is still being decided.
See the Architecture Decision Records in `docs/adr/` for the current design state.
```
