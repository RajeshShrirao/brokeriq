# BrokerIQ Roadmap

Status of the build (all milestones through s10 complete and committed).

## Done

- **s0** Scaffold — uv project, pyproject, gitignore, README stub
- **s1** Core package — config, logging, state/models, hello-world LangGraph agent
- **s2** GitHub repo pushed (public)
- **s3** Tools — web search, NAICS lookup, compliance RAG tool
- **s4** Full graph — supervisor + research/qualification/compliance/report agents,
  HITL interrupt
- **s5** RAG pipeline — Qdrant hybrid (BM42 + dense + RRF), cross-encoder rerank,
  Redis semantic cache
- **s6** Memory extraction loop on PostgresStore
- **s7** FastAPI SSE streaming + HITL resume API
- **s8** MCP 2.0 server exposing `compliance_search`
- **s9** Evals harness (offline FakeLLM + live LLM tiers) + promptfoo CI config
- **s10** GitHub Actions CI + Docker compose (qdrant / redis / postgres / api)
- **s11** Production docs — README, architecture, API contract, roadmap, final review

## Next up (candidate)

- Auth on the API (API keys / gateway) before real deployment
- Ingestion CLI + scheduler for the carrier corpus (watch a docs directory, re-index)
- More corpus: state-specific commercial auto, E&O, umbrella, workers-comp by class
- Live LLM-judge evals wired into CI (needs a provider key in GitHub secrets)
- Deployment: managed Postgres + Qdrant Cloud + Redis, uvicorn workers behind a load
  balancer, `brokeriq-mcp` as a hosted SSE/streamable-http MCP endpoint
- Feedback loop: real broker decisions on HITL reviews → retrain ICP thresholds
