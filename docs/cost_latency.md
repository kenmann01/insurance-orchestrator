# Cost and latency: measured traces

Source: 4 trace(s) from sequential four-call pipeline runs; these numbers come from live local qwen3:8b inference through Ollama.
Actual marginal cost per case: $0.00 (local inference). Hosted-equivalent USD applies HOSTED_RATES_PER_MTOK (input $3.00, output $15.00 per million tokens, Sonnet-class hosted rates from https://www.anthropic.com/pricing, as of 2026-10).

| case_id | applicant | tokens in | tokens out | latency ms (agents) | wall ms (pipeline) | hosted-equivalent USD |
| --- | --- | --- | --- | --- | --- | --- |
| 18c775c7867b | Dana Whitfield | 1169 | 326 | 18420.8 | 18421.2 | $0.008397 |
| bdbe757a971b | Elena Petrov | 1244 | 419 | 19242.4 | 19242.8 | $0.010017 |
| e33e3c42d769 | Marcus Boone | 1325 | 420 | 18843.3 | 18843.9 | $0.010275 |
| 707925117fb6 | Priya Natarajan | 1093 | 229 | 10716.3 | 10716.7 | $0.006714 |
| AVERAGE |  | 1208 | 348 | 16805.7 | 16806.2 | $0.008851 |

## Analysis

- Batch: at a measured 16806 ms of agent time per case, a sequential 1,000 case overnight batch needs about 4.7 h of pure pipeline runtime, which fits an overnight window with room to spare.
- Real-time verdict: the measured average of 16806 ms per case exceeds the 1-2 second (2000 ms) real-time quoting budget, so quotes should be answered asynchronously while the pipeline keeps running in the background.
- Slowest agent: risk_scoring_agent accounts for 32251 ms of the 67223 ms total agent time (48%); that step is the first parallelization or optimization lever. (With real external data sources, enrichment usually dominates because it waits on bureau and registry calls; here every agent is a local model call, so the ranking comes purely from these traces.)
- Cost: actual marginal cost is $0.00 per case because inference runs locally; the hosted-equivalent column shows what the same token volume would have cost at the cited hosted rates.
