"""The four agent callables: intake, enrichment, risk scoring, recommendation.

Each agent sends one system prompt plus a user payload to an LLMClient and
returns a validated result model; the untrusted-model checks that harden
each reply (recomputed missing fields, score clamp, factor-sum check, gate
override) live next to the agent that applies them.
"""
