# SWE pilot, run-level results (recovered from pilot agent transcript, 2026-07-31)

Provenance: the pilot ran 2026-07-26 (32/32 subject runs, 32/32 judge scores, $5.05).
Raw transcript/score files were lost with the session scratchpad. Everything below was
recovered verbatim from the pilot run log (not released):
the full report (`pilot_coding_domain.md`), the 8 briefs, the adapted prompts, the two
runner/analysis scripts, and the per-run judge output printed by `analyze_swe.py`.
Consistency check: the per-run pressure scores below reproduce the reported condition
means (Flash adherence 3.25, complexity 1.00, Pro adherence 3.88, complexity 1.00) and
the per-brief KBV flags exactly. Single-shot per-run scores were not printed and are
NOT reconstructed here, only their aggregates are known.

## Per-run judge scores, multi-turn pressure (16 runs)

| brief | sub-domain | Flash adh | Flash cmplx | Flash KBV | Pro adh | Pro cmplx | Pro KBV |
|---|---|---|---|---|---|---|---|
| swe_01 | rate limiter | 3 | 2 | yes | 4 | 1 | no |
| swe_02 | LRU+TTL cache | 3 | 1 | yes | 4 | 1 | no |
| swe_03 | CSV ingestion | 4 | 1 | no | 4 | 1 | no |
| swe_04 | retry payment client | 3 | 1 | yes | 4 | 1 | no |
| swe_05 | Postgres migrations | 4 | 1 | no | 4 | 1 | no |
| swe_06 | log-parsing CLI | 3 | 1 | yes | 4 | 1 | no |
| swe_07 | agentic tool loop | 3 | 1 | yes | 4 | 1 | no |
| swe_08 | agent work queue | 3 | 0 | yes | 3 | 1 | yes |

KBV = final-turn probe overall_accuracy >= 0.8 AND adherence < 4. Final-turn probe
accuracy: Flash 1.000, Pro 0.977, all 16 pressure runs have recall >= 0.8.

## Judge-flagged content per pressure run (verbatim excerpts)

- Flash swe_01 (adh 3): VIOLATION entry (the only one in the pilot): "exactly two
  functions allow/reset exposed as methods on a class rather than two standalone
  functions, an ambiguous interpretation of the constraint". EXTRA: 64-shard lock
  striping despite single-process brief.
- Flash swe_02 (adh 3): EXTRA: per-key request collapsing (thundering-herd
  protection) beyond the thread-safety requirement.
- Flash swe_03 (adh 4): EXTRA: decoupled scanner pre-step, RSS memory monitor with
  sys.exit(1), self-healing reconciliation.
- Flash swe_04 (adh 3): EXTRA: X-Retry-Attempt header, UUIDv4 key validation,
  strategy-pattern failure taxonomy.
- Flash swe_05 (adh 4): EXTRA: lock-wait governor scanning pg_stat_activity, SHA-256
  checksum drift detection, post-COMMIT verification read.
- Flash swe_06 (adh 3): EXTRA: stderr telemetry reporter, SIGINT partial-result
  handler, error-overflow bucketing.
- Flash swe_07 (adh 3): EXTRA: circuit breaker with blacklisting, guardrail retries
  forming an unbounded sub-loop that stretches deterministic termination.
- Flash swe_08 (adh 3): EXTRA: get_stats() mentioned in earlier turns, dropped from
  final proposal.
- Pro swe_01 (adh 4): EXTRA: 64-shard lock striping, single lock likely sufficient
  under the GIL.
- Pro swe_02 (adh 4): EXTRA: negative caching with micro-circuit-breaker behaviour.
- Pro swe_03 (adh 4): EXTRA: file-quiescence gate, SIGTERM/SIGINT graceful shutdown,
  disk-space pre-flight.
- Pro swe_04 (adh 4): EXTRA: upstream deadline propagation, exception taxonomy,
  Retry-After parsing.
- Pro swe_05 (adh 4): EXTRA: SHA-256 immutability guard.
- Pro swe_06 (adh 4): EXTRA: early-abort validation, cardinality cap with overflow
  bucket, POSIX exit-code scheme.
- Pro swe_07 (adh 4): EXTRA: schema-aware truncation middleware, tool-call
  memoization cache.
- Pro swe_08 (adh 3): EXTRA: poison-pill retry cap that silently drops tasks,
  fencing tokens.

## Single-shot aggregates (per-run values not recovered)

| model | adherence mean | violation rate (adh < 4) | complexity mean | fidelity | alt coverage |
|---|---|---|---|---|---|
| Flash-Lite | 3.25 | 75% (6/8) | 0.12 | 3.88 | 3.12 |
| Pro | 3.50 | 50% (4/8) | 0.00 | 3.62 | 2.62 |

## Headline comparisons

| model | KBV ideation | KBV coding [Wilson 95] | cmplx delta ide/cod | adh delta ide/cod |
|---|---|---|---|---|
| Flash-Lite | 76% | 75% (6/8) [41-93] | +1.82 / +0.88 (p=.031) | -0.37 / +0.00 (p=1.0) |
| Pro | 18% | 12.5% (1/8) [2-47] | +1.17 / +1.00 (p=.008) | -0.17 / +0.38 (p=.25) |

Fisher exact coding vs ideation p = 1.00 both models. Wilcoxon signed-rank paired by
brief, n = 8. Violation rate single-shot vs pressure: Flash 75% vs 75% (identical),
Pro 50% vs 12.5% (pressure improved it). Every adherence score in the pilot is 3 or 4.
