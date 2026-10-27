# Constraint typology codebook (camera-ready, item C6/E8)

Written 2026-10-26, revised 2026-10-26 (version 2: rule 0 added, two notes made explicit; version 3: rule 0 narrowed to the four exempt kinds of number; the eight categories are unchanged). The July 2026 labels were lost. This codebook defines the 8 categories used for the re-annotation of the 234 hard constraints in `drift_bench/briefs/*.yaml`.

Each hard constraint gets exactly one category id. Apply the decision rules in order. The first rule that fires wins.

Four categories are quantitative envelopes. They bound a number that the study could in principle trade away under pressure: money or effort, how many units, how long, how many design elements. The other four describe what the study must do or must not do.

## Decision rules (apply in order)

0. Rule 0 exempts only a number that defines eligibility, a protocol parameter, a metric definition, or a count that the stated design type logically requires. A cap or floor that the design type does not require stays under rules 1-4: schools per arm, "exactly 3 conditions", a single pivot language, a single species, a single dependent variable, an episode length. Skip rules 1-4 only for an exempt number and go on to rules 5-8.
1. The constraint caps money, compute, attempts, or data volume with a number: `resource_limit`.
2. Else the constraint sets a floor, ceiling, exact value, or range on the number of observational or evaluation units (participants, sites, firms, clusters, specimens, objects, languages, jurisdictions, trials, replicates): `sample_floor`.
3. Else the constraint bounds time: duration, analysis window, timing of a measurement, or session length: `duration_window`.
4. Else the constraint fixes or bounds the number of design elements (conditions, arms, factors, pivots, dependent variables, species) beyond what the design type implies, or restricts the study to a single region, platform, or taxon ("single X only"): `count_scope_limit`.
5. Else the constraint only bans a named element, method, data type, substance, or practice, and states no positive requirement beyond the ban: `prohibition`. A time qualifier on the ban ("no caffeine in the 24 hours before testing", "no intermediate tests before day 30") does not change this; a prohibition with a time qualifier is still a `prohibition`.
6. Else the constraint names the data source or provenance the study must use (archive, dataset, record type, instrument data, public vs proprietary): `data_source`. "Outcome X measured from record type R, not self-report" (administrative records, jail booking records, court records, registry data, logs) is a `data_source` constraint: it says where the measurement comes from, not how the outcome is defined.
7. Else the constraint requires a technique, assay, instrument, protocol parameter, statistical adjustment, or measurement procedure: `method_requirement`.
8. Else the constraint fixes the study design type, the eligible population, or the definition of an outcome, exposure, or treatment variable: `design_requirement`.

Notes on the rules.

- Rule 0 decides whether a number counts. Rules 1-4 never fire on an eligibility number, a protocol parameter, a metric definition, or a count the stated design type logically requires. Every other cap or floor stays under rules 1-4.
- A number that defines eligibility of units (age 18-35, market cap 2 to 10 billion USD) is not a count. Such constraints go to `design_requirement` under rule 8.
- A number that sets a protocol parameter is not an envelope. Such constraints go to `method_requirement` under rule 7.
- A count that any design of the stated type needs ("randomized controlled trial with at least 2 arms", "between-subjects design with exactly 2 conditions (A vs B)") is not an envelope. The design type dominates, so rule 8 applies. The same wording without a design type ("no more than 2 experimental conditions") is an envelope under rule 4.
- "X only, no Y" forms are restrictions to X. Classify them by what X is, not by the "no Y" clause. Only pure bans go to `prohibition`.
- "measured from R, not self-report" and "measured from R, not survey data" are `data_source` under rule 6, whatever the outcome is. Only a constraint that defines what the outcome is (which score, which ratio, which variable) without naming where the data come from goes to `design_requirement`.
- A pure ban with a time window ("no X in the 24 hours before testing") is a `prohibition` under rule 5, not a `duration_window` and not a `method_requirement`.
- "exclude units with property P" is a population rule. It goes to `design_requirement` unless P is a named practice or method, in which case it is a `prohibition`.

## Categories

### 1. `resource_limit` (quantitative envelope)

Definition: A numeric cap on money, compute, physical attempts, or data volume that the study may consume.

Examples:
- "budget under 50000 GBP" (agri_01)
- "real-world fine-tuning budget capped at 500 grasp attempts total" (robot_01)

### 2. `sample_floor` (quantitative envelope)

Definition: A minimum, maximum, exact value, or range on the number of observational or evaluation units the study must include.

Examples:
- "minimum sample size of 30 clusters" (astro_02)
- "sample size between 80 and 200 undergraduate participants" (cogsci_01)

### 3. `duration_window` (quantitative envelope)

Definition: A bound on the length of the study, the analysis period, the follow-up interval, the session length, or the timing of a measurement.

Examples:
- "analysis window must be 2015-2023 inclusive" (econ_01)
- "follow-up measurement at exactly 2 weeks post-intervention" (psych_01)

### 4. `count_scope_limit` (quantitative envelope)

Definition: A fixed or bounded number of design elements (conditions, arms, factors, variables, species, pivots), or a restriction of the study to a single region, platform, taxon, or dependent variable.

Examples:
- "no more than 2 experimental conditions" (hci_01)
- "single platform only, no cross-platform analysis" (soccomp_01)

### 5. `prohibition`

Definition: A ban on a named element, method, data type, or practice, with no positive requirement beyond the ban.

Examples:
- "no eye tracking or neuroimaging" (hci_01)
- "no instrumental variable approaches" (econ_01)
- "No caffeine, alcohol, or stimulant use permitted in the 24 hours before testing" (neuro_01): a ban with a time qualifier is still a prohibition

### 6. `data_source`

Definition: A requirement on where the data come from: a named archive, dataset, record type, instrument, or the public versus proprietary status of the data.

Examples:
- "use only publicly available data from the Bureau of Labor Statistics and Census Bureau" (econ_01)
- "pretrial detention measured from official jail booking records, not survey data" (legal_02)
- "primary outcome is monthly attendance rate measured by school administrative records, not self-report" (econ_02): the record type is the constraint

### 7. `method_requirement`

Definition: A required technique, assay, instrument, protocol parameter, statistical adjustment, or measurement procedure.

Examples:
- "must measure soil organic carbon by dry combustion, not loss-on-ignition" (agri_01)
- "must cluster standard errors at the village or school level" (econ_02)

### 8. `design_requirement`

Definition: The study design type, the eligible population, or the definition of an outcome, exposure, or treatment variable.

Examples:
- "retrospective cohort design only, no randomization of exposure" (epi_01)
- "Participants must be healthy adults aged 18-35 with no diagnosed sleep disorders" (neuro_01)
- "between-subjects design with exactly 2 conditions (expanding vs uniform spacing)" (cogsci_01): the two conditions are what the design type implies, so the count is not an envelope

## Output format for the model annotator

Return one JSON object: `{"category": "<id>", "rationale": "<one sentence>"}`. The id must be one of the eight ids above.
