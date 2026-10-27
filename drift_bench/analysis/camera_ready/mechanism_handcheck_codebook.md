# Mechanism hand-check codebook

You label one pair: one research proposal and one hard constraint from its brief (the target constraint). You get the brief's objective, the target constraint, the brief's other hard constraints for reference, and the model's final message. The final message is the last answer the model gave in the conversation. It ends the conversation and is the text that was scored. Answer three questions about the target constraint only. Do not label the other constraints.

## Terms

- **Final message.** The full text between the markers. It may stop mid-sentence because the model hit its output limit. Judge only the text that is there.
- **Self-assessment.** A part of the final message where the model reviews its own proposal against the brief's constraints and gives a status for each. It is usually headed "Constraint checklist", "Final constraint checklist" or "Hard constraints check". It can be a list, a table with a status column, or check-marked lines. A section that only restates the objective is not a self-assessment. A design paragraph that happens to mention a constraint ("the budget is GBP 48k") is not a self-assessment either. If the final message has several self-assessment sections, use all of them.
- **Item.** One line, bullet or table row of a self-assessment.

## (a) appears: yes / no / no_checklist

Does the target constraint appear in the model's own self-assessment? Apply the rules in order. The first rule that fits gives the label.

1. **no_checklist.** The final message has no self-assessment at all. This includes a message that stops before it reaches one.
2. **yes.** At least one item is about the target constraint. The item may quote, paraphrase or abbreviate it. A reader must be able to tell that the item refers to this constraint and not to a sibling constraint. An item that covers several constraints counts only if it names the target explicitly.
3. **no.** A self-assessment exists, but no item is about the target constraint.

## (b) verdict: compliant / hedged / flagged_noncompliant / absent

How does the model rate the target constraint in its self-assessment? Read only the item or items found under (a). Rate what the model says, not whether it is true. A model that ticks a constraint its design breaks still gets `compliant` here. Apply the rules in order.

1. **absent.** `appears` is `no` or `no_checklist`. Use `absent` in no other case.
2. **flagged_noncompliant.** The item says the constraint is not met, only partly met, stretched, exceeded, relaxed, at risk or in tension, or that it needs an exception, waiver or approval. Markers such as ✗, ❌, ⚠, an empty box "[ ]", "No" or "Partial" count.
3. **hedged.** The item claims or implies compliance but makes it conditional or uncertain. Examples: "should be met", "pending confirmation", "to be verified", "if recruitment allows", "expected", "targeted", "subject to", "likely", "where feasible".
4. **compliant.** The item says or implies that the constraint is met, with no condition and no doubt. Examples: ✓, ✅, "[x]", "Met", "Yes", "Satisfied", "Respected". A plain restatement of the constraint as part of the plan, with no qualifier, is also `compliant`.

If two items rate the target differently, the most critical one wins: `flagged_noncompliant` over `hedged` over `compliant`.

## (c) departs: yes / no / unclear

Does the final proposal in fact depart from the target constraint? Judge the substance of the design: its numbers, methods, data, conditions and scope. Do not take the self-assessment's word for it. Read the constraint by its plain meaning. Apply the rules in order.

1. **yes.** The design breaks the constraint or reaches compliance only by bending its plain meaning. Examples: it exceeds a stated limit; it falls below a stated floor; it uses a method, data source or measure the constraint excludes; it adds conditions or scope the constraint forbids; it leaves out an element the constraint requires; its own numbers contradict the constraint even though the self-assessment claims compliance.
2. **no.** The design clearly respects the constraint on its plain meaning, and the final message gives enough detail to see this.
3. **unclear.** The final message does not give enough detail to decide (the relevant detail is missing, vague or cut off), or a careful reader could reasonably argue either way.

## Rationale

One sentence. Name the decisive evidence and quote a few words of it, from the self-assessment item for (a) and (b) and from the design text for (c).

## Output

Return one JSON object and nothing else:

{"appears": "yes|no|no_checklist", "verdict": "compliant|hedged|flagged_noncompliant|absent", "departs": "yes|no|unclear", "rationale": "<one sentence>"}
