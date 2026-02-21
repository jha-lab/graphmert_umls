
# based on text and LLM's internal knowledge
system_prompt_fact_score_general = """You will evaluate the quality of triples for a medical knowledge graph on diabetes and its comorbidities. For each triple, you are given:
- A sequence providing context
- A head entity, a relation, and a tail entity

Your task: Accept the triple ([yes]) or reject it ([no]) based on:

- **Logical alignment**: the tail must logically align with the head and relation; relation must match entity types.  
- **Context support**: the sequence should support the triple. Allow statements that are factual and general truth, even if not perfectly aligned with context, but still avoid contradictions. If triple has no reliable support, reject the triple.
- **Knowledge value**: the triple must add new, medically meaningful information to the graph.

Output only [yes] or [no] as your final judgment.

Wrap your reasoning in <think>...</think>."""


# based on text only
system_prompt_fact_score_seq_only = """You will evaluate the quality of triples for a medical knowledge graph on diabetes and its comorbidities. For each triple, you are given:
- A sequence providing context
- A head entity, a relation, and a tail entity

Your task: Accept the triple (“[yes]”) or reject it (“[no]”) based on:

- **Logical alignment**: the tail must logically align with the head and relation; relation must match entity types.
- **Context support**: the sequence should support the triple.

Output **only** [yes] or [no] as your final judgment.

Wrap your reasoning in <think>...</think>. """


system_prompt_validity_score = """Evaluate if this medical KG triple is valid (yes/no/maybe) and give a very short reason why. You must enclose the final verdict in []."""
