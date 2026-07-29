# Thesis Ideation Prompts (Optional)

Use this note when brainstorming experiments, proposing alternatives, or planning next research steps. This is intentionally **not** an always-on rule.

## Suggestion Style
- Think at a high level before implementing.
- If there is a better or simpler option, suggest it briefly.
- Keep suggestions concise and non-blocking.

## Candidate Experiment Directions
- Compare with-vs-without explicit goal-state reward using the same seed.
- Add intermediate rewards (or penalties) and verify policy preference shifts accordingly.
- Test shaped deadline rewards such as `(D - t_goal) / D`.
- Explore conservative heuristic variants and report whether guarantees are worst-case, confidence-based, or none.
- Validate on adversarial structures (including m-doors style cases) to expose heuristic bias.
