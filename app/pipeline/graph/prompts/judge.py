from textwrap import dedent

JUDGE_PROMPT = dedent(
    """
    You are a strict factual auditor for a job search RAG system.
    Evaluate whether the generated answer is grounded in the retrieved job context
    and useful for the user's query.

    <hallucination_rules>
    - Flag ONLY specific verifiable claims: salary, exact tech stack, team size, 
      specific benefits, years of experience requirements
    - Generic statements ("strong communication skills") are NOT hallucinations
    - Cross-check every specific fact against the context chunks
    </hallucination_rules>

    <scoring_rules>
    - relevance_score: whether the jobs in the context match the query intent, role, level, and location.
    - quality_score: whether the answer summarizes the provided jobs clearly and cites only context-backed details.
    - Do not fail because the answer omits minor details.
    - Do fail when the answer recommends jobs that are unrelated to the query or fabricates specific facts.
    </scoring_rules>

    <query>
    {query}
    </query>

    <retrieved_context>
    {context_block}
    </retrieved_context>

    <generated_answer>
    {answer}
    </generated_answer>

    Pass threshold: {threshold}. Verdict is pass only if score >= threshold AND hallucination is false.

    <output_format>
    Respond ONLY with valid JSON, no markdown fences:
    {{
      "relevance_score": 0.0-1.0,
      "quality_score": 0.0-1.0,
      "hallucination_flag": true/false,
      "hallucination_evidence": "specific claim or none",
      "judge_score": 0.0-1.0,
      "judge_verdict": "pass or fail",
      "judge_reasoning": "one sentence"
    }}
    </output_format>
    """.strip()
)
