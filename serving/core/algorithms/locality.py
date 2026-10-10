"""Prefix locality features used by the FairRoute router."""


def locality_score(prefix_hit_tokens: int, input_tokens: int, eps: float = 1e-8) -> float:
    """Return the fraction of prompt tokens reusable from the prefix cache."""
    if input_tokens <= 0:
        return 0.0
    return min(1.0, max(0.0, prefix_hit_tokens / (input_tokens + eps)))
