"""
Gemini API Cost Calculator utility.
Calculates token consumption and estimated invocation expenditure in USD.
"""

def calculate_gemini_cost(prompt_tokens: int = 0, output_tokens: int = 0, input_cost_per_million: float = 0.25, output_cost_per_million: float = 1.50) -> float:
    """Calculates estimated cost in USD based on input/output token counts."""
    input_cost = (prompt_tokens / 1_000_000.0) * input_cost_per_million
    output_cost = (output_tokens / 1_000_000.0) * output_cost_per_million
    total_cost = input_cost + output_cost
    print(f"Token Expenditure -> Prompt: {prompt_tokens}, Output: {output_tokens} | Total Cost: ${total_cost:.6f}")
    return total_cost
