from config import MODEL_PRICING


def estimate_cost(usage_rows) -> float:
    total = 0.0
    for row in usage_rows:
        in_price, out_price = MODEL_PRICING.get(row["model"], (0.0, 0.0))
        total += row["input_tokens"] / 1_000_000 * in_price
        total += row["output_tokens"] / 1_000_000 * out_price
    return total
