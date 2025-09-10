import json
import pytest
from tests.llm_as_judge import evaluate_answer

# ---- Predefined prompts ----

PROMPTS =[
        "A lonely lighthouse keeper discovers a mysterious glowing pearl that washes ashore.",
        "A young witch accidentally turns her grumpy cat into a talking, flying broomstick.",
        "An astronaut gets stranded on a planet where the plants communicate through music.",
        "A detective in a steampunk city must solve the theft of a priceless clockwork heart.",
        "Two rival chefs must team up to win a magical cooking competition.",
        "A timid librarian finds a book that allows him to enter into the stories he reads.",
        "A group of kids builds a spaceship out of junk and actually travels to the moon.",
        "A knight who is afraid of dragons is tasked with saving a princess from one.",
        "An ancient robot wakes up in a post-apocalyptic world and tries to find its purpose.",
        "A musician discovers her guitar can control the weather when she plays certain chords."]

@pytest.mark.parametrize("query", PROMPTS)
def test_evaluate_answer_with_prompts(query, bedrock_env_ready, request):
    result = evaluate_answer(query)

    # save for summary reporting
    results = request.config.cache.get("evaluation_results", [])
    results.append((query, result))
    request.config.cache.set("evaluation_results", results)

    assert set(result.keys()) == {"Hallucination", "Completeness", "Relevancy"}
    for k, v in result.items():
        assert isinstance(v, int)
        assert 1 <= v <= 100