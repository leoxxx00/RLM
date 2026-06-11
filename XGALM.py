
import os
import random
import re

from llama_cpp import Llama


POPULATION_SIZE = 6
GENERATIONS = 3
KEEP_TOP = 2
RANDOM_SEED = 7

P_COUNTER_ENV = "P_NUMERIC_VALUE"
P_TEXT_ENV = "P_TEXT_VALUE"


llm = Llama.from_pretrained(
    repo_id="QuantFactory/Meta-Llama-3-8B-Instruct-GGUF",
    filename="Meta-Llama-3-8B-Instruct.Q4_K_M.gguf",
    n_ctx=4096,
    n_threads=8,
    n_gpu_layers=-1,  # use Mac Metal GPU if available
    verbose=False,
)


def ask_llm(messages, temperature=0.7, max_tokens=512):
    response = llm.create_chat_completion(
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response["choices"][0]["message"]["content"].strip()


def store_p_in_environment(P):
    P_numeric_value = int(os.environ.get(P_COUNTER_ENV, "0")) + 1
    os.environ[P_COUNTER_ENV] = str(P_numeric_value)
    os.environ[P_TEXT_ENV] = P
    return P_numeric_value


def get_prompt_environment_block():
    return (
        "External prompt environment:\n"
        f"- P_NUMERIC_VALUE={os.environ.get(P_COUNTER_ENV, '0')}\n"
        f"- P_TEXT_VALUE={os.environ.get(P_TEXT_ENV, '')}"
    )


def seed_prompts(P):
    env_block = get_prompt_environment_block()

    return [
        f"{env_block}\n\nUse P_TEXT_VALUE as the user's real request. Do not mention P_TEXT_VALUE or the environment. Answer clearly and directly.",
        f"{env_block}\n\nUse P_TEXT_VALUE as the user's real request. Do not explain the variable. Give the best concise answer.",
        f"{env_block}\n\nUse P_TEXT_VALUE as the user's real request. Act as a careful expert assistant and answer with useful detail.",
        f"{env_block}\n\nUse P_TEXT_VALUE as the user's real request. Rewrite it internally into a precise task, then answer only the user.",
        f"{env_block}\n\nUse P_TEXT_VALUE as the user's real request. Give a practical high-quality response. Include examples only if useful.",
        f"{env_block}\n\nUse P_TEXT_VALUE as the user's real request. Find the user's intent and answer naturally.",
    ]


def mutate_prompt(prompt):
    mutations = [
        "Be specific and avoid vague language.",
        "Use a clean structure with short paragraphs.",
        "Check assumptions before giving the answer.",
        "Prefer practical steps over theory.",
        "Include the final answer first, then supporting details.",
        "Make the answer easy for a beginner to follow.",
        "Mention any risks, limits, or missing information only if useful.",
        "Use examples only where they improve the answer.",
        "Do not mention P_TEXT_VALUE, P_NUMERIC_VALUE, or the external environment.",
        "Answer naturally as if replying directly to the user.",
    ]

    action = random.choice(["append", "prepend", "replace"])

    if action == "append":
        return f"{prompt}\n\nExtra instruction: {random.choice(mutations)}"

    if action == "prepend":
        return f"{random.choice(mutations)}\n\n{prompt}"

    lines = prompt.splitlines()

    if not lines:
        return prompt

    lines[random.randrange(len(lines))] = random.choice(mutations)
    return "\n".join(lines)


def crossover_prompt(parent_a, parent_b):
    a_lines = parent_a.splitlines()
    b_lines = parent_b.splitlines()

    a_cut = max(1, len(a_lines) // 2)
    b_cut = max(1, len(b_lines) // 2)

    return "\n".join(a_lines[:a_cut] + b_lines[b_cut:])


def score_prompt(P, candidate_prompt):
    judge_messages = [
        {
            "role": "system",
            "content": (
                "You are a strict prompt quality judge. Score how well the candidate "
                "prompt will answer the user's request. Prefer prompts that answer "
                "naturally and do not expose internal variables. Return only a number "
                "from 0 to 100."
            ),
        },
        {
            "role": "user",
            "content": f"User request:\n{P}\n\nCandidate prompt:\n{candidate_prompt}",
        },
    ]

    raw_score = ask_llm(
        judge_messages,
        temperature=0.1,
        max_tokens=16,
    )

    match = re.search(r"\d+(?:\.\d+)?", raw_score)

    if not match:
        return 0.0

    return max(0.0, min(100.0, float(match.group())))


def evolve_prompt(P):
    population = seed_prompts(P)

    for generation in range(1, GENERATIONS + 1):
        print(f"\n--- Generation {generation} ---")
        scored = []

        for index, prompt in enumerate(population, start=1):
            score = score_prompt(P, prompt)
            scored.append((score, prompt))

            preview = prompt.replace("\n", " ")[:160]
            print(f"[G{generation} P{index}] score={score:.1f} prompt={preview}")

        scored.sort(reverse=True, key=lambda item: item[0])

        survivors = [prompt for _, prompt in scored[:KEEP_TOP]]
        print(f"Best score this generation: {scored[0][0]:.1f}")

        next_population = survivors[:]

        while len(next_population) < POPULATION_SIZE:
            parent_a, parent_b = random.sample(survivors, 2)
            child = crossover_prompt(parent_a, parent_b)
            child = mutate_prompt(child)
            next_population.append(child)

        population = next_population

    final_scored = [(score_prompt(P, prompt), prompt) for prompt in population]
    final_scored.sort(reverse=True, key=lambda item: item[0])

    return final_scored[0]


messages = [
    {
        "role": "system",
        "content": (
            "You are a helpful local Llama 3 assistant. Answer naturally. "
            "Never mention internal prompt variables unless the user asks about them."
        ),
    }
]

random.seed(RANDOM_SEED)

print("Llama 3 8B Q4 genetic prompt chat started. Type 'exit' to quit.\n")
print(f"Population={POPULATION_SIZE}, generations={GENERATIONS}, keep_top={KEEP_TOP}\n")

while True:
    P = input("You: ")

    if P.lower() in ["exit", "quit"]:
        break

    P_numeric_value = store_p_in_environment(P)

    print("\n=== External Environment Values ===")
    print(f"{P_COUNTER_ENV}={P_numeric_value}")
    print(f"{P_TEXT_ENV}={os.environ[P_TEXT_ENV]}")

    best_score, best_prompt = evolve_prompt(P)
    updated_prompt_value = best_prompt

    print("\n=== Updated Prompt Value ===")
    print(f"Score: {best_score:.1f}")
    print(updated_prompt_value)

    final_prompt = (
        updated_prompt_value
        + "\n\nImportant: Answer the user naturally. Do not mention "
        "P_TEXT_VALUE, P_NUMERIC_VALUE, or the external environment."
    )

    answer_messages = messages + [
        {
            "role": "user",
            "content": final_prompt,
        }
    ]

    answer = ask_llm(
        answer_messages,
        temperature=0.7,
        max_tokens=512,
    )

    print("\n=== Final Answer ===")
    print(answer, "\n")

    messages.append({"role": "user", "content": P})
    messages.append({"role": "assistant", "content": answer})
