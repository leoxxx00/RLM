from llama_cpp import Llama

llm = Llama.from_pretrained(
    repo_id="QuantFactory/Meta-Llama-3-8B-Instruct-GGUF",
    filename="Meta-Llama-3-8B-Instruct.Q4_K_M.gguf",
    n_ctx=4096,
    n_threads=8,
    n_gpu_layers=-1,   # use Mac Metal GPU if available
    verbose=False,
)

messages = [
    {"role": "system", "content": "You are a helpful local AI assistant."}
]

print("Llama 3 8B Q4 chat started. Type 'exit' to quit.\n")

while True:
    user_input = input("You: ")

    if user_input.lower() in ["exit", "quit"]:
        break

    messages.append({"role": "user", "content": user_input})

    response = llm.create_chat_completion(
        messages=messages,
        temperature=0.7,
        max_tokens=512,
    )

    answer = response["choices"][0]["message"]["content"]
    print("\nLlama:", answer, "\n")

    messages.append({"role": "assistant", "content": answer})