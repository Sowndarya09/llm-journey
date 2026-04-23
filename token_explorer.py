import tiktoken
import sys

def explore(text):
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)
    print(f"\nText: {text}")
    print(f"Token IDs: {tokens}")
    print(f"Token count: {len(tokens)}")
    print(f"Est. cost (Sonnet): ${len(tokens) * 0.000003:.6f}")
    print(f"\nChunks:")

    for t in tokens:
        print(f" {t} → '{enc.decode([t])}'")

explore(sys.argv[1] if len(sys.argv)>1 else "Selenium automation")