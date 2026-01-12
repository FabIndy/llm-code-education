import subprocess

prompt = "Dis bonjour en français."

result = subprocess.run(
    ["ollama", "run", "mistral", prompt],
    capture_output=True,
    text=True
)

print(result.stdout)
