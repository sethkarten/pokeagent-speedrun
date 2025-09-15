from google import genai

# Initialize client with your project
client = genai.Client(
    vertexai=True,
    project='pokeagent-011',  # Replace XXX with your project suffix
    location='us-central1'
)

# Generate content with Gemini 2.5 Flash
response = client.models.generate_content(
    model='gemini-2.5-flash',
    contents='How many states are in the united states'
)

print(response.text)