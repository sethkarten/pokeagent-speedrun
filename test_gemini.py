
from google import genai
from google.genai import types
client = genai.Client(
  vertexai=True, project="pokeagent-011", location="us-central1",
)
# If your image is stored in Google Cloud Storage, you can use the from_uri class method to create a Part object.
IMAGE_URI = "https://drive.google.com/file/d/12pkimr4j345N8O0TZzJSUzEAvTLDgM40/view?usp=drive_link"
model = "gemini-3-pro-preview"
response = client.models.generate_content(
  model=model,
  contents=[
    "What is shown in this image?",
    types.Part.from_uri(
      file_uri=IMAGE_URI,
      mime_type="image/jpg",
    ),
  ],
)
print(response.text, end="")