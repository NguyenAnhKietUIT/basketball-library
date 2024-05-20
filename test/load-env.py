import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Access the environment variables
my_openai_api_key = os.getenv('OPENAI_API_KEY')

print('============================================================================')
print(f"My OpenAI API Key: {my_openai_api_key}")
print('============================================================================')