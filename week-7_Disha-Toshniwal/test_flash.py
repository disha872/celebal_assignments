import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv(override=True)
api_key = os.environ.get("GEMINI_API_KEY")

genai.configure(api_key=api_key)

print("--- Testing gemini-2.5-flash (without prefix) ---")
try:
    model = genai.GenerativeModel("gemini-2.5-flash")
    res = model.generate_content("Say hello.")
    print("Success without prefix:", res.text)
except Exception as e:
    print("Failed without prefix:", type(e), e)

print("\n--- Testing models/gemini-2.5-flash (with prefix) ---")
try:
    model = genai.GenerativeModel("models/gemini-2.5-flash")
    res = model.generate_content("Say hello.")
    print("Success with prefix:", res.text)
except Exception as e:
    print("Failed with prefix:", type(e), e)

print("\n--- Testing models/gemini-flash-latest ---")
try:
    model = genai.GenerativeModel("models/gemini-flash-latest")
    res = model.generate_content("Say hello.")
    print("Success with latest:", res.text)
except Exception as e:
    print("Failed with latest:", type(e), e)
