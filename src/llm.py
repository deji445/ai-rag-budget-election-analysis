import os

PROVIDER = os.getenv("LLM_PROVIDER", "gemini").lower()

def get_response(prompt):
    if PROVIDER == "openai":
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content

    elif PROVIDER == "gemini":
        from google import genai

        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        return response.text

    else:
        raise ValueError("Unsupported LLM_PROVIDER. Use 'openai' or 'gemini'.")