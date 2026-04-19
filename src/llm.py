import os
import streamlit as st

PROVIDER = os.getenv("LLM_PROVIDER") or st.secrets.get("LLM_PROVIDER", "gemini")


def get_secret(name, default=None):
    return os.getenv(name) or st.secrets.get(name, default)


def get_response(prompt):
    provider = str(PROVIDER).lower()

    if provider == "openai":
        from openai import OpenAI

        client = OpenAI(api_key=get_secret("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content

    elif provider == "gemini":
        from google import genai

        client = genai.Client(api_key=get_secret("GEMINI_API_KEY"))
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        return response.text

    else:
        raise ValueError("Unsupported LLM_PROVIDER. Use 'openai' or 'gemini'.")