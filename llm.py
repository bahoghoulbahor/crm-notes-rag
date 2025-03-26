from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_answer(context, query, model="gpt-3.5-turbo"):
    prompt = f"Use the following CRM notes to answer the question accurately and concisely:\n{context}\n\nQUESTION:\n{query}"

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are a CRM assistant. Each note includes a date, sales representative, and message."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        max_tokens=300,
        temperature=0.3
    )

    return completion.choices[0].message.content.strip()


if __name__ == "__main__":
    answer = generate_answer('none', 'none')
    print(answer)