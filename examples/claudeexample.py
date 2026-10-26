import json
import os

import requests
from dotenv import load_dotenv

load_dotenv()

api_key = os.environ.get("OPENROUTER_API_KEY")
if not api_key:
    raise RuntimeError("OPENROUTER_API_KEY is missing. Add it to your .env file.")

url = "https://openrouter.ai/api/v1/messages"
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json",
    # Adds OpenRouter routing/provider details to the response when available.
    "X-OpenRouter-Metadata": "enabled",
}

model = "nvidia/nemotron-3.5-super-vl:free"

# This is a local Python function exposed to the model as a tool. The model
# requests it; this script executes it and sends the result back to the model.
def count_letter(text: str, letter: str) -> dict:
    return {"text": text, "letter": letter, "count": text.lower().count(letter.lower())}


tools = [
    {
        "name": "count_letter",
        "description": "Count how many times a letter occurs in a piece of text.",
        "input_schema": {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "letter": {"type": "string"},
            },
            "required": ["text", "letter"],
            "additionalProperties": False,
        },
    }
]

messages = [
    {
        "role": "user",
        "content": (
            "Use the count_letter tool to count the letter 'r' in 'strawberry'. "
            "Then tell me the count and identify the model answering."
        ),
    }
]


def send_message(payload: dict) -> dict:
    response = requests.post(url, headers=headers, json=payload, timeout=60)
    if not response.ok:
        print("HTTP status:", response.status_code)
        print("Response body:", response.text[:2000])
        response.raise_for_status()
    return response.json()


# Force the first turn to demonstrate Anthropic-style tool_use and stop_reason.
first_payload = {
    "model": model,
    "max_tokens": 2048,
    "messages": messages,
    "tools": tools,
    "tool_choice": {"type": "tool", "name": "count_letter"},
}

api_requests = [json.loads(json.dumps(first_payload))]
first_response = send_message(first_payload)
print("--- First Messages API response ---")
print(json.dumps(first_response, indent=2))

api_responses = [first_response]

# Execute each tool_use block and return its result in a tool_result block.
tool_uses = [
    block for block in first_response.get("content", [])
    if block.get("type") == "tool_use"
]

if first_response.get("stop_reason") == "tool_use" and tool_uses:
    messages.append({"role": "assistant", "content": first_response["content"]})

    tool_results = []
    for tool_use in tool_uses:
        if tool_use["name"] != "count_letter":
            raise ValueError(f"Unexpected tool requested: {tool_use['name']}")

        result = count_letter(**tool_use["input"])
        tool_results.append(
            {
                "type": "tool_result",
                "tool_use_id": tool_use["id"],
                "content": json.dumps(result),
            }
        )

    messages.append({"role": "user", "content": tool_results})

    # Let the model answer after it receives the tool result.
    followup_payload = {
        "model": model,
        "max_tokens": 2048,
        "messages": messages,
        "tools": tools,
        "tool_choice": {"type": "auto"},
    }
    api_requests.append(json.loads(json.dumps(followup_payload)))
    followup_response = send_message(followup_payload)
    print("\n--- Final Messages API response ---")
    print(json.dumps(followup_response, indent=2))

    # Store the assistant's final message in the conversation history too.
    messages.append({"role": "assistant", "content": followup_response.get("content", [])})
    api_responses.append(followup_response)
else:
    print("No tool call was returned; stop_reason:", first_response.get("stop_reason"))
    messages.append({"role": "assistant", "content": first_response.get("content", [])})

# Save both the Anthropic-style conversation and the complete raw API responses.
conversation = {
    "model": model,
    "tools": tools,
    "messages": messages,
    "api_requests": api_requests,
    "api_responses": api_responses,
}
with open("src/conversation.json", "w", encoding="utf-8") as file:
    json.dump(conversation, file, indent=2, ensure_ascii=False)

print("\nSaved the full conversation to conversation.json")
