from __future__ import annotations  # PEP 563/649 future-proof

import json
import os
import pathlib
import re
import sys
from typing import Any, Dict, Final, List, TypedDict

from autogen import AssistantAgent, UserProxyAgent
from dotenv import load_dotenv

# ------- 常量設定 -------
# Use absolute path for more reliable file operations
ROOT_DIR: Final[pathlib.Path] = pathlib.Path(__file__).parent.parent.parent.absolute()
print(f"[DEBUG] ROOT_DIR: {ROOT_DIR}")
WORKSPACE: Final[pathlib.Path] = ROOT_DIR / "test" / "workspace"
print(f"[DEBUG] WORKSPACE: {WORKSPACE}")
WORKSPACE.mkdir(parents=True, exist_ok=True)

# 載入環境變數從 .env 檔案
ENV_PATH = ROOT_DIR / ".env"
if not ENV_PATH.exists():
    print(f"[ERROR] .env file not found at {ENV_PATH}")
    sys.exit(1)
load_dotenv(ENV_PATH)

# Check for required environment variables
if "OPENAI_API_KEY" not in os.environ:
    print("[ERROR] OPENAI_API_KEY environment variable not found in .env file")
    sys.exit(1)


# ---------- Domain model (PEP 484 / PEP 585) ----------
class RequirementDoc(TypedDict):
    user_stories: List[str]
    acceptance_criteria: List[str]
    open_questions: List[str]


# ---------- LLM config ----------
LLM_CFG: Final[Dict[str, Any]] = {
    "model": "gpt-4o-mini",  # swap to your model
    "api_key": os.environ["OPENAI_API_KEY"],
    "temperature": 0.2,
}

# ---------- Agent definitions ----------
analyst = AssistantAgent(
    name="analyst",
    llm_config=LLM_CFG,
    system_message=(
        "You are a senior systems analyst. "
        "Break any given requirement into:\n"
        '• user_stories (max 7, using format: {"as_a": "...", "i_want": "...", "so_that": "..."})\n'
        '• acceptance_criteria (using format: {"given": "...", "when": "...", "then": "..."})\n'
        "• open_questions (items needing stakeholder clarification)\n"
        "Return valid JSON matching the RequirementDoc TypedDict with these fields, without any markdown formatting."
    ),
)

user = UserProxyAgent(
    name="user",
    human_input_mode="NEVER",  # fully automated POC
)


# Function to clean response and extract JSON
def extract_json_from_response(response: str) -> str:
    """Extract JSON content from a potentially markdown-formatted response."""
    # Remove markdown code block delimiters if present
    json_pattern = r"```(?:json)?\s*([\s\S]*?)```"
    match = re.search(json_pattern, response)

    if match:
        return match.group(1).strip()

    # If no markdown formatting, return the original response
    return response.strip()


def main() -> None:
    try:
        raw_need = (
            "We need a mobile app that lets motorcycle riders track every gear shift, "
            "log route telemetry, and get maintenance reminders. It must sync to the "
            "cloud, work offline when signal is bad, and export CSV for mechanics."
        )

        # Use simple direct chat - most reliable approach
        response = analyst.generate_reply(messages=[{"role": "user", "content": raw_need}])

        print("\n=== Structured Requirement ===\n")
        print(response)

        # Extract JSON and parse it
        try:
            # Clean response by removing markdown formatting if present
            cleaned_response = extract_json_from_response(response)
            json_data = json.loads(cleaned_response)
            print("\n=== JSON Parsed Successfully ===")

            # Save the structured requirement to a file
            output_file = WORKSPACE / "requirement_doc.json"
            with open(output_file, "w") as f:
                json.dump(json_data, f, indent=2)
            print(f"\n=== JSON saved to {output_file} ===")

        except json.JSONDecodeError as e:
            print(f"\n=== WARNING: JSON Parse Error: {e} ===")
            print("Raw cleaned response:", cleaned_response)

    except Exception as e:
        print(f"[ERROR] An error occurred: {str(e)}")
        sys.exit(1)


# ---------- Kick-off ----------
if __name__ == "__main__":
    main()
