from typing import List
from openai import OpenAI
from utils import (
    get_env_values,
    read_text,
    process_llm_results,
)
from logger import logger
import re
import argparse


def generate_and_process_llm_responses(
    client: OpenAI,
    text: str,
    debug: bool,
    file_name: str,
    system_prompts: List[str],
    temperatures: List[float],
    all_results: List,
) -> None:
    """
    Generate responses using the language model with different system prompts and temperatures,
    and process the results.
    """
    sanitized_text = re.sub(r"[\x00-\x1F]+", "", text)

    # Use only low temperature if debug is True
    effective_temperatures = (
        [t for t in temperatures if t <= 0.5] if debug else temperatures
    )

    for temperature in effective_temperatures:
        llm_name = (
            "LLM call with high temperature"
            if temperature > 0.5
            else "LLM call with low temperature"
        )
        for system_prompt in system_prompts:
            messages = [
                {"role": "system", "content": f"{system_prompt}: {sanitized_text}"}
            ]
            try:
                completion = client.chat.completions.create(
                    messages=messages,
                    model="gpt-3.5-turbo",
                    temperature=temperature,
                )
                model_reply = completion.choices[0].message.content
                all_results.append(model_reply)
                process_llm_results(
                    str(model_reply), text, all_results, file_name, llm_name, debug
                )
                if debug:
                    # Debug-specific output or processing
                    logger.info(
                        f"Generated with temperature {temperature} using prompt '{system_prompt}'"
                    )
            except Exception as e:
                logger.error(
                    f"An error occurred with prompt '{system_prompt}' and temperature {temperature}: {e}"
                )


def parse_args():
    parser = argparse.ArgumentParser(description="Run LLM with different prompts.")
    parser.add_argument(
        "--debug",
        default=True,
        help="Enable debug mode. If enabled, only low temperatures will be used and some debug-specific output will be printed.",
    )
    parser.add_argument(
        "--file-path",
        default="./news.txt",
        type=str,
        help="Path to the email text file.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    env_values = get_env_values()
    openai_api_key = env_values["OPENAI_API_KEY"]
    args = parse_args()
    client = OpenAI(api_key=openai_api_key)

    news_text = read_text(args.file_path)

    system_prompts = [
        # "Please perform abstractive summarization on the following text",
        # "Please perform extractive summarization on the following text",
        "Summarize the following text briefly",
    ]
    temperatures = [0.2, 0.7]

    generate_and_process_llm_responses(
        client, news_text, args.debug, "Original Text", system_prompts, temperatures, []
    )
