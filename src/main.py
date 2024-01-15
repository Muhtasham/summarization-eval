import re
import argparse
from typing import List
from openai import OpenAI
from utils import (
    get_env_values,
    read_text,
    process_llm_results,
)
from logger import logger


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
    and process the results. If debug mode is enabled, only use low temperature for generation.

    Args:
        client (OpenAI): The OpenAI client for making API requests.
        text (str): The text to be processed.
        debug (bool): Flag to enable debug mode.
        file_name (str): The name of the file being processed.
        system_prompts (List[str]): List of system prompts to use for generation.
        temperatures (List[float]): List of temperatures to use for generation.
        all_results (List): List to store all generated results.
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
                # Process the results in rich text format and print them in the terminal
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
        default="assets/news.txt",
        type=str,
        help="Path to the news text file.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    env_values = get_env_values()
    openai_api_key = env_values["OPENAI_API_KEY"]
    args = parse_args()
    client = OpenAI(api_key=openai_api_key)

    news_text = read_text(args.file_path)

    # Below we define the system prompts and temperatures we want to use.

    system_prompts = [
        # "Please perform abstractive summarization on the following text",
        # "Please perform extractive summarization on the following text",
        "Summarize the following text briefly",
    ]
    temperatures = [0.2, 0.7]  # Low and high temperatures for some variety

    generate_and_process_llm_responses(
        client, news_text, args.debug, "Original Text", system_prompts, temperatures, []
    )
