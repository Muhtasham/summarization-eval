import os
import re
import textwrap
import spacy
import openai
from typing import List, Set, Union, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from spacy.lang.en.stop_words import STOP_WORDS
from sentence_transformers import SentenceTransformer, util
from rich import print as rprint
from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.traceback import install

install(show_locals=True)
console = Console()
# Load the spaCy model
nlp = spacy.load("en_core_web_lg")
# Load the embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

def process_llm_results(
    result: object,
    text: str,
    all_results: List[object],
    file_name: str,
    llm_name: str,
    debug: bool,
) -> None:
    """
    Process the results obtained from the LLM, performing various analysis and comparison steps.

    Args:
    - result (object): The result object from the LLM.
    - text (str): The original email text.
    - all_results (List[object]): List of previous results for comparison.
    - file_name (str): The name of the file cirrenlty being processed
    - llm_name (str): The mode of the current LLM
    """
    # Extract the generated text
    try:
        generated_text = result.choices[0].message["content"]
        rprint(
            Panel.fit(
                generated_text,
                title=f"[bold red]{llm_name}[/bold red]",
                subtitle=f"[bold red]{file_name}[/bold red]",
            )
        )
    except Exception as e:
        rprint(f"Error extracting generated text: {e}")
        return

    # General Analysis Table
    table = Table(title="General Analysis")
    table.add_column("Metric", style="magenta")
    table.add_column("Value", justify="right", style="cyan")
    table.add_column("Higher/Lower is Better", style="green", no_wrap=True)

    try:
        overlap_with_original = word_overlap(text, generated_text)
        table.add_row(
            "Word overlap with the original email",
            f"{overlap_with_original * 100:.2f}%",
            "Higher",
        )
    except Exception as e:
        table.add_row("Word overlap with the original email", f"Error: {e}", "Higher")

    if (
        debug and len(all_results) > 1
    ):  # Ensures there's a previous result to compare with
        try:
            previous_result = (
                all_results[-2].generations[0][0].text
            )  # Get the previous result

            # Cosine Similarity with Previous Summary
            cosine_similarity_prev = cosine_sim(generated_text, previous_result)
            table.add_row(
                "Cosine similarity (TF-IDF based) with the previous result",
                f"{cosine_similarity_prev:.2f}",
                "Higher",
            )

            # Length Variation
            percentage_difference = (
                (len(generated_text) - len(previous_result)) / len(previous_result)
            ) * 100
            table.add_row(
                "Percentage difference in length with the previous result",
                f"{percentage_difference:.2f}%",
                "N/A",
            )

        except IndexError:
            table.add_row(
                "Comparison with the previous result", "No previous result found", "N/A"
            )
        except Exception as e:
            table.add_row("Comparison with the previous result", f"Error: {e}", "N/A")

    try:
        words_original = process_text(text)
        words_summary = process_text(generated_text)
        jaccard_sim = jaccard_similarity(words_original, words_summary)
        table.add_row(
            "Jaccard similarity with the original email", f"{jaccard_sim:.2f}", "Higher"
        )
    except Exception as e:
        table.add_row(
            "Jaccard similarity with the original email", f"Error: {e}", "Higher"
        )

    try:
        cosine_sim_tfidf = cosine_sim(text, generated_text)
        table.add_row(
            "Cosine similarity (TF-IDF based) with the original email",
            f"{cosine_sim_tfidf:.2f}",
            "Higher",
        )
    except Exception as e:
        table.add_row(
            "Cosine similarity (TF-IDF based) with the original email",
            f"Error: {e}",
            "Higher",
        )

    # Count of new words in the summary
    try:
        new_words = new_words_in_summary(text, generated_text)
        table.add_row("Count of new words in the summary", str(len(new_words)), "Lower")
    except Exception as e:
        table.add_row("Count of new words in the summary", f"Error: {e}", "Lower")

    rprint(table)
    rprint("\n")  # Add a separator between the tables

    # Cosine Similarity (Embeddings) Table
    try:
        cosine_sim_pairs = cosine_similarity_embeddings(text, generated_text)
        cos_table = Table(title="Top Similar Sentences (Embeddings)")
        cos_table.add_column("Original Sentence")
        cos_table.add_column("Summary Sentence")
        cos_table.add_column("Score", justify="right", style="cyan")

        top_n = 3
        for i, pair in enumerate(cosine_sim_pairs[:top_n]):
            cos_table.add_row(
                pair["original_sentence"],
                pair["summary_sentence"],
                f"{pair['score']:.2f}",
            )
            cos_table.add_row("", "", "")  # Add an empty row for spacing
        rprint(cos_table)
    except Exception as e:
        rprint(
            f"[bold red]Error computing cosine similarity (embeddings):[/bold red] {e}"
        )

    # Display new words in summary
    try:
        if new_words:
            rprint(Panel.fit("[bold cyan]New Words in the Summary[/bold cyan]"))
            rprint(Columns(new_words))
    except Exception as e:
        rprint(f"[bold red]Error computing new words in summary:[/bold red] {e}")

    rprint("\n")  # Add another separator

    # After computing other metrics, display potential hallucinations:
    try:
        hallucinated_sentences = detect_hallucinations(
            text, generated_text, manual_threshold=0.3
        )

        if hallucinated_sentences:
            rprint(Panel.fit("[bold cyan]Potential Hallucinated Sentences[/bold cyan]"))

            # Create a table to display the hallucinated sentences, their scores, and the threshold
            hallucination_table = Table(title="Hallucination Analysis")
            hallucination_table.add_column("Sentence", style="magenta")
            hallucination_table.add_column("Score", justify="right", style="cyan")
            hallucination_table.add_column("Threshold", justify="right", style="green")

            for entry in hallucinated_sentences:
                hallucination_table.add_row(
                    entry["sentence"],
                    f"{entry['score']:.2f}",
                    f"{entry['threshold']:.2f}",
                )

            rprint(hallucination_table)
            rprint("\n")  # Add a separator
        else:
            rprint(
                "[bold cyan]No potential hallucinated sentences detected.[/bold cyan]"
            )
            rprint("\n")  # Add a separator
    except Exception as e:
        rprint(f"[bold red]Error detecting hallucinated sentences:[/bold red] {e}")
        rprint("\n")  # Add a separator

def get_env_values():
    """
    Retrieves required environment variables.

    Returns:
        dict: A dictionary containing the values of required environment variables.

    Raises:
        ValueError: If any required environment variable is not set.
    """
    # List of required environment variables
    required_env_vars = ["OPENAI_API_KEY"]

    # Using a dictionary comprehension to construct the environment variables dictionary
    env_values = {var: os.environ.get(var) for var in required_env_vars}

    # Check if all required environment variables are set
    missing_vars = [var for var, value in env_values.items() if not value]
    if missing_vars:
        raise ValueError(f"Missing environment variables: {', '.join(missing_vars)}")

    return env_values

def read_text(file_path: str) -> str:
    """
    Reads the content from the specified file path.

    Args:
    - file_path (str): Path to the text file.

    Returns:
    - str: Content of the if the file is read successfully, else an empty string.
    """
    if not os.path.exists(file_path):
        print(f"File {file_path} not found!")
        return ""

    try:
        with open(file_path, "r") as file:
            text = file.read()
        return text
    except Exception as e:
        print(f"Error reading email text: {e}")
        return ""


def preprocess_text(text: str) -> str:
    # Convert to lowercase and remove punctuation and numbers
    text = re.sub(r"[^\w\s]", "", text.lower())

    # Tokenize, lemmatize, and remove stop words
    tokens = [
        token.lemma_ for token in nlp(text) if not token.is_stop and token.is_alpha
    ]

    return " ".join(tokens)

def new_words_in_summary(original_text: str, summary: str) -> Set[str]:
    """
    Identify new words present in the summary that were not in the original text.

    Args:
    - original_text (str): Original text.
    - summary (str): Generated summary.

    Returns:
    - set: Set of new words in the summary.
    """
    original_words = set(
        filter_words([token.text for token in nlp(original_text.lower())])
    )
    summary_words = set(filter_words([token.text for token in nlp(summary.lower())]))

    # Find the new words in the summary
    new_words = summary_words - original_words
    return new_words


def filter_words(word_list: List[str]) -> Set[str]:
    """
    Filters out stop words, numbers, and short words from a list of words.

    Args:
    - word_list (List[str]): List of words.

    Returns:
    - Set[str]: Filtered set of words.
    """
    return {
        word
        for word in word_list
        if word not in STOP_WORDS and word.isalpha() and len(word) > 2
    }


def process_text(text: str) -> Set[str]:
    """
    Process a given text, tokenizing, lemmatizing, and extracting named entities.

    Args:
    - text (str): Text to be processed.

    Returns:
    - Set[str]: A set of processed words and named entities.
    """
    # Process the text using spaCy
    doc = nlp(text)

    # Extract words and named entities
    words = {
        token.lemma_.lower() for token in doc if not token.is_stop and token.is_alpha
    }
    named_entities = {ent.text.lower() for ent in doc.ents}

    # Combine words and named entities
    return words.union(named_entities)


def word_overlap(text1: str, text2: str) -> float:
    """
    Compute the word overlap between two texts.

    Args:
    - text1 (str): First text.
    - text2 (str): Second text.

    Returns:
    - float: Overlap ratio.
    """
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    overlap = words1.intersection(words2)
    return len(overlap) / (len(words1) + len(words2) - len(overlap))


def jaccard_similarity(set1: Set[str], set2: Set[str]) -> float:
    """
    Calculate Jaccard Similarity between two sets.

    Args:
    - set1 (Set[str]): First set.
    - set2 (Set[str]): Second set.

    Returns:
    - float: Jaccard similarity score.
    """
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union


def cosine_sim(text1: str, text2: str) -> float:
    """
    Calculate cosine similarity between two texts using TF-IDF.

    Args:
    - text1 (str): First text.
    - text2 (str): Second text.

    Returns:
    - float: Cosine similarity score.
    """
    vectorizer = TfidfVectorizer().fit_transform([text1, text2])
    vectors = vectorizer.toarray()
    return cosine_similarity(vectors)[0][1]


def wrap_text(text: str, width: int = 50, subsequent_indent: str = "") -> str:
    """Wrap text to the specified width."""

    # Break the text into paragraphs (preserving existing line breaks)
    paragraphs = text.split("\n")

    # Wrap each paragraph
    wrapped_paragraphs = [
        textwrap.fill(p, width, subsequent_indent=subsequent_indent) for p in paragraphs
    ]

    # Combine the wrapped paragraphs
    return "\n".join(wrapped_paragraphs)


def cosine_similarity_embeddings(
    original_text: str, summary: str, return_pairs: bool = True
) -> Union[List[Dict[str, float]], List[float]]:
    """
    Compute the cosine similarity between the embeddings of summary and each sentence in original_text.

    This function uses a pre-defined `model` (from the sentence-transformers library)
    to obtain embeddings for each sentence in original_text and the entire summary.

    Parameters:
    - original_text (str): The original text.
    - summary (str): The summary or generated text.
    - return_pairs (bool): Whether to return sorted pairs or just scores.

    Returns:
    - Union[List[Dict[str, float]], List[float]]: Depending on return_pairs, either a list of dictionaries containing sentence pairs with their cosine similarity score or just the scores.
    """
    try:
        original_text = preprocess_text(original_text)
        summary = preprocess_text(summary)

        original_sentences = [sent.text for sent in nlp(original_text).sents]
        summary_sentences = [sent.text for sent in nlp(summary).sents]

        original_embeddings = model.encode(original_sentences, convert_to_tensor=True)
        scores = []

        for sentence in summary_sentences:
            sentence_embedding = model.encode(sentence, convert_to_tensor=True)
            cosine_scores = util.pytorch_cos_sim(
                sentence_embedding, original_embeddings
            )
            best_match_idx = cosine_scores.argmax().item()
            scores.append(cosine_scores[0][best_match_idx].item())

        if return_pairs:
            pairs = [
                {
                    "original_sentence": wrap_text(original_sentences[i]),
                    "summary_sentence": wrap_text(summary_sentences[i]),
                    "score": scores[i],
                }
                for i in range(len(scores))
            ]
            return sorted(pairs, key=lambda x: x["score"], reverse=True)
        else:
            return scores

    except Exception as e:
        print(f"Error computing cosine similarity (embeddings): {e}")
        return []


def detect_hallucinations(
    original_text: str,
    summary: str,
    method: str = "average",
    manual_threshold: float = None,
) -> List[dict]:
    """
    Detect potential hallucinated sentences in the summary using semantic similarity.

    Args:
    - original_text (str): The original text.
    - summary (str): The generated summary.
    - method (str, optional): Method to compute the threshold. Can be "average" or "median".
    - manual_threshold (float, optional): If set, this threshold will be used instead of dynamically computed thresholds.

    Returns:
    - List[dict]: Each dictionary contains the hallucinated sentence, its score, and the threshold used.
    """
    scores = cosine_similarity_embeddings(original_text, summary, return_pairs=False)

    # Compute threshold
    if manual_threshold:
        threshold = manual_threshold
    else:
        threshold = (
            sum(scores) / len(scores)
            if method == "average"
            else sorted(scores)[len(scores) // 2]
        )

    summary_sentences = [sent.text for sent in nlp(preprocess_text(summary)).sents]
    results = []

    # Detect hallucinations
    for idx, score in enumerate(scores):
        if score < threshold:
            results.append(
                {
                    "sentence": wrap_text(summary_sentences[idx]),
                    "score": score,
                    "threshold": threshold,
                }
            )

    return results

# based on https://eugeneyan.com/writing/abstractive/
# ToDO:
from rouge import Rouge

def calculate_rouge_c(summary, document):
    rouge = Rouge()
    scores = rouge.get_scores(summary, document)
    return scores  # This will return scores for ROUGE-1, ROUGE-2, and ROUGE-L

from bert_score import score as bert_score

def calculate_bert_score(summary, document):
    P, R, F1 = bert_score([summary], [document], lang="en", rescale_with_baseline=True)
    return {"precision": P, "recall": R, "f1": F1}

# ToDo: Add JSON enforcement
def g_eval_with_gpt(summary, document, openai_api_key,):
    
    env_values = get_env_values()
    openai_api_key = env_values["OPENAI_API_KEY"]
    client = openai.OpenAI(api_key=openai_api_key)

    evaluation_prompt = (
        f"Evaluate the following summary based on fluency, coherence, relevance, and consistency: \n"
        f"Summary: {summary}\nDocument: {document}\n"
        "Rate each aspect from 1 to 5 and provide a brief justification for your rating."
    )

    messages = [
        {"role": "system", "content": evaluation_prompt}
    ]

    try:
        completion = client.Completions.create(
            messages=messages,
            model="gpt-4-turbo",
            seed=42,
        )
        evaluation_response = completion.choices[0].message["content"]
        return evaluation_response
    except Exception as e:
        print(f"An error occurred during G-Eval with GPT: {e}")
        return None

