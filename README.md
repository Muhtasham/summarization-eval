
# Summarization Evaluation

Elegantly automate text summarization evaluation in reference-free manner with potential hallucination detection.

![Demo 1](assets/demo1.png)
![Demo 2](assets/demo2.png)

## Features

- **Easy to Use**: Simply provide a text file containing the summary to be evaluated, and the script will handle the rest.
- **Custom Metrics**: Utilize metrics such as word overlap and Jaccard similarity for in-depth content analysis.
- **Hallucination Detection**: Systematically identify hallucinated content in summaries by analyzing semantic discrepancies with the original text, also use new word detection to identify potentially hallucinated summaries.If you came here for this just check `detect_hallucinations` under `src/utils.py`.
- **GPT-based Evaluation**: Use GPT for nuanced qualitative assessments of summaries which takes care of json enforcement with pydantic, for easier parsing.
- **Adapted ROUGE & BERTScore**: Rework traditional metrics for use in a reference-free context, focusing on the intrinsic qualities of summaries, as highlighted in [Eugene Yan's writing](https://eugeneyan.com/writing/abstractive/).
- **Extensible**: Easily add new metrics and models to the project to expand its capabilities, open an issue or a PR if you want to add something.

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/Muhtasham/summarization-eval
cd summarization-eval
pip install -r requirements.txt
```

### Usage

To use the `main.py` script under `src`, simply provide a text file containing the text you want to summarize. The script reads the file, generates summary, processes the summary, and outputs the evaluation results in a structured and readable format in the terminal.

Example:

```bash
python main.py --file-path "assets/news.txt"
```

**Note**: You will need to have an OpenAI API key set up in your environment to run the script.

## Contributing

Contributions to enhance and expand this project are welcome. Please see the `CONTRIBUTING.md` file for guidelines on how to contribute.

## License

This project is licensed under the [MIT License](LICENSE).
