
# Context-based (Reference-Free) Evaluation for Abstractive Summaries

## Overview

This project offers a suite of tools for automated evaluation of text summarizations in a context-based, reference-free manner. It uniquely assesses summaries through a blend of custom metrics and established metrics from literature, including those based on large language models. The results are elegantly displayed in the terminal, providing an insightful analysis of summary quality.

## Features

- **Custom Metrics**: Utilize metrics such as word overlap and Jaccard similarity for in-depth content analysis.
- **Hallucination Detection**: Systematically identify and highlight hallucinated or fabricated content in summaries.
- **GPT-based Evaluation**: Employ GPT models for nuanced qualitative assessments of summaries, inspired by ideas discussed in [Eugene Yan's work on abstractive summarization](https://eugeneyan.com/writing/abstractive/).
- **Adapted ROUGE & BERTScore**: Rework traditional metrics for use in a reference-free context, focusing on the intrinsic qualities of summaries, as highlighted in [Eugene Yan's writing](https://eugeneyan.com/writing/abstractive/).

## Main Functionality

The `main.py` script serves as the entry point for the project. It is designed to process a text file (e.g., `news.txt`) and automatically perform a comprehensive evaluation of the contained summary. The script orchestrates the entire evaluation process, leveraging various metrics and models to assess the quality of the summary text.

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/Muhtasham/summarization-eval
cd summarization-eval
pip install -r requirements.txt
```

### Usage

To use the script, simply provide a text file containing the summary to be evaluated. The script reads the file, processes the summary, and outputs the evaluation results in a structured and readable format in the terminal.

Example:

```bash
python main.py --input_file "assets/news.txt"
```

This command will process the summary in `news.txt`, evaluating it using the suite of metrics and models integrated into the project.

Here what the output looks like:

![Demo 1](assets/demo1.png)
![Demo 2](assets/demo2.png)

**Note**: You will need to have an OpenAI API key set up in your environment to run the script.

## Contributing

Contributions to enhance and expand this project are welcome. Please see the `CONTRIBUTING.md` file for guidelines on how to contribute.

## License

This project is licensed under the [MIT License](LICENSE).
