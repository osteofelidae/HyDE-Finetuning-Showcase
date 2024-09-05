# HyDE + Finetuning Showcase
This project was made over the span of 4 weeks (1 week requirements analysis, 2 weeks programming, 1 week results 
analysis). It showcases the potential of HyDE being used to supplement contaminated data for fine-tuning.\
This showcase is specialized in answering questions about Roblox scripting.

## Acknowledgements
Thank you to CSTL at KAIST for letting me work on this project there.\
Thank you to Dr. Joseph Seering and Juhoon Lee, my faculty advisor and mentor respectively, for guiding me throughout
the four weeks I was there.

## Features
* **15% increase in factual accuracy:** Using Llama 3 8b Instruct, experimental factual accuracy was increased
from 65% to upwards of 80%.
* **Mitigating inaccurate/incorrectly formatted training data:** The data used for fine-tuning was of a different
format than the intended use case, which is often the case in real life. This approach preserves information and allows it to be leveraged in a variety
of tasks.
* **Hot-swappable knowledge base:** Within reason, this system can take into account additions/updates to the knowledge 
base without costly fine-tuning, as long as the relevant pages are added/web scraped which is much less computationally
intensive.

## Citations
Gao, Luyu et. al. _Precise Zero-Shot Dense Retrieval without Relevance Labels._ ArXiv, Dec 2022. https://arxiv.org/abs/2212.10496