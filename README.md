## xlm-roberta-ua-distilled 🇺🇦🇬🇧

Check out the model card on [HF](https://huggingface.co/panalexeu/xlm-roberta-ua-distilled) 📄 — learn more about how it was built and what it can do.

Also, try the model in action directly via the interactive demo on [HF Spaces](https://huggingface.co/spaces/panalexeu/xlm-roberta-ua-distilled) 🧪
No setup required — test its capabilities right in your browser! 💻

![Playground](./pics/playground.png)


### Training Approach

To train the model, the approach proposed by Nils Reimers and Iryna Gurevych in the following [research paper](https://arxiv.org/pdf/2004.09813) was used.

The idea of the approach is to distill knowledge from the teacher model to the student model, with the loss function being Mean Squared Error (MSE). 

The MSE is calculated between the teacher model’s embedding of a sentence (e.g., in English) and the student model’s embedding of the same sentence in English, as well as versions of the same sentence in other languages (in our case, Ukrainian only).

![https://www.sbert.net/examples/sentence_transformer/training/multilingual/README.html](./pics/paper_approach.png)

In this way, the proposed approach not only distills knowledge from the teacher model to the student, but also "squeezes" the embeddings of different training languages together - which makes sense, since semantically equivalent sentences should have similar vector representations across languages.

This results in improved model performance across several training languages and better cross-lingual transfer.

[SentenceTransformers](https://sbert.net/) library provides ready-to-use tools to implement the training process described above.

The teacher model chosen was [multi-qa-mpnet-base-dot-v1](https://huggingface.co/sentence-transformers/multi-qa-mpnet-base-dot-v1). This model is monolingual (English) and achieves great performance on semantic search tasks (which is well-suited for RAG), based on the benchmarks provided [here](https://sbert.net/docs/sentence_transformer/pretrained_models.html).

The student model chosen was [XLM-RoBERTa](https://huggingface.co/FacebookAI/xlm-roberta-base). This model is a multilingual version of RoBERTa, trained on CommonCrawl data covering 100 languages.

The training was performed on the following parallel sentence datasets, specifically on the `en-uk` subsets:

* [parallel-sentences-talks](https://huggingface.co/datasets/sentence-transformers/parallel-sentences-talks);
* [parallel-sentences-wikimatrix](https://huggingface.co/datasets/sentence-transformers/parallel-sentences-wikimatrix);
* [parallel-sentences-tatoeba](https://huggingface.co/datasets/sentence-transformers/parallel-sentences-tatoeba);

The combined training dataset resulted in more than 500,000 sentence pairs. 

The training was performed for 4 epochs with a batch size of 48. The training hardware was a GPU P100 with 16 GB of memory provided by [Kaggle](https://www.kaggle.com/). On the GPU P100, training took more than 8 hours.

You can check out the training process in more detail in the following [notebook](./researches/research_final.ipynb). 

### Benchmarks

For evaluation and benchmarking, the [sts17-crosslingual-sts](https://huggingface.co/datasets/mteb/sts17-crosslingual-sts) (semantic textual similarity) dataset was used. It consists of multilingual sentence pairs and a similarity score from 0 to 5 annotated by humans. However, the `sts17-crosslingual-sts` dataset does not provide sentence pairs for the Ukrainian language, so they were machine-translated using `gpt-4o`, resulting in `en-en`, `en-ua`, and `ua-ua` evaluation subsets. You can check out the translation process in more detail in the following [notebook](./researches/dataset_translation.ipynb). 

To see the benchmarking process in more detail, check out the following [notebook](./researches/benchmarks.ipynb).

