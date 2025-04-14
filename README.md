## xlm-roberta-ua-distilled 🇺🇦🇬🇧

Check out the model card on [HF](https://huggingface.co/panalexeu/xlm-roberta-ua-distilled) 📄 — learn more about how it was built and what it can do.

Also, try the model in action directly via the interactive demo on [HF Spaces](https://huggingface.co/spaces/panalexeu/xlm-roberta-ua-distilled) ⚙️🧪
No setup required — test its capabilities right in your browser! 💻✨

![Playground](./pics/playground.png)


### Training Approach

To train the model, the approach proposed by Nils Reimers and Iryna Gurevych in the following research [paper](https://arxiv.org/pdf/2004.09813) was used.

The idea of the approach is to distill knowledge from the teacher model to the student model, with the loss function being Mean Squared Error (MSE). 

The MSE is calculated between the teacher model’s embedding of a sentence (e.g., in English) and the student model’s embedding of the same sentence in English, as well as versions of the same sentence in other languages (in our case, Ukrainian only).

![https://www.sbert.net/examples/sentence_transformer/training/multilingual/README.html](./pics/paper_approach.png)

In this way, the proposed approach not only distills knowledge from the teacher model to the student, but also "squeezes" the embeddings of different training languages together - which makes sense, since semantically equivalent sentences should have similar vector representations across languages.

This results in improved model performance across several training languages and better cross-lingual transfer.

### Benchmarks

To see the process of benchmarking, check out the following [notebook](./researches/benchmarks.ipynb).

work in progress ...
