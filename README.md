## xlm-roberta-ua-distilled 🇺🇦🇬🇧

Check out the model card on [HF](https://huggingface.co/panalexeu/xlm-roberta-ua-distilled) 📄 — learn more about how it was built and what it can do.

Also, try the model in action directly via the interactive demo on [HF Spaces](https://huggingface.co/spaces/panalexeu/xlm-roberta-ua-distilled) ⚙️🧪  
No setup required — test its capabilities right in your browser! 💻✨

![Playground](./pics/playground.png)


### Training approach 

To train the model, approach proposed by Nils Reimers and Iryna Gurevych  in the following research [paper](https://arxiv.org/pdf/2004.09813) was used.

The idea of the approach is to distill knowledge from the teacher model to student model, with the loss function being MSE.
The MSE is calcualted between teacher's embedding of some sentence (e.g. in English), and student's model embedding of the same sentence in English,
and versions of the same sentence in other languages, (in our case Ukrainian only).

![https://www.sbert.net/examples/sentence_transformer/training/multilingual/README.html](./pics/paper_approach.png)

This way the proposed approach not only distills knowledge from teacher model to student, but also "squeezes" embeddings of training languages into each other
(which makes sense, since semantically the same sentences should have close vectors across languages).
This results in increase of model performance across several training languages and cross lingual transfer. 


### Benchmarks

To see the process of benchmarking, check out the following [notebook](./researches/benchmarks.ipynb).

work in progress ...
