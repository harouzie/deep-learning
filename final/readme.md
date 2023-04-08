# Sentence Paraphrasing

## Overview

### __Approach__

We define this task as a combination of several problems namely `text-summarization`, `natural-language-inference`, and `sentence-similarity`. Thus, the choice of dataset will have a major impact on our models performance. To save time and resources, we would fine-tune pretrained transformers from HuggingFace and collectively report best result.

### __Frameworks__

`transformers`, `sentence-transformer`, `sentencepiece`, `nltk`, `datasets`, `evaluate` 

### __Pretrained models__

Having mentioned above, this project will mainly use pretrained models from HuggingFace Hub. You can come there, pick yourself a good transformer, and start your own business.  

- [`pegasus-xsum`](https://huggingface.co/google/pegasus-xsum) - __Pegasus__ (Pre-training with Extracted Gap-sentences for Abstractive Summarization) is a transformer model that was introduced by Google in 2020. It was pre-trained using a variant of the denoising autoencoder objective and fine-tuned on summarization tasks. It has shown to be effective in generating high-quality summaries for long documents.

    ![google-pegasus](https://1.bp.blogspot.com/-TSor4o51jGI/Xt50lkj6blI/AAAAAAAAGDs/TrDe9jv13WEwk9NQNebQL63jtY8n6JFGwCLcBGAsYHQ/s1600/image1.gif)

- [`t5-base`](https://huggingface.co/t5-base) - T5 (Text-to-Text Transfer Transformer) - T5 is a large-scale transformer model that was introduced by Google in 2020. It was trained on a diverse set of natural language tasks, including summarization, and has achieved state-of-the-art results on various summarization benchmarks.

    ![google-t5](https://1.bp.blogspot.com/-o4oiOExxq1s/Xk26XPC3haI/AAAAAAAAFU8/NBlvOWB84L0PTYy9TzZBaLf6fwPGJTR0QCLcBGAsYHQ/s1600/image3.gif)

- [`bart-base`](https://huggingface.co/facebook/bart-base) - BART (Bidirectional and Auto-Regressive Transformer) - BART is a sequence-to-sequence transformer model that was introduced by Facebook AI in 2019. It is pre-trained on a combination of denoising autoencoding and masked language modeling tasks, and has achieved state-of-the-art results on various summarization datasets.

    ![bart](https://dezyre.gumlet.io/images/blog/transformers-bart-model-explained/image_10036092571642833003977.png?w=900&dpr=2.0)

As you can see from the list that all the 3 transformers are encoder-decoder based. The reason why we choose these models is logically affected by the objective outcome. For the paraphrasing task, the choice of Transformer architecture depends on the specific requirements of the task and the available resources.

- __Encoder-only__ Transformer models are typically used for tasks such as language modeling, where the goal is to predict the next token in a sequence given the previous tokens. However, since the paraphrasing task involves generating a new sequence that is semantically equivalent to the input sequence, _an encoder-only model may not be sufficient to capture the complex relationships between the input and output sequences._

- __Decoder-only__ Transformer models are used for tasks such as language generation, where the goal is to generate a sequence from scratch. While a decoder-only model can be used for the paraphrasing task, _it may not be able to leverage the information in the input sequence to generate more accurate and coherent paraphrases._

- __Encoder-decoder__ Transformer models, on the other hand, are designed to capture the relationships between the input and output sequences by using an encoder to represent the input sequence and a decoder to generate the output sequence. _This architecture has been shown to be effective for many sequence-to-sequence tasks, including machine translation and summarization, and may also be suitable for the paraphrasing task._

In summary, an encoder-decoder Transformer architecture may be the most suitable for the paraphrasing task, as it can leverage the information in the input sequence to generate more accurate and coherent paraphrases. _However, it's important to keep in mind that the choice of architecture depends on the specific requirements of the task, and experimenting with different architectures may be necessary to determine the best approach._

### Datasets



## References

- [Exploring Transfer Learning with T5: the Text-To-Text Transfer Transformer](https://ai.googleblog.com/2020/02/exploring-transfer-learning-with-t5.html)
- [PEGASUS: A State-of-the-Art Model for Abstractive Text Summarization](https://ai.googleblog.com/2020/06/pegasus-state-of-art-model-for.html)
- [Transformers BART Model Explained for Text Summarization](https://www.projectpro.io/article/transformers-bart-model-explained/553)

<!-- - [link]() -->
