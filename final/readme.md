# Sentence Paraphrasing

## Table of Content

[Overview](#overview)

- [Approach](#approach)
- [Frameworks](#frameworks)
- [Pretrained Models](#pretrained-models)
- [Datasets](#datasets)
  - [1. PAWS](#1-paws-paraphrase-adversaries-from-word-scrambling)
  - [2. QQP](#2-qqp-quora-question-pairs)
  - [3. MNLI](#3-mnli-multi-genre-natural-language-inference-corpus)
  - [4. SNLI](#4-snli-standford-natural-language-inference-corpus)

[References](#references)

## Overview

### __Approach__

We define this task as a combination of several problems namely `text-summarisation`, `text2text-generation`, `natural-language-inference[understanding]`, and `sentence-similarity`. Thus, the choice of dataset will have a major impact on our models performance. To save time and resources, we would fine-tune pretrained transformers from HuggingFace and collectively report best result.

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

> In summary, an encoder-decoder Transformer architecture may be the most suitable for the paraphrasing task, as it can leverage the information in the input sequence to generate more accurate and coherent paraphrases. _However, it's important to keep in mind that the choice of architecture depends on the specific requirements of the task, and experimenting with different architectures may be necessary to determine the best approach._

### __Datasets__

#### __1. [PAWS](https://huggingface.co/datasets/paws) (Paraphrase Adversaries from Word Scrambling)__

This dataset contains 108,463 human-labeled and 656k noisily labeled pairs that feature the importance of modeling structure, context, and word order information for the problem of paraphrase identification. The dataset has two subsets, one based on Wikipedia and the other one based on the Quora Question Pairs (QQP) dataset.

Column Name   | Data
:------------ | :--------------------------
id            | A unique id for each pair
sentence1     | The first sentence
sentence2     | The second sentence
(noisy_)label | (Noisy) label for each pair

Each label has two possible values: `0` indicates the pair has different meaning, while `1` indicates the pair is a paraphrase.

|     | Sentence 1                    | Sentence 2                    | Label |
| :-- | :---------------------------- | :---------------------------- | :---- |
| (1) | Although interchangeable, the body pieces on the 2 cars are not similar. | Although similar, the body parts are not interchangeable  on the 2 cars.  | 0     |
| (2) | Katz was born in Sweden in 1947 and moved to New York City at the age of 1.      | Katz was born in 1947 in Sweden and moved to New York at the age of one.   | 1     |

The number of examples and the proportion of paraphrase (Yes%) pairs are shown
below:

Data                | Train   | Dev    | Test  | Yes%   |
:------------------ | ------: | -----: | ----: | ----:  |
Labeled (Final)     | 49,401  | 8,000  | 8,000 | 44.2%  |
Labeled (Swap-only) | 30,397  | --     | --    | 9.6%   |
Unlabeled (Final)   | 645,652 | 10,000 | --    | 50.0%  |

#### __2. [QQP](https://huggingface.co/datasets/glue/viewer/qqp/test_matched) (Quora Question Pairs)__

| Question 1  | Question 2  |label_id|  idx   | label |
| ------ | ----- | ----:| ----:  | ---|
| "How is the life of a math student? Could you describe your own experiences?" | "Which level of prepration is enough for the exam jlpt5?" | 0 | 0 |"not duplicate" |
|"How do I control my horny emotions?" |"How do you control your horniness?" |1 |1 |"duplicate"

|train |validation          |test|
|-----:|-----------------:  |--: |  
|363846|              40430|391k |

#### __3. [MNLI](https://huggingface.co/datasets/multi_nli) (Multi-Genre Natural Language Inference Corpus)__

The _Multi-Genre Natural Language Inference Corpus_ is a crowdsourced collection of sentence pairs with textual entailment annotations. Given a premise sentence and a hypothesis sentence, the task is to predict whether the premise entails the hypothesis (entailment), contradicts the hypothesis (contradiction), or neither (neutral). The premise sentences are gathered from ten different sources, including transcribed speech, fiction, and government reports. The authors of the benchmark use the standard test set, for which they obtained private labels from the RTE authors, and evaluate on both the matched (in-domain) and mismatched (cross-domain) section. They also uses and recommend the SNLI corpus as 550k examples of auxiliary training data.

- Size of downloaded dataset files: 226.85 MB
- Size of the generated dataset: 76.95 MB
- Total amount of disk used: 303.81 MB

Example of a data instance:

```python
{
    "promptID": 31193,
    "pairID": "31193n",
    "premise": "Conceptually cream skimming has two basic dimensions - product and geography.",
    "hypothesis": "Product and geography are what make cream skimming work. ",
    "genre": "government",
    "label": 1
}
```

Data Fields

The data fields are the same among all splits.

- `promptID`: Unique identifier for prompt
- `pairID`: Unique identifier for pair
- `{premise,hypothesis}`: combination of `premise` and `hypothesis`
- `genre`: a `string` feature.
- `label`: a classification label, with possible values including `entailment` (0), `neutral` (1), `contradiction` (2). Dataset instances which don't have any gold label are marked with -1 label. Make sure you filter them before starting the training using `datasets.Dataset.filter`.

Data Splits

|train |validation_matched|validation_mismatched|
|-----:|-----------------:|--------------------:|
|392702|              9815|                 9832|

#### __4. [SNLI](https://huggingface.co/datasets/snli) (Standford Natural Language Inference corpus)__

The SNLI corpus (version 1.0) is a collection of __570k human-written English sentence pairs manually labeled for balanced classification with the labels entailment__, contradiction, and neutral, supporting the task of natural language inference (NLI), also known as recognizing textual entailment (RTE).

Dataset Structure

Data Instances

For each instance, there is a string for the premise, a string for the hypothesis, and an integer for the label. Note that each premise may appear three times with a different hypothesis and label. See the [SNLI corpus viewer](https://huggingface.co/datasets/viewer/?dataset=snli) to explore more examples.

```python
{
    'premise': 'Two women are embracing while holding to go packages.'
    'hypothesis': 'The sisters are hugging goodbye while holding to go packages after just eating lunch.'
    'label': 1
}
```

The average token count for the premises and hypotheses are given below:

| Feature    | Mean Token Count |
| ---------- | ---------------- |
| Premise    | 14.1             |
| Hypothesis | 8.3              |

Data Fields

- `premise`: a string used to determine the truthfulness of the hypothesis
- `hypothesis`: a string that may be true, false, or whose truth conditions may not be knowable when compared to the premise
- `label`: an integer whose value may be either _0_, indicating that the hypothesis entails the premise, _1_, indicating that the premise and hypothesis neither entail nor contradict each other, or _2_, indicating that the hypothesis contradicts the premise. Dataset instances which don't have any gold label are marked with -1 label. Make sure you filter them before starting the training using `datasets.Dataset.filter`.

Data Splits

The SNLI dataset has 3 splits: _train_, _validation_, and _test_. All of the examples in the _validation_ and _test_ sets come from the set that was annotated in the validation task with no-consensus examples removed. The remaining multiply-annotated examples are in the training set with no-consensus examples removed. Each unique premise/caption shows up in only one split, even though they usually appear in at least three different examples.

| Dataset Split | Number of Instances in Split |
| ------------- |----------------------------- |
| Train         | 550,152                      |
| Validation    | 10,000                       |
| Test          | 10,000                       |

### __Training strategy__

<!-- | col1  | col2|
| :--:     | --: |
| &#9745;  | random text is here  | -->

## Our models

| index | model architecture  | dataset | objective | repo on HF |
| :---: | :--------------: |  :-----:  | :--------- |   :-----:  |
| [1]   | `bart-base`      | `paws-unlabeled` | Learning how to paraphrasing sentences in a simple way - scramble give words to address a new outcome |     |
| [2]   | `bart-base`      | `paws-unlabeled`, `qqp` | Adding ability to paraphrasing question, giving the model a diversity of sentence structure to work on|   |
 



## References

- [Exploring Transfer Learning with T5: the Text-To-Text Transfer Transformer](https://ai.googleblog.com/2020/02/exploring-transfer-learning-with-t5.html)
- [PEGASUS: A State-of-the-Art Model for Abstractive Text Summarization](https://ai.googleblog.com/2020/06/pegasus-state-of-art-model-for.html)
- [Transformers BART Model Explained for Text Summarization](https://www.projectpro.io/article/transformers-bart-model-explained/553)

<!-- - [link]() -->
