# Benchmarks-for-NLP

## Open-Source LLM (10-35B, Chinese & English)
> Leaderboard from [OpenCompass](https://opencompass.org.cn/leaderboard-llm)

| Model                |              Org.               | Param. | Examination | Language | Knowledge | Understanding | Reasoning |
| :------------------- | :-----------------------------: | :----: | :---------: | :------: | :-------: | :-----------: | :-------: |
| GPT-4                |             OpenAI              |  N/A   |    77.2     |    62    |   73.5    |      70       |   74.4    |
| **ChatGLM3-6B-Base** |             ZhipuAI             |   6B   |    67.2     |   52.4   |    62     |     70.3      |   67.4    |
| Qwen-14B             |             Alibaba             |  14B   |    71.3     |   52.7   |   56.1    |     68.8      |   60.1    |
| Yi-34B               |              01.AI              |  34B   |    78.1     |   48.9   |   64.5    |     69.2      |   55.5    |
| Qwen-14B-Chat        |             Alibaba             |  14B   |    71.2     |   52.1   |   61.2    |     68.2      |   54.9    |
| InternLM-20B         |   Shanghai AI Lab & SenseTime   |  20B   |    62.5     |    55    |   60.1    |     67.3      |   54.9    |
| Aquila2-34B          |              BAAI               |  34B   |     70      |   47.2   |   59.2    |     66.9      |   50.1    |
| Baichuan2-13B-Chat   | Baichuan Intelligent Technology |  13B   |    59.8     |   51.5   |   51.9    |     63.1      |   50.1    |

## GLUE Benchmark (NLU Tasks, English)
> Single-task single models on dev
> 
| Model          | MNLI | QNLI | QQP  | RTE  | SST  | MRPC | CoLA | STS  |
| :------------- | :--: | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| [BERT-large](https://arxiv.org/pdf/1810.04805.pdf)     | 86.6 | 92.3 | 91.3 | 70.4 | 93.2 | 88.0 | 60.6 | 90.0 |
| [XLNet-large](https://arxiv.org/pdf/1906.08237.pdf)    | 89.8 | 93.9 | 91.8 | 83.8 | 95.6 | 89.2 | 63.6 | 91.8 |
| [RoBERTa-large](https://arxiv.org/pdf/1907.11692.pdf)  | 90.2 | 94.7 | 92.2 | 86.6 | 96.4 | 90.9 | 68.0 | 92.4 |
|[ALBERT-xxlarge](https://arxiv.org/pdf/1909.11942.pdf) | 90.8 | 95.3 | 92.2 | 89.2 | 96.9 | 90.9 | 71.4 | 93.0 |

**其它语言模型**
1. [ERNIE 3.0](https://arxiv.org/pdf/2107.02137.pdf)：自回归与自编码结合的框架
2. [BART](https://arxiv.org/pdf/1910.13461.pdf)：编码器mask+解码器自回归
3. [PERT](https://arxiv.org/pdf/2203.06906.pdf)：基于文本乱序自监督的预训练编码器
4. [MacBERT](https://aclanthology.org/2020.findings-emnlp.58/)：针对中文提出Whole Word Masking和N-gram masking技术


[**About GLUE**](https://gluebenchmark.com/)
1. CoLA：单句二分类任务，判断是否符合语法。
2. SST：单句二分类任务，判断情感积极/消极。
3. MRPC：双句释义二分类，判断两句话是否相同含义。
4. STS：双句相似性回归任务，1-5之间打分。
5. QQP：双句相似性分类任务，判断两句问句含义是否相同。
6. MNLI：三分类任务，判断两句关系：包含、矛盾和中立。输入句子对，一个为条件，另一个为假设。
7. QNLI：二分类，判断问题和句子之间是否为包含关系。
8. RTE：二分类，判断句子对是否互为包含关系。


## RE LLM 
> Leaderboard from [OpenCompass]([https://opencompass.org.cn/leaderboard-llm](https://paperswithcode.com/sota/relation-extraction-on-docred))

|              Model                |      F1     |     Ign F1     | Extra Training Data |
| :-------------------------------- | :---------: | :------------: | :-----------------: |
| [DREEAM](https://paperswithcode.com/paper/dreeam-guiding-attention-with-evidence-for)                            |    67.53    |     65.47      |   NONE  |
| [KD-Rb-I](https://paperswithcode.com/paper/document-level-relation-extraction-with-4)                           |    67.28    |     65.24      |   NONE  |
| [SSAN-RoBERTa-large+Adaptation](https://paperswithcode.com/paper/entity-structure-within-and-throughout)     |    65.92    |     63.78      |   NONE  |
| [SAIS-RoBERTa-large](https://paperswithcode.com/paper/sais-supervising-and-augmenting-intermediate)                |    65.11    |     63.44      |   NONE  |
| [Eider-RoBERTa-large](https://paperswithcode.com/paper/eider-evidence-enhanced-document-level)               |    64.79    |     62.85      |   NONE  |
|[DocuNet-RoBERTa-large](https://paperswithcode.com/paper/document-level-relation-extraction-as)             |    64.55    |     62.40      |   NONE  |
|[CGM2IR-RoBERTalarge](https://paperswithcode.com/paper/document-level-relation-extraction-with-2)               |    63.89    |     61.96      |   NONE  |
| [SETE-Roberta-large](https://paperswithcode.com/paper/document-level-relation-extraction-with-6)                |    63.74    |     61.78      |   NONE  |


