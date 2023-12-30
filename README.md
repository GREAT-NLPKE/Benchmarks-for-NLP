# Benchmarks-for-NLP
<!-- TOC -->

- [Benchmarks-for-NLP](#benchmarks-for-nlp)
  - [Open-Source LLM (10-35B, Chinese \& English)](#open-source-llm-10-35b-chinese--english)
  - [GLUE Benchmark (NLU Tasks, English)](#glue-benchmark-nlu-tasks-english)
  - [Named Entity Recognition](#named-entity-recognition)
    - [LLM](#llm)
    - [Flat NER](#flat-ner)
    - [Nested NER](#nested-ner)
    - [Biomedical NER](#biomedical-ner)
  - [Relation Extraction](#relation-extraction)
    - [Document-level](#document-level)
    - [Biomedical](#biomedical)
  - [Key Information Extraction](#key-information-extraction)
  - [Link Prediction](#link-prediction)
    - [WN18RR](#wn18rr)

<!-- /TOC -->
<!-- /TOC -->
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
| Model                                                  | MNLI | QNLI | QQP  | RTE  | SST  | MRPC | CoLA | STS  |
| :----------------------------------------------------- | :--: | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| [BERT-large](https://arxiv.org/pdf/1810.04805.pdf)     | 86.6 | 92.3 | 91.3 | 70.4 | 93.2 | 88.0 | 60.6 | 90.0 |
| [XLNet-large](https://arxiv.org/pdf/1906.08237.pdf)    | 89.8 | 93.9 | 91.8 | 83.8 | 95.6 | 89.2 | 63.6 | 91.8 |
| [RoBERTa-large](https://arxiv.org/pdf/1907.11692.pdf)  | 90.2 | 94.7 | 92.2 | 86.6 | 96.4 | 90.9 | 68.0 | 92.4 |
| [ALBERT-xxlarge](https://arxiv.org/pdf/1909.11942.pdf) | 90.8 | 95.3 | 92.2 | 89.2 | 96.9 | 90.9 | 71.4 | 93.0 |

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

## Named Entity Recognition
> F1 for all<br>

### LLM
| Method                                               | Param. | Org.                                                  | Year | CoNLL03 | OntoNotes | ACE05 | BC5CDR | NCBI  |
| :--------------------------------------------------- | :----: | ----------------------------------------------------- | ---- | ------- | --------- | ----- | ------ | ----- |
| [UniversalNER](https://arxiv.org/pdf/2308.03279.pdf) |   7B   | University of Southern California, Microsoft Research | 2023 | 93.30   | 89.91     | 86.69 | 89.34  | 86.96 |


### Flat NER
| Method                                                                                           |                                                                         Model                                                                          |                  Org.                  |   Year    | [CoNLL03(Eng)](https://data.deepai.org/conll2003.zip) | [OntoNotes]( https://github.com/yhcc/OntoNotes-5.0-NER) |
| :----------------------------------------------------------------------------------------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------: | :------------------------------------: | :-------: | :---------------------------------------------------: | :-----------------------------------------------------: |
| [ACE](https://arxiv.org/pdf/2010.05006v4.pdf)                                                    |                                                              BiLSTM-CRF + BiLSTM-Biaffine                                                              |        ShanghaiTech, UCAS, DAMO        |  ACL2021  |                         94.6                          |                                                         |
| [CL-KL](https://arxiv.org/pdf/2105.03654v3.pdf)                                                  |                                                                     Bio-BERT + CRF                                                                     |                  DAMO                  |  ACL2021  |                         93.56                         |                                                         |
| [BOPN](https://arxiv.org/pdf/2310.18349.pdf)                                                     |                                                              BERT-Large+BiLSTM+CLN+3DConv                                                              |           CAS, UCAS, Meituan           | EMNLP2023 |                         93.19                         |                          91.16                          |
| [PromptNER](https://arxiv.org/pdf/2305.17104.pdf)                                                |                                                                     RoBERTa-large                                                                      |                  ZJU                   | ACL 2023  |                         93.08                         |                                                         |
| [PIQN](https://arxiv.org/pdf/2203.10545v1.pdf)                                                   |                                                                      BERT+2BiLSTM                                                                      |               ZJU, DAMO                |  ACL2022  |                         92.87                         |                          90.96                          |
| [BART NER](https://arxiv.org/pdf/2106.01223v1.pdf)                                               |                                                                       BART-large                                                                       |                  FDU                   |  ACL2021  |                         93.24                         |                          90.38                          |
| [Locate and Label](https://arxiv.org/pdf/2105.06804v2.pdf)                                       |                                                                    BERT-large-cased                                                                    |               ZJU, USTC                |  ACL2021  |                         92.94                         |                                                         |
| [Named Entity Recognition as Dependency Parsing](https://aclanthology.org/2020.acl-main.577.pdf) | BERT-Large + [fastText embeddings](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00051/43387/Enriching-Word-Vectors-with-Subword-Information) | Queen Mary University, Google Research |  ACL2020  |                         92.5                          |                          89.83                          |

### Nested NER
| Method                                                                                              |                                                                         Model                                                                          |                  Org.                  |   Year    | [ACE2004](https://catalog.ldc.upenn.edu/LDC2005T09) | [ACE2005](https://catalog.ldc.upenn.edu/LDC2006T06) | [Genia](http://www.geniaproject.org/genia-corpus) | [KBP17](https://catalog.ldc.upenn.edu/LDC2017D55) |
| :-------------------------------------------------------------------------------------------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------: | :------------------------------------: | :-------: | :-------------------------------------------------: | :-------------------------------------------------: | :-----------------------------------------------: | :-----------------------------------------------: |
| [BOPN](https://arxiv.org/pdf/2310.18349.pdf)                                                        |                                                              BERT-Large+BiLSTM+CLN+3DConv                                                              |           CAS, UCAS, Meituan           | EMNLP2023 |                        89.26                        |                        90.39                        |                       82.14                       |                                                   |
| [PromptNER](https://arxiv.org/pdf/2305.17104.pdf)                                                   |                                                                     RoBERTa-large                                                                      |                  ZJU                   | ACL 2023  |                        88.72                        |                        88.26                        |                                                   |                                                   |
| [PIQN](https://arxiv.org/pdf/2203.10545v1.pdf)                                                      |                                                                      BERT+2BiLSTM                                                                      |               ZJU, DAMO                |  ACL2022  |                        88.14                        |                        87.42                        |                       81.77                       |                       84.50                       |
| [Locate and Label](https://arxiv.org/pdf/2105.06804v2.pdf)                                          |                                                                    BERT-large-cased                                                                    |               ZJU, USTC                |  ACL2021  |                        87.41                        |                        86.67                        |                       80.54                       |                       84.05                       |
| [BART NER](https://arxiv.org/pdf/2106.01223v1.pdf)                                                  |                                                                       BART-large                                                                       |                  FDU                   |  ACL2021  |                        86.84                        |                        84.74                        |                       79.23                       |                                                   |
| [Named Entity Recognition as Dependency Parsing](https://aclanthology.org/2020.acl-main.577.pdf)    | BERT-Large + [fastText embeddings](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00051/43387/Enriching-Word-Vectors-with-Subword-Information) | Queen Mary University, Google Research |  ACL2020  |                        85.67                        |                        84.61                        |                       78.87                       |                                                   |
| [Neural Architectures for Nested NER through Linearization](https://arxiv.org/pdf/1908.06926v1.pdf) |                                                                   seq2seq+BERT+Flair                                                                   |           Charles University           |  ACL2019  |                        84.40                        |                        84.33                        |                       78.31                       |                                                   |
### Biomedical NER
| Method                                           |         Model         |                      Org.                      |             Year             | [BC5CDR](https://academic.oup.com/database/article/doi/10.1093/database/baw068/2630414) |                          [NCBI-disease](https://www.ncbi.nlm.nih.gov/CBBresearch/Dogan/DISEASE/)                           |
| :----------------------------------------------- | :-------------------: | :--------------------------------------------: | :--------------------------: | :----: | :-------------------------------------------------------------: |
| [BINDER](https://arxiv.org/pdf/2208.14565v2.pdf) |      PubMedBERT       |               Microsoft Research               |          ICLR 2023           |  91.9  |                              90.9                               |
| [ConNER](https://arxiv.org/pdf/2210.12949v1.pdf) |     BioBERT/BioLM     |                Korea University                | Oxford University Press 2022 | 91.3 |89.9|
| [CL-KL](https://arxiv.org/pdf/2105.03654v3.pdf)  |    Bio-BERT + CRF     |                      DAMO                      |           ACL2021            | 90.93  |                              88.96                              |
| [BioFLAIR](https://arxiv.org/pdf/1908.05760.pdf) | BioFLAIR (V1)+BioELMo | Manipal Institute of Technology, Elsevier Labs |             Arxiv2019             | 89.42  |                              88.85                              |
| [BioBERT](https://arxiv.org/pdf/1901.08746.pdf)  |        BioBERT        |           Korea University, Clova AI           | Oxford University Press 2020 |        | [87.70(token-level F1)](https://arxiv.org/pdf/2105.03654v3.pdf) |


## Relation Extraction
> RE F1 for all<br>
> **DocRED** is a large scale dataset constructed from Wikipedia and Wikidata. <br>
> **CDR**(Chemical-Disease Reactions) is a biomedical dataset constructed using PubMed abstracts. <br>
> **GDA**(Gene-Disease Associations) is also a binary relation classification task that identify Gene and Disease concepts interactions.
### Document-level
| Method                                                         |     Model     |                    Org.                    |    Year    | [DocRED](https://github.com/thunlp/DocRED)-Test | [Re-DocRED](https://github.com/tonytan48/Re-DocRED)-Test |
| :------------------------------------------------------------- | :-----------: | :----------------------------------------: | :--------: | :---------------------------------------------: | :------------------------------------------------------: |
| [AA-RE](https://arxiv.org/pdf/2310.18604.pdf)                  | RoBERTa-large |             Beihang University             | EMNLP2023  |                      64.98                      |                          81.20                           |
| [DREEAM](https://arxiv.org/pdf/2302.08675v1.pdf)               | RoBERTa-large |       Tokyo Institute of Technology        |  ACL2023   |                      64.27                      |                          80.73                           |
| [SAIS](https://arxiv.org/pdf/2109.12093v2.pdf)                 | RoBERTa-large |               CMU, Stanford                | NAACL 2022 |                      65.11                      |                            -                             |
| [EIDER](https://arxiv.org/pdf/2106.08657v2.pdf)                | RoBERTa-large | University of Illinois at Urbana-Champaign |  ACL2022   |                      64.79                      |                            -                             |
| [KD-DocRE](https://aclanthology.org/2022.findings-acl.132.pdf) | RoBERTa-large |   DAMO, National University of Singapore   |  ACL2022   |                      64.28                      |                          78.65                           |
| [SSAN](https://arxiv.org/pdf/2102.10249v1.pdf)                 | RoBERTa-large |                USTC, Baidu                 |  AAAI2021  |                      61.42                      |                            -                             |


### Biomedical
| Method                                          |  Model  |                    Org.                    |    Year    |  CDR  | GDA   |
| :---------------------------------------------- | :-----: | :----------------------------------------: | :--------: | :---: | ----- |
| [SAIS](https://arxiv.org/pdf/2109.12093v2.pdf)  | SciBERT |               CMU, Stanford                | NAACL 2022 | 79.0  | 87.1  |
| [EIDER](https://arxiv.org/pdf/2106.08657v2.pdf) | SciBERT | University of Illinois at Urbana-Champaign |  ACL2022   | 70.63 | 84.54 |
| [SSAN](https://arxiv.org/pdf/2102.10249v1.pdf)  | SciBERT |                USTC, Baidu                 |  AAAI2021  | 68.7  | 83.7  |

## Key Information Extraction
> **SROIE** : Scanned receipts OCR and information extraction <br>
> **CORD** is a Consolidated Receipt Dataset for Post-OCR Parsing. <br>
> **FUNSD** is a Form Understanding in Noisy Scanned Documents (FUNSD) comprises 199 real, fully annotated, scanned forms. <br>
> F1 for all.<br>
> default: BASE-MODEL

| Model                                                                                                                                                         |                               Org.                                |    Year     | Modality | [SROIE](https://rrc.cvc.uab.es/?ch=13) | [CORD](https://github.com/clovaai/cord)  | [FUNSD](https://guillaumejaume.github.io/FUNSD/) |
| :------------------------------------------------------------------------------------------------------------------------------------------------------------ | :---------------------------------------------------------------: | :---------: | :------: | :------------------------------------: | :---: | :---: |
| [GeoLayoutLM-Large](https://arxiv.org/pdf/2304.10759v1.pdf)                                                                                                   |                               DAMO                                |  CVPR 2023  |  T+L+V   |                   -                    | 97.97 | 92.86 |
| [LayoutLMv3](https://arxiv.org/pdf/2204.08387v3.pdf)                                                                                                          |          Sun Yat-sen University, Microsoft Research Asia          | ACM MM 2022 |  T+L+V   |                 95.30                  | 96.56 | 90.29 |
| [LILT](https://arxiv.org/pdf/2202.13669v1.pdf)                                                                                                                |                               SCUT                                |  ACL 2022   |  T+L+V   |                 97.65                  | 95.11 |   -   |
| [GenKIE](https://arxiv.org/pdf/2310.16131.pdf)                                                                                                                | University of Michigan, National University of Defense Technology | EMNLP 2023  |  T+L+V   |                 97.40                  | 95.75 | 83.45 |
| [DocFormer](https://openaccess.thecvf.com/content/ICCV2021/papers/Appalaraju_DocFormer_End-to-End_Transformer_for_Document_Understanding_ICCV_2021_paper.pdf) |                              AWS AI                               |  ICCV2021   |  T+L+V   |                   -                    | 96.33 | 83.34 |
| [StrucText](https://dl.acm.org/doi/10.1145/3474085.3475345)                                                                                                   |                               Baidu                               | ACM MM 2021 |  T+L+V   |                 96.88                  |   -   | 83.09 |

## Link Prediction
> MRR: mean reciprocal rank<br>
> Hits@k: H@k for brevity<br>

### WN18RR

| Model                                                            |      Method       |                    Org.                    |     Year     | MRR  | H@1  | H@3  | H@10 |
| :--------------------------------------------------------------- | :---------------: | :----------------------------------------: | :----------: | :--: | :--: | :--: | :--: |
| [MoCoSA](https://arxiv.org/pdf/2308.08204v1.pdf)                 | description-based |            Kuaishou Technology             |     2023     | 69.6 | 62.4 | 73.7 | 82.0 |
| [SimKGC](https://arxiv.org/pdf/2203.02167.pdf)                   | description-based | Microsoft Research Asia, Yuanfudao AI Lab  |   ACL 2022   | 67.1 | 58.7 | 73.1 | 81.7 |
| [CSPromp-KG](https://aclanthology.org/2023.findings-acl.729.pdf) |  structure-based  |      Nanyang Technological University      |   ACL 2023   | 57.5 | 52.2 | 59.6 | 67.8 |
| [KG-S2S](https://aclanthology.org/2022.coling-1.352.pdf)         | description-based |      Nanyang Technological University      | COLING 2022  | 57.4 | 53.1 | 59.5 | 66.1 |
| [NBFNet](https://arxiv.org/pdf/2106.06935v4.pdf)                 |                   |         Mila - Québec AI Institute         | NeurIPS 2021 | 55.1 | 49.7 | 57.3 | 66.6 |
| [C-LMKE](https://arxiv.org/pdf/2206.12617.pdf)                   | description-based |              Fudan University              |  IJCAI 2022  | 59.8 | 48.0 | 67.5 | 80.6 |
| [TuckER](https://aclanthology.org/D19-1522.pdf)                  |  structure-based  | University of Edinburgh, Samsung AI Centre | COLING 2019  | 47.0 | 44.3 | 48.2 | 52.6 |
