# 📄 *"When Data is Scarce, Prompt Smarter"... Approaches to Grammatical Error Correction in Low-Resource Settings*

*Somsubhra De, Harsh Kumar, Arun Prakash A*  

The repo contains the codes and data for the **BHASHA Workshop** shared task-1 on Grammatical Error Correction (Indic-GEC) in Indic & low-resource languages, co-located with **IJCNLP-AACL 2025**.

[![arXiv](https://img.shields.io/badge/arXiv-2511.20120-red.svg)](https://arxiv.org/abs/2511.20120)
[![PDF](https://img.shields.io/badge/PDF-Paper-blue)](https://arxiv.org/pdf/2511.20120)
[![CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

---

TL;DR: Prompt-based adaptation of LLMs with few-shot tuning significantly improves grammatical error correction in low-resource Indic languages.

Abs:

> Grammatical error correction (GEC) is an important task in Natural Language Processing that aims to automatically detect and correct grammatical mistakes in text. While recent advances in transformer-based models and large annotated datasets have greatly improved GEC performance for high-resource languages such as English, the progress has not extended equally. For most Indic languages, GEC remains a challenging task due to limited resources, linguistic diversity and complex morphology. In this work, we explore prompting-based approaches using state-of-the-art large language models (LLMs), such as GPT-4.1, Gemini-2.5 and LLaMA-4, combined with few-shot strategy to adapt them to low-resource settings. We observe that even basic prompting strategies, such as zero-shot and few-shot approaches, enable these LLMs to substantially outperform fine-tuned Indic-language models like Sarvam-22B, thereby illustrating the exceptional multilingual generalization capabilities of contemporary LLMs for GEC. Our experiments show that carefully designed prompts and lightweight adaptation significantly enhance correction quality across multiple Indic languages. We achieved leading results in the shared task--ranking 1st in Tamil (GLEU: 91.57) and Hindi (GLEU: 85.69), 2nd in Telugu (GLEU: 85.22), 4th in Bangla (GLEU: 92.86), and 5th in Malayalam (GLEU: 92.97). These findings highlight the effectiveness of prompt-driven NLP techniques and underscore the potential of large-scale LLMs to bridge resource gaps in multilingual GEC.  

---

## Citation

If you find this useful, please cite:

```bibtex
@misc{de2025whendatascarceprompt,
      title={"When Data is Scarce, Prompt Smarter"... Approaches to Grammatical Error Correction in Low-Resource Settings}, 
      author={Somsubhra De and Harsh Kumar and Arun Prakash A},
      year={2025},
      eprint={2511.20120},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2511.20120}, 
}
```