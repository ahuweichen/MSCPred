# Prediction of human pathogenic start loss variants based on multi-level language model embedding feature fusion

MSCPred is a prediction method specifically designed to identify pathogenic start loss variants. This method leverages multi-level embeddings derived from pre-trained language models across four biological modalities—genomic DNA sequence, RNA transcript context, protein-level representation, and epigenetic modification profiles—and integrates them through a fusion module combining TextCNN, bilinear attention, and feature concatenation. Unlike existing deep learning approaches that primarily rely on embedding features quantified by DNA language models and often overlook the biological consequences of mutations on transcription and translation, MSCPred explicitly addresses this limitation by fusing multi-level language model embeddings that capture regulatory and functional impacts across the central dogma. This work provides a more comprehensive framework for accurately deciphering the pathogenicity of start loss variants in the human genome.


## Basic requirements
To install dependencies, create a new conda environment:
```bash
conda env create -f MSCPred.yml
```
We run the program on the Ubuntu 22.04.4 LTS system.

## Quick start

### Input format
MSCPred supports variants in CSV format as input. The input file should contain at least 7 columns in the header as follows. [Sample file](./data/test.csv)

|  Chr  | Pos |  Ref  |  Alt  |  Label  |   ...  |
| ----- | --- | ----- | ----- | ------- |  ----- |


### Quantitative features based on genetic language model
For more information about GPN-MSA, see https://doi.org/10.1101/2023.10.10.561776 and https://github.com/songlab-cal/gpn.

For more information about HyenaDNA, see https://doi.org/10.52202/075280-1872 and https://github.com/HazyResearch/hyena-dna.

For more information about ERNIE-RNA., see https://doi.org/10.1101/2024.03.17.585376 and https://github.com/Bruce-ywj/ERNIE-RNA.

For more information about CaLM, see https://doi.org/10.1038/s42256-024-00791-0 and https://github.com/oxpig/CaLM.

For more information about DanQ, see https://doi.org/10.1093/nar/gkw226 and http://github.com/uci-cbcl/DanQ.

### Pathogenicity prediction
```bash
conda activate MSCPred
cd MSCPred
python main.py
```

### Output format
This program produces two output files.

The first file is 'sta_test.csv' [Sample file](./result/sta_test.csv). It provides a comprehensive overview of MSCPred's predictive performance on the dataset, encompassing various metrics such as recall (SEN), specificity (SPE), precision (PRE), F1-score (F1), Matthew's correlation coefficient (MCC), accuracy (ACC), the area under the receiver operating characteristic curve (AUC), and the area under the precision-recall curve (AUPR).

The second file is 'test_pred_score.txt' [Sample file](./result/test_pred_score.txt). It contains a list of scores assigned by MSCPred to each variant in the dataset, with the sample order matching that of the input data in 'data/test.csv'. The scoring threshold for MSCPred is established at 0.5, whereby variants scoring below 0.5 are designated as benign and those scoring above 0.5 are identified as pathogenic.

