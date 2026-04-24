# LucaProt-Simplified

This repository provides a simplified implementation derived from Alibaba's [LucaProt](https://github.com/alibaba/LucaProt), a deep learning framework for protein function prediction from sequence and structural information.

LucaProt is used to predict the probability that a protein sequence is a viral RNA-dependent RNA polymerase (RdRP).

## Requirements

Required environment:

- python=3.9
- numpy~=1.24
- torch~=1.13
- transformers~=4.26
- fair-esm[esmfold]
- subword_nmt==0.3.8

Additional requirements:

- Linux is recommended.
- More than 18 GB of system memory is required to load the models.
- More than 18 GB of GPU memory is required for GPU inference.

## Usage

Run the CLI with a protein sequence:

```bash
python main.py --sequence MWVAWRTGLPGTWCPGVHANCVHNEIAALTKRSLAPLPCGPDPVLSSGVCEMYASLRHLARRYRGSQWSYLETAFSYSGAMRRRYVEAERSLSVDGPLGPLDWRLSAFLKAEKLGAAKDQKPRMIFPRSPRFNLVVASWLKPFEHWLWGFLTAKRLFGGSNTRVVGKGLGPVRRGNLIKRKFDSFADCVVFEVDGKAFEAHVTENHLVHERRIYKAAYPGCAGLAEVLEHQRFAGVTQNGVKFSRRGGRASGDFNTGMGNTLIMLAVTCGVLSRYQIKYDVLVDGDNALVFLERGASTAVIGSFYQNVLESSGFEMTLEKPVSYMEGIRFGRSAPLFLGPTRGWTMVREPESVLSGAYASHRWLREPSFGRRWVNGVARCELSLARGVPVLQAAALHVLLSTETAKKVPVEALSDWFVIGAWLAGAGDVIDVSREARASFERAFGISKEEQLVWERKLRSVAPGPVPGCVQHRPPSSWQLAEPGLYEAYIDAHI
```

You can also read the sequence from a file:

```bash
python main.py --sequence-file sequence.txt
```

Optional paths can be provided if your model assets are stored elsewhere:

```bash
python main.py \
  --sequence-file sequence.txt \
  --ssfn-model-dir /path/to/ssfn_model \
  --tokenizer-dir /path/to/ssfn_tokenizer \
  --bpe-codes-path /path/to/protein_codes_rdrp_20000.txt
```

You can also override the ESM model name:

```bash
python main.py --sequence-file sequence.txt --esm-model-name esm2_t36_3B_UR50D
```

The first run downloads the selected ESM model used for structure-aware embeddings. For `esm2_t36_3B_UR50D`, the download size is about 6 GB, so it may take some time.

## Output

Example output:

```text
Protein ID: protein_1
Probability of being a viral RdRP: 1.000000
```

This means the input protein sequence is predicted to be a viral RdRP with probability 1.0.
