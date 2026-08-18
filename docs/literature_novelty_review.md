# Aptafind Literature and Novelty Review

## Status

This is a living, preliminary novelty review initiated before implementation of the modern Aptafind pipeline.

It is not yet a publication-ready systematic review. A later revision must record bibliographic databases, complete search strings, search dates, screening criteria, exclusions, duplicate handling, and citation chaining.

Last updated: 2026-08-13.

## Revised research question

> Do multi-round SELEX trajectories contain transferable information about aptamer-ligand interactions beyond endpoint binding labels alone?

## Why the scope changed

An initial proposal focused on using early-round steroid SELEX data to predict later enrichment and generate a compact candidate library. Preliminary literature searching showed that early-round evaluation, multi-round enrichment modeling, sequence-structure trend analysis, and steroid-specific guided-library design are already active areas with directly relevant methods.

Aptafind should not claim novelty for early-to-late enrichment prediction alone.

## Preliminary search themes

The initial search covered combinations of:

- aptamer and early-round SELEX prediction
- HT-SELEX enrichment modeling
- sequence-structure selection trends
- aptamer candidate ranking
- small-molecule aptamer interaction prediction
- steroid aptamer generation
- hydrocortisone and testosterone SELEX
- aptamer target conditioning
- molecular graphs and cross-attention
- sequence-family and target-disjoint evaluation

Sources discovered through web indexing included PubMed Central, journal sites, arXiv, bioRxiv, OpenReview, GitHub, Hugging Face, and repository records. The formal review must repeat these searches in structured bibliographic databases.

## Prior-work matrix

| Work | Primary task | Data/target scope | Relevant contribution | Relationship to Aptafind |
|---|---|---|---|---|
| [AptaTRACE](https://doi.org/10.1186/s13059-016-1094-1) | Discover sequence-structure motifs whose distributions change across HT-SELEX rounds | HT-SELEX experiments | Explicit sequence-structure selection trends | Establishes that multi-round structural trend analysis is not novel by itself |
| [MPBind](https://doi.org/10.1093/bioinformatics/btu491) | Predict binding potential of SELEX-derived aptamers | SELEX sequence motifs | Meta-motif statistical ranking beyond raw abundance | Required classical baseline or comparison |
| [Generative and interpretable ML for aptamer selection](https://doi.org/10.1371/journal.pcbi.1010561) | Learn sequence fitness and generate/diversify aptamers | Multi-round in-vitro selection | Restricted Boltzmann machine representation of evolving sequence pools | Early-to-late fitness modeling and generation are established directions |
| [RaptGen](https://doi.org/10.1038/s43588-022-00249-6) | Generate aptamers from HT-SELEX sequence families | RNA HT-SELEX | VAE with profile-HMM decoder, latent sampling, and optimization | Strong sequence-generative comparator |
| [AptaDiff](https://doi.org/10.1093/bib/bbae517) | Generate and optimize aptamer sequences | Primarily target-specific HT-SELEX datasets | VAE-derived motif space plus discrete diffusion and affinity optimization | Establishes discrete diffusion for aptamers; Aptafind must add distinct target-transfer evidence |
| [DeepAptamer](https://pmc.ncbi.nlm.nih.gov/articles/PMC11787022/) | Predict high-affinity candidates from SELEX | Target-specific multi-round ssDNA SELEX | CNN/BiLSTM using sequence and DNA conformational features | Sequence-plus-structure enrichment/affinity prediction is not novel alone |
| [RaptScore](https://doi.org/10.1093/nar/gkaf1480) | Rank aptamers, including early-round candidates | Multiple SELEX datasets | Pretrained masked DNA language model with experimental comparison to frequency/enrichment | Early-round ranking is already directly addressed |
| [Molecular Cues to Smart Sequences](https://openreview.net/forum?id=tWzfvV0fvu) | Refine early-round small-molecule aptamers toward later-round patterns | Theophylline proof of concept | Ligand descriptors plus predicted structure and language-model editing | Overlaps strongly with early-to-late target-aware refinement; currently workshop-level evidence |
| [DL-SELEX](https://doi.org/10.1093/bib/bbaf680) | Design guided steroid libraries and summarize post-SELEX candidates | Steroid family; hydrocortisone and testosterone validation | AptaVAE initial-library design, AptaClux sequence/structure clustering, ITC validation | Directly occupies steroid guided-library and HT-SELEX analysis space |
| [AptaBench](https://openreview.net/forum?id=MsZa6NgqWJ) | Benchmark aptamer-small-molecule binding and affinity prediction | Current hosted release: 6,289 pairs, 1,610 aptamers, 942 ligands | Experimentally grounded active/inactive labels, affinity values, molecule- and aptamer-disjoint splits | Becomes Aptafind's likely endpoint benchmark rather than rebuilding the same task independently |
| [AptaTrans](https://doi.org/10.1186/s12859-023-05577-6) | Aptamer-protein interaction prediction and candidate recommendation | Protein targets and RNA aptamers | Pretrained target/aptamer encoders and interaction modeling | Relevant architectural precedent but not small-molecule/steroid evidence |
| [AptaBLE preprint](https://doi.org/10.64898/2026.01.06.698056) | Aptamer-protein prediction and generation | Protein targets and ssDNA aptamers | Symmetric bidirectional cross-attention and de novo generation | Shows cross-attention is not generally novel; small-molecule graph conditioning remains a distinct setting to test |

## Preliminary novelty assessment

### Not sufficient as standalone novelty

- Predicting which early-round sequences enrich later
- Ranking candidates using frequency or enrichment
- Combining sequence and predicted structure
- Using a VAE to generate aptamer-like sequences
- Applying discrete diffusion to aptamer generation
- Using a language model to rank early-round candidates
- Designing a guided steroid library
- Using cross-attention for target-aptamer interaction prediction in general
- Creating another general aptamer-small-molecule endpoint benchmark without a clear improvement over AptaBench

### Candidate Aptafind contribution

The most defensible current contribution is the combination of:

1. Multi-round hydrocortisone and testosterone SELEX trajectory pretraining
2. Endpoint small-molecule interaction fine-tuning using a frozen, leakage-audited AptaBench revision
3. Steroid-focused molecule-disjoint evaluation
4. Sequence-family-disjoint evaluation beyond exact sequence holdout
5. Chemically related steroid hard negatives grounded in measured inactivity or cross-reactivity
6. Steroid atom graphs combined with ssDNA nucleotide/secondary-structure graphs
7. Atom-to-nucleotide bidirectional cross-attention
8. A controlled endpoint-only versus trajectory-pretrained ablation
9. Prospective experimental validation if collaboration becomes available

No individual item should be claimed as novel until the formal review verifies it. The potential contribution lies in the research question, evidence integration, evaluation rigor, and experimental scope.

## Central ablation

```text
Model A
AptaBench endpoint interaction training only

Model B
Steroid-SELEX trajectory pretraining
    -> same AptaBench endpoint interaction training

Model C
Matched pretraining using the same sequences and similar compute
without intact trajectory supervision
    -> same AptaBench endpoint interaction training

Controls
- identical downstream architecture
- identical endpoint folds
- identical hyperparameter-selection policy
- identical evaluation metrics
- cross-source sequence-family deduplication
- repeated seeds or folds
```

Primary endpoint:

> Does Model B improve held-out-steroid binding and specificity ranking relative to both Model A and the matched Model C control?

Model C is required because Model B receives additional sequence exposure and optimization. Possible controls include shuffled round labels, disrupted sequence trajectories, or a sequence-only self-supervised objective. This helps distinguish information carried by trajectory order and enrichment from benefits caused by generic pretraining.

Secondary endpoints:

- Does improvement persist under sequence-family-disjoint evaluation?
- Is improvement limited to hydrocortisone/testosterone-like ligands?
- Do predicted secondary-structure edges add value beyond sequence alone?
- Does atom-to-nucleotide cross-attention outperform simple embedding concatenation?
- Does pretraining help measured inactive/active discrimination as well as affinity ranking?

## Critical confounders

### Cross-source sequence overlap

DL-SELEX candidate sequences, source literature, and AptaBench records may overlap. Pretraining sequences or close family members must not appear in downstream evaluation folds without explicit disclosure.

### Target leakage

Hydrocortisone and testosterone trajectory pretraining exposes the model to those targets. They cannot later be described as unseen targets. Held-out-steroid evaluation should use other steroids, while separate experiments measure transfer to and from the pretraining targets.

### Constructed negatives

AptaBench reports experimentally grounded inactive labels. These are preferable to random target mismatches. Constructed steroid mismatches may be used only as a separate contrastive condition.

### Dataset revision drift

AptaBench counts changed across manuscript revisions. Aptafind must freeze a specific file revision, commit hash where available, checksums, and split definitions.

### Assay heterogeneity

Endpoint affinity and activity labels arise from different publications, assays, buffers, temperatures, and thresholds. Apparent predictive performance may partly reflect source or assay identity.

### Selection bias

HT-SELEX enrichment reflects selection, PCR, sequencing, nonspecific binding, and library design. Trajectory pretraining must not be interpreted as direct affinity supervision.

## Formal review protocol still required

Before making a novelty claim in a paper, complete:

1. Structured searches in PubMed, Web of Science or Scopus where accessible, Crossref, arXiv, bioRxiv, and OpenReview.
2. Exact search strings and search dates.
3. Title/abstract screening followed by full-text screening.
4. Forward and backward citation chaining from DL-SELEX, AptaBench, RaptScore, AptaDiff, RaptGen, and AptaTRACE.
5. Separate inclusion tables for methods, datasets, experimental studies, and reviews.
6. A reproducibility/code/data availability assessment.
7. Patent and commercial-landscape review as a separate exercise.
8. Periodic search updates because this field is developing rapidly.

## Current decision

Do not freeze the final model or publication claim yet.

Proceed with:

1. Acquire and freeze the current AptaBench release.
2. Profile its DNA-only steroid subset and experimental negatives.
3. Audit sequence-family and source overlap.
4. Verify what unique supervision the steroid HT-SELEX rounds add.
5. Reassess feasibility and novelty before implementing the graph/cross-attention ranker.
