# Data card

Every source Kwan uses, who made it, how to get it, what its licence allows, and what Kwan does with it.

Kwan stores **none** of the raw data below in its repository. The curriculum PDFs and the learner datasets are downloaded by scripts when you build or evaluate Kwan. The sizes and counts on this page describe each source as it is published. Kwan's own results are on the [Evidence](evidence.md) page.

## Curriculum documents

Kwan's skill map follows the NaCCA Standards-Based Curriculum for Mathematics.

| Id | Document | Where it comes from |
|---|---|---|
| C1 | NaCCA Mathematics, Basic 1 to Basic 3 (2019) | `https://curriculumresources.edu.gh/basic/wp-content/uploads/2026/05/MATHEMATICS-BASIC-B1-B3-2.pdf` |
| C2 | NaCCA Mathematics, Basic 4 to Basic 6 (2019) | `https://curriculumresources.edu.gh/basic/wp-content/uploads/2026/05/MATHEMATICS-BASIC-B4-B6-1.pdf` |

- **Owner:** © NaCCA, Ministry of Education, Ghana, 2019.
- **How Kwan uses them:** `kwan curriculum fetch` downloads both PDFs and records the address, size, SHA-256 hash and date of each. `kwan curriculum parse` reads the indicator codes, names, descriptions and exemplars.
- **What is committed:** indicator codes, class, strand and sub-strand numbers, and Kwan's own short labels. The descriptions and exemplars are NaCCA's text. They stay on your machine and are not committed unless NaCCA gives written permission. See [Governance](governance.md).

!!! note "Not an endorsement"
    Kwan uses the NaCCA Standards-Based Curriculum to organise skills. Kwan is not reviewed, approved or endorsed by NaCCA or the Ministry of Education.

## Learner datasets

The learner model is tested on public research data: answers that real pupils gave to maths questions. Every dataset here downloads without an account.

The column `commercial_params_ok` matters most. It says whether numbers fitted on a dataset (model parameters, item difficulties) may be shipped inside a Kwan bundle. Where it is `false`, Kwan uses the dataset only to measure how good the model is. The bundle builder refuses parameters that come from such a dataset.

### Summary

| Id | Dataset | Who and where | Licence | Role in Kwan | `commercial_params_ok` |
|---|---|---|---|---|---|
| L1 | ASSISTments 2009 skill-builder (corrected) | United States, middle-school maths | Research use | Learner model evaluation | `true` |
| L2 | Eedi NeurIPS 2020, task 3/4 | United Kingdom, maths | Research use, cite | Learner model evaluation | `true` |
| L3 | Eedi NeurIPS 2020, task 1/2 (full) | United Kingdom, maths | Research use, cite | Evaluation at scale; item-difficulty priors | `true` |
| L4 | AMMORE | African pupils in Ghana, Nigeria, Kenya and South Africa | CC BY-NC 4.0 | Evaluation, Ghana-only evaluation, fitting the forgetting rate for evaluation | `false` |
| L5 | ASSISTments 2012-13, 2015 and 2017 | United States | Research use | Wider evaluation; standard benchmarks for deep models | `true` |
| L6 | XES3G5M | China, Grade 3 maths | MIT | Primary-age evaluation; deep model benchmark | `true` |
| L7 | Junyi Academy 2015 | Taiwan, maths from kindergarten to Grade 12 | Non-commercial | Checking the method that suggests skill links; evaluation | `false` |

### L1. ASSISTments 2009 skill-builder (corrected)

| | |
|---|---|
| Who and where | Students in the United States using the ASSISTments online tutor for middle-school maths |
| Size | 259k answers from 4,163 students |
| Access | `http://base.ustc.edu.cn/data/ASSISTment/2009_skill_builder_data_corrected.zip` (9 MB) |
| Licence | Research use |
| Citation | Feng, Heffernan and Koedinger (2009) |
| Role | Learner model evaluation |
| `commercial_params_ok` | `true` |

### L2. Eedi NeurIPS 2020, task 3/4

| | |
|---|---|
| Who and where | Pupils in the United Kingdom answering multiple-choice maths questions on the Eedi platform |
| Size | 1.38M answers from 4,918 students |
| Access | `https://dqanonymousdata.blob.core.windows.net/neurips-public/data.zip` (657 MB) |
| Licence | Research use, with citation |
| Citation | Wang et al. (2020), arXiv:2007.12061 |
| Role | Learner model evaluation |
| `commercial_params_ok` | `true` |

### L3. Eedi NeurIPS 2020, task 1/2 (full)

| | |
|---|---|
| Who and where | Pupils in the United Kingdom, same platform as L2 |
| Size | 15.9M answers from 118,971 students on 27,613 questions |
| Access | The same zip file as L2 |
| Licence | Research use, with citation |
| Citation | Wang et al. (2020), arXiv:2007.12061 |
| Role | Learner model evaluation at scale; starting values for item difficulty |
| `commercial_params_ok` | `true` |

### L4. AMMORE

The African Middle-school Math Open REsponse dataset, from Rori, a maths tutor on WhatsApp run by Rising Academies.

| | |
|---|---|
| Who and where | African pupils in Ghana, Nigeria, Kenya and South Africa, in grades 6 to 9 |
| Size | 53,031 answers (50,507 marked right or wrong) from 2,388 students on 1,601 questions in 151 micro-lessons, with timestamps |
| Access | `https://github.com/owenhenkel/ammore_dataset` |
| Licence | CC BY-NC 4.0 (non-commercial) |
| Citation | Henkel et al. (2025), *Journal of Learning Analytics* |
| Role | Learner model evaluation, a Ghana-only result, and fitting the forgetting rate for evaluation |
| `commercial_params_ok` | `false` |

AMMORE is the only dataset here with answers from Ghanaian pupils. Because its licence is non-commercial, nothing fitted on it is shipped in a Kwan bundle.

### L5. ASSISTments 2012-13, 2015 and 2017

| | |
|---|---|
| Who and where | Students in the United States using ASSISTments |
| Size | Downloads of 576 MB (2012-13), 3 MB (2015) and 68 MB (2017) |
| Access | `http://base.ustc.edu.cn/data/ASSISTment/` (files `2012-2013-data-with-predictions-4-final.zip`, `2015_100_skill_builders_main_problems.zip` and `anonymized_full_release_competition_dataset.zip`) |
| Licence | Research use |
| Citation | Cite the ASSISTments project as the data provider asks |
| Role | Wider learner model evaluation; standard benchmarks for deep knowledge tracing |
| `commercial_params_ok` | `true` |

### L6. XES3G5M

| | |
|---|---|
| Who and where | Grade 3 pupils in China, maths |
| Size | More than 5M interactions from 18k students on 7,652 questions, with question text and skill routes |
| Access | `https://github.com/ai4ed/XES3G5M` (the files are on Google Drive, linked from that page) |
| Licence | MIT |
| Citation | Liu et al. (2023), NeurIPS Datasets and Benchmarks |
| Role | Evaluation on primary-age pupils; deep knowledge tracing benchmark |
| `commercial_params_ok` | `true` |

### L7. Junyi Academy 2015

| | |
|---|---|
| Who and where | Learners in Taiwan on the Junyi Academy platform, maths from kindergarten to Grade 12 |
| Size | A problem log (a 1.3 GB download), plus prerequisite links and relationship notes made by experts |
| Access | `http://base.ustc.edu.cn/data/JunyiAcademy_Math_Practicing_Log/junyi.rar` (needs `unrar`). Original source: PSLC DataShop, dataset 1198. |
| Licence | Non-commercial |
| Citation | Chang, Hsu and Chen (2015), EDM |
| Role | Ground truth for checking the method that suggests skill links; learner model evaluation |
| `commercial_params_ok` | `false` |

Junyi comes with prerequisite links drawn by experts. Kwan runs its link-suggestion method on Junyi first and compares the result with those expert links, before using the method on the NaCCA skills.

## What may be committed, fetched, or only evaluated

| Material | Commit to git | Fetch at build time | Evaluation only | Notes |
|---|---|---|---|---|
| Kwan code, tests, docs | Yes (MIT) | | | |
| Kwan-written practice settings, short labels, translations, templates | Yes (CC BY 4.0) | | | Written by Kwan contributors, never copied from NaCCA |
| Indicator codes, class, strand numbers, graph edges | Yes | | | Structural facts |
| NaCCA indicator descriptions and exemplars | Only after written permission | Yes, from the official PDFs | | Each page of the PDFs carries "© NaCCA, Ministry of Education 2019" |
| Raw learner datasets (L1 to L7) | Never | Yes (download scripts) | Yes | |
| Parameters fitted on non-commercial data (L4, L7) | Never in shipped bundles | | Yes | Shipped parameters are fitted on L1 to L3, L5 and L6 |
| Aggregate metrics (AUC and similar) | Yes | | | Results, not data |

## Privacy

- The learner datasets were made anonymous by their publishers. Kwan does not try to identify anyone in them and does not join them to other data.
- Tests use small made-up data. No real row from any dataset is committed, not even as a test fixture.
- Kwan itself collects nothing. See the children's data section of the [Model card](model-card.md).

## Not in 1.0

Ghana survey datasets that need a free account (World Bank STARS, USAID Early Grade Mathematics Assessment) are planned for later. See the [Roadmap](roadmap.md).

## How to cite

If you publish work that uses Kwan with one of these datasets, cite the dataset as its authors ask, and cite Kwan using the `CITATION.cff` file in the repository.

- ASSISTments 2009 skill-builder data. Feng, Heffernan and Koedinger (2009).
- Eedi NeurIPS 2020 Education Challenge. Wang et al. (2020), arXiv:2007.12061.
- AMMORE, African Middle-school Math Open REsponse dataset, from Rori by Rising Academies. Henkel et al. (2025), *Journal of Learning Analytics*. CC BY-NC 4.0.
- XES3G5M. Liu et al. (2023), NeurIPS Datasets and Benchmarks. MIT.
- Junyi Academy 2015. Chang, Hsu and Chen (2015), EDM. Non-commercial.

*This page is not legal advice.*
