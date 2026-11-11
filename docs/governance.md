# Governance

Who looks after Kwan, how decisions are made, and where Kwan stands with NaCCA.

## Maintainers

| Name | GitHub | Role |
|---|---|---|
| Rofy Ray | [@rofyray](https://github.com/rofyray) | Maintainer |

Maintainers review and merge pull requests, cut releases, send out and import teacher review sheets, look after the Code of Conduct, and keep the rules in this document.

**Becoming a maintainer.** A contributor who has done steady, careful work over several months, on code or on teacher review, may be invited by the current maintainers. Kwan wants at least one maintainer who teaches or has taught in a Ghanaian primary school.

## How decisions are made

Kwan keeps this simple.

1. **Everyday changes** (bug fixes, tests, docs, small features) are decided in the pull request. They need green CI and one review. If nobody objects, the change goes in.
2. **Bigger changes** start as an issue before any code is written, so people can comment. This covers changes to the public Python interface, the REST API, the bundle format, the learner model or the recommender.
3. **Teaching content** is decided by teachers. A skill link suggested by the decision model, a practice setting, a short label or a translation ships only when a teacher reviewer has accepted it. A maintainer cannot overrule a reviewer by editing the data. If a maintainer disagrees, the item goes back to review.
4. **When people disagree**, the maintainers listen to everyone, then decide and write the reason on the issue. With one maintainer, that person decides.
5. **Some rules are not up for a vote.** They can only change through a public issue that stays open for comment for at least two weeks:
    - Kwan stores no personal data about pupils.
    - Numbers come only from `data/eval_results.json`.
    - Raw datasets and NaCCA curriculum text are not committed.
    - Parameters fitted on non-commercial data are not shipped.
    - The licences: MIT for code, CC BY 4.0 for Kwan-written content.
    - Kwan never claims approval or endorsement by NaCCA or the Ministry of Education.

Conduct problems are handled under the [Code of Conduct](https://github.com/rofyray/kwan/blob/main/CODE_OF_CONDUCT.md), in private.

## Releases and versions

- The `kwan` package follows semantic versioning.
- Each subject bundle has its own `bundle_version`, for example `math-1.0.0`. Learner state records the bundle version it was built with.
- A release is cut only when every check in CI is green and `data/eval_results.json` is up to date.

## Kwan and NaCCA

Kwan uses the NaCCA Standards-Based Curriculum to organise skills. Kwan is not reviewed, approved or endorsed by NaCCA or the Ministry of Education.

### What Kwan uses, and how

- The curriculum documents for Mathematics, Basic 1 to Basic 6 (2019), are © NaCCA, Ministry of Education, Ghana.
- This repository commits only **structural facts** (indicator codes, class, strand and sub-strand numbers, links between skills) and **Kwan-written text** (short labels, practice settings, translations).
- The **indicator descriptions and exemplars** are NaCCA's text. Kwan downloads them from the official PDFs when someone builds Kwan on their own machine. They are not stored in this repository and are not in the bundles the project publishes.
- Without NaCCA's text, Kwan runs in "labels" mode and shows its own short labels.
- The [Data card](data-card.md) lists every source Kwan uses, with its licence and what Kwan may do with it.

### Permission status

**Status: not yet requested.**

The maintainer will send the request below to NaCCA. Until NaCCA gives written permission, nothing changes in what the repository commits. If permission is given, the reply and any conditions will be recorded here before any curriculum text is committed.

### Correspondence log

| Date | From | To | Channel | Summary | Result |
|---|---|---|---|---|---|
| | | | | | |

*No correspondence yet.*

### Permission request (ready to send)

To be sent from the maintainer's own email address to NaCCA at `info@nacca.gov.gh`. Fill in the name and the email address in the last line before sending.

> **Subject:** Request for permission to use curriculum text in an open-source learning tool for Ghanaian pupils
>
> Dear Director-General,
>
> I am writing to ask for written permission to reproduce the learning indicators and exemplars from the 2019 Standards-Based Curriculum for Mathematics and English Language (Basic 1 to 6) in Kwan, a free, open-source tool that helps Ghanaian pupils choose what to practise next.
>
> Kwan organises practice around the official indicator codes so that every suggestion matches what pupils learn in class. We would like to include the indicator statements and exemplars in our public code repository and in tools built on Kwan, always credited to NaCCA, with a clear statement that NaCCA has not reviewed or approved the tool. We will not sell the curriculum text, alter its meaning, or use NaCCA's name or logo to suggest endorsement.
>
> We would welcome any conditions you wish to set, and we would be glad to show the tool to your team or share it with teachers for feedback.
>
> Yours faithfully,
>
> [Name], Maintainer, Kwan · [email] · github.com/rofyray/kwan

### If permission is refused or never answered

Kwan keeps working as it does today. Labels mode needs no NaCCA text, and the full text remains something each user downloads from the official source at build time.

*This page is not legal advice.*

## Changing this document

Open a pull request. Changes to the "not up for a vote" list and to the NaCCA section follow rule 5 above.
