# Roadmap

Kwan 1.0 covers **NaCCA Mathematics, Basic 1 to Basic 6**. Kwan 1.0.0 is on PyPI and the demo is live. This page lists what 1.0 did not reach, and then the work that comes later. None of the later work has a date, and none of it has been started.

## What 1.0 did not reach

These were in the plan for 1.0 and are not done. The code paths exist; what is missing is people's time.

- **Reviewed practice settings.** Every indicator has drafted settings and every draft passes the automatic checks, but no teacher has reviewed one yet. Kwan ships the general strand settings by default until that changes. The [Evidence](evidence.md) page counts the reviewed settings.
- **Translations.** The Twi, Ewe and Ga pipeline (machine draft, teacher review, import) is built. No reviewed translation ships.
- **Accepted augmented links.** The decision model, the Junyi benchmark and the review sheets work end to end, but the local model tried first is below the precision bar and its proposals are pending review. The shipped map holds rule-based links only. A stronger decision model is the first thing to try.
- **NaCCA permission.** The request for written permission to republish curriculum text has not been sent. See [Governance](governance.md).

If you teach in Ghana, the first three items are yours to move. See [CONTRIBUTING.md](https://github.com/rofyray/kwan/blob/main/CONTRIBUTING.md).

## Later, not in 1.0

Kwan is built to carry any NaCCA subject. Each new one is mostly a data task: a curriculum layout file, a skill map and practice settings.

### English Language, Basic 1 to Basic 6

The next subject. The official NaCCA English Language documents for Basic 1 to 3 and Basic 4 to 6 use the same kind of indicator codes as Mathematics. Kwan 1.0 already leaves room for them:

- skill ids carry a subject prefix (`math:` today, `eng:` for English);
- the curriculum parser reads its document layout from a settings file, so a new subject is mostly a data task.

### JHS Mathematics, Basic 7 to Basic 9

From the NaCCA Common Core Programme. That document states that all rights are reserved, so Kwan would show its own short labels only, unless permission is granted.

### Calibration on Ghana survey data

Two surveys hold maths answers from Ghanaian primary pupils. Both need a free account to download, which is why they are not in 1.0.

- World Bank STARS, 2018 and 2019: item-level maths answers from Ghanaian pupils in P4 to P6. Planned use: testing predictions on questions the model has not seen, and better starting profiles for the simulator.
- USAID Ghana Early Grade Mathematics Assessment, 2017 to 2018. Planned use: starting values for lower primary.

### A TypeScript port

For apps and for offline use on phones and tablets. Kwan 1.0 keeps its arithmetic simple on purpose (half-up rounding, CRC32 for picks, a small portable random number generator), so that a port in another language can give exactly the same recommendations.

### Platform integrations

Connections to learning platforms, each planned under its own document.

### Misconception tags

Tags for common mistakes, using Eedi's "Mining Misconceptions in Mathematics" data, so that Kwan could say which mistake a pupil is making and which skill it points back to.

## Always open

These are not waiting for a later version. Help is welcome at any time.

- **More teacher review:** skill links, practice settings and short labels, from teachers in every region.
- **More translations:** reviewed Twi, Ewe and Ga settings for Basic 1 to Basic 3.
- **Names and places:** additions to the [context style guide](context-style-guide.md) from groups and regions that are thin there today.
- **Curriculum fixes:** any indicator that Kwan has wrong.

See [CONTRIBUTING.md](https://github.com/rofyray/kwan/blob/main/CONTRIBUTING.md) to take part.

## How the roadmap changes

Ideas start as an issue on GitHub. Bigger changes follow the steps in [Governance](governance.md).

---

Kwan uses the NaCCA Standards-Based Curriculum to organise skills. Kwan is not reviewed, approved or endorsed by NaCCA or the Ministry of Education.
