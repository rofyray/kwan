# Teacher review guide

Kwan is checked by people who teach. This guide explains what a reviewer does, what each decision means, and how you are credited. You do not need to write code or use GitHub beyond opening one issue.

## What reviewers check

| Sheet | The question you answer |
|---|---|
| **Skill links** | Would a pupil really need skill A before skill B? |
| **Practice settings** | Is this "Try it in Ghana" line true to Ghana, right for the skill, and easy to read? |
| **Short labels** | Does this short name say what the skill is, in plain words? |
| **Translations** | Does the Twi, Ewe or Ga version say the same thing, the way a teacher would say it in class? |

## How a review works

1. **Ask for a sheet.** Open a ["Teacher review" issue](https://github.com/rofyray/kwan/issues/new?template=teacher-review.yml). Say which classes you teach, your region, and what you would like to review. Please do not put a phone number or an email address in the issue. It is public.
2. **Receive the sheet.** A maintainer replies and sends you a review sheet. It is a CSV file, which opens in Google Sheets, Excel or LibreOffice. A first sheet has about 25 to 100 rows.
3. **Choose a reviewer id.** This is a short code that stands for you in Kwan's data, for example your initials and a number. It is not your name. Write the same id on every row you review.
4. **Review each row.** Read it, choose a decision, and add a note when it helps. Leave a row blank if you are not sure. A blank row stays "pending" and nothing ships from it.
5. **Send the sheet back** the same way you received it. Keep the file as CSV and do not change the column names or the order of the columns.
6. **Import.** A maintainer imports your decisions. Your reviewer id and the date are saved next to each one, so anyone can see which items a teacher has checked.

Only reviewed work ships. A skill link suggested by the decision model, a practice setting, a short label or a translation reaches a Kwan release only after a reviewer has accepted it.

!!! note "Keep the sheet private"
    A review sheet may show indicator descriptions from the NaCCA curriculum so that you can judge the skill. That text belongs to NaCCA and the Ministry of Education. Use the sheet for your review and do not publish it or pass it on.

## Skill links

Kwan has a map of skills. A link from A to B means "learn A before B". Some links come from the order of the curriculum. Others are suggested by a decision model that reads the two skills. Those suggestions are the ones you review.

**Columns in the sheet**

| Column | What it holds |
|---|---|
| A | The code of the earlier skill, for example `B3.1.2.1.1` |
| B | The code of the later skill |
| Labels | The short label of A and of B |
| Descriptions | What each skill asks the pupil to do |
| Probability | How sure the model is, from 0 to 1. Treat it as a hint and judge the link yourself. |
| Decision | **You fill this in:** `accepted` or `rejected` |
| Reviewer id | Your reviewer id |
| Note | Optional. A few words on why. |

**The one question: would a pupil really need A before B?**

Accept the link when:

- a pupil who cannot yet do A would be stuck when learning B;
- you would go back and teach A first if a pupil struggled with B.

Reject the link when:

- B can be learned without A, even if the two are about the same topic;
- A is helpful for B but not needed;
- the order is the wrong way round (B is needed for A);
- A and B are really the same skill at the same level;
- A sits so far below B that many skills lie between them. The link should go through those skills instead.

Two examples in plain words. "Count to 100" before "Add two numbers within 100": accept, because a pupil cannot add what she cannot count. "Name 2D shapes" before "Add two numbers within 100": reject, because one is not needed for the other.

There is no "edit" for a link. If you think A should point to a different skill, reject the row and write the better link in the note.

## Practice settings

Read the [context style guide](context-style-guide.md) first. It lists real places in all 16 regions, names, the coins and notes, how to write money, and what to leave out.

**Columns in the sheet**

Each row of the sheet is one draft: a practice setting (`kind` = `setting`) or a short label (`kind` = `label`). The label row of a skill comes first, then its two or three settings.

| Column | What it holds |
|---|---|
| kind | `setting` or `label` |
| indicator | The skill's code, for example `B4.1.1.1.2` |
| class | Basic 1 to Basic 6 |
| short_label | Kwan's short name for the skill |
| exemplar_summary | A line on what the curriculum asks the pupil to do. It is filled only when the maintainer has the curriculum text; it may be empty. |
| draft_text | The draft setting (or, on a `label` row, the draft label) |
| decision | **You fill this in:** `accept`, `edit` or `reject` |
| edited_text | Your corrected version, when the decision is `edit` |
| reviewer_id | Your reviewer id |
| region_taught | The region where you teach |
| note | Optional |
| row_id | A code that tells Kwan which draft the row is. Do not change it. |

A first sheet holds about 100 settings and their labels, mixed across the six classes. The maintainer makes it with `kwan context review-export` and reads your answers back with `kwan context review-import your-sheet.csv`. An import that finds a problem (an edited text that fails a check, an unknown decision, a reviewer id that looks like a name or an email address) names the row and the reason, and applies every other row.

**What to check, in this order**

1. **The maths.** Does the setting practise exactly this skill? Are the numbers right for the class? Is the answer correct?
2. **Ghana.** Is every place real and in the right region? Does every coin and note exist? Are the prices and distances believable in your area this year?
3. **The child.** Can a pupil in that class read it alone? Is it 220 characters or fewer? Is there one clear question or task?
4. **Fairness and safety.** No stereotypes, no brand names, nothing unsuitable for a young child.
5. **Own words.** It must not be a NaCCA exemplar with a few words changed.

Your region matters here. If a price is right in Accra but wrong in Lawra, say so in the note. If a food, a game or a word is not used where you teach, say so too.

## Short labels

A short label is Kwan's own name for a skill, 60 characters or fewer, for example "Read and write numbers to 100,000". Kwan shows it wherever the full curriculum text is not available.

The label is reviewed on the same sheet as the practice settings. For each label choose `accept`, `edit` or `reject`.

A good label:

- says what the pupil will be able to **do** ("Add fractions with the same denominator");
- uses the words a teacher uses in class;
- is different from the labels of nearby skills, so two skills cannot be confused;
- is written fresh, not cut from the curriculum sentence;
- has 60 characters or fewer.

## Translations

For Basic 1 to Basic 3, Kwan offers practice settings in Twi (`tw`), Ewe (`ee`) and Ga (`gaa`). The first draft of each translation is made by machine. It stays hidden, marked as not reviewed, until a teacher who reads and writes the language well approves it.

The translation sheet has the same columns, plus `lang` (`tw`, `ee` or `gaa`) and `english_text`, the English setting the draft was made from. Only translations of English settings that a teacher has already accepted are sent out, so the English will not change under you.

For each translation choose `accept`, `edit` or `reject`, and check that:

- it **means the same** as the English setting, with the same numbers and the same question;
- it sounds like a teacher **speaking to a young child**, and is not word-for-word English;
- the spelling follows the way the language is written in school;
- maths words are the ones used in class in that language. Where teachers keep the English word, keep it;
- money, numbers and place names are unchanged.

Review only a language you read and write well. If a setting cannot be said naturally in the language, reject it and say why.

## What accept, edit and reject mean

| Decision | When to choose it | What happens next |
|---|---|---|
| **Accept** | It is right as it stands. You would use it in your class. | It is marked as reviewed and can ship in the next release. |
| **Edit** | The idea is good but something needs fixing: a price, a name, a word, a number. Write the corrected text in the `edited_text` column. | Your version replaces the draft and is marked as reviewed. Kwan runs its automatic checks on your version (length, reading level, money format, banned words, a Ghanaian term; for a translation, the same numbers as the English). If a check fails, the row is not applied and a maintainer comes back to you. |
| **Reject** | It is wrong, unsafe, copied, or does not fit the skill, and a small fix will not save it. A short note helps the next writer. | The draft does not ship. A new one may be written for a later sheet. |
| *(blank)* | You are not sure, or you ran out of time. | The row stays pending. Nothing ships from it. |

For skill links the sheet uses the words `accepted` and `rejected`, and there is no edit.

Do not worry about being "too strict". A rejected row costs a little time. A wrong setting in front of thousands of pupils costs much more.

When two reviewers disagree on the same row, the item stays pending until a maintainer has talked with both. The record of who decided what is kept.

## How credit is given

- Every reviewer and translator is listed in [`CONTRIBUTORS.md`](https://github.com/rofyray/kwan/blob/main/CONTRIBUTORS.md), under "Teacher reviewers" or "Translators", with the name you choose. You may add the region where you teach and the languages you reviewed.
- Your name is added in the same pull request that imports your sheet.
- If you prefer not to be named, say so. Your work still counts and your reviewer id still marks the rows.
- Kwan's data files hold only your reviewer id and the date, never your name or contact details.
- Settings, labels and translations that you write or edit are shared under the CC BY 4.0 licence, like the rest of Kwan's written content. Anyone may reuse them if they give credit to the Kwan contributors.

Reviewing helps Kwan get things right. It does not make Kwan an official or approved teaching material.

## Questions

Ask on your "Teacher review" issue. No question is too small.

---

Kwan uses the NaCCA Standards-Based Curriculum to organise skills. Kwan is not reviewed, approved or endorsed by NaCCA or the Ministry of Education.
