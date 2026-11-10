#!/usr/bin/env bash
# Fetch the raw data Kwan is built and validated on, into raw/ at the repo root.
# Safe to re-run: files that are already present and complete are skipped.
#   raw/m13.pdf, raw/m46.pdf, raw/m13.txt, raw/m46.txt    NaCCA Mathematics curriculum, Basic 1-3 and Basic 4-6
#   raw/a09.zip, raw/skill_builder_data_corrected.csv     ASSISTments 2009 skill builder (corrected)
#   raw/eedi.zip, raw/eedi/data/...                       Eedi NeurIPS 2020, task 3/4 subset
set -euo pipefail

M13_URL="https://curriculumresources.edu.gh/basic/wp-content/uploads/2026/05/MATHEMATICS-BASIC-B1-B3-2.pdf"
M46_URL="https://curriculumresources.edu.gh/basic/wp-content/uploads/2026/05/MATHEMATICS-BASIC-B4-B6-1.pdf"
A09_URL="http://base.ustc.edu.cn/data/ASSISTment/2009_skill_builder_data_corrected.zip"
EEDI_URL="https://dqanonymousdata.blob.core.windows.net/neurips-public/data.zip"

A09_FILES="skill_builder_data_corrected.csv"
EEDI_FILES="data/metadata/answer_metadata_task_3_4.csv
data/metadata/question_metadata_task_3_4.csv
data/metadata/student_metadata_task_3_4.csv
data/metadata/subject_metadata.csv
data/train_data/train_task_3_4.csv"

say()  { echo "[download_data] $*"; }
fail() { echo "[download_data] ERROR: $*" >&2; exit 1; }

need() {  # need <command> <how to install it>
  command -v "$1" >/dev/null 2>&1 || fail "'$1' is not installed. $2"
}
need curl      "Install curl (macOS: brew install curl; Debian/Ubuntu: sudo apt-get install curl)."
need unzip     "Install unzip (macOS: brew install unzip; Debian/Ubuntu: sudo apt-get install unzip)."
need pdftotext "It ships with poppler (macOS: brew install poppler; Debian/Ubuntu: sudo apt-get install poppler-utils)."

cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p raw && cd raw

size_of() { wc -c < "$1" | tr -d '[:space:]'; }

# A complete PDF starts with %PDF and carries the %%EOF trailer; a cut-off download has no trailer.
pdf_ok() { [ -s "$1" ] && [ "$(head -c 4 "$1")" = "%PDF" ] && tail -c 2048 "$1" | grep -aq '%%EOF'; }

# A complete zip has a readable central directory (it sits at the end of the file, so a cut-off download fails).
zip_ok() { [ -s "$1" ] && unzip -l "$1" >/dev/null 2>&1; }

# fetch <url> <file> <label>: download to <file>.part, then rename, so <file> only ever exists complete.
fetch() {
  local url="$1" out="$2" label="$3"
  local quiet="-sS"; [ -t 2 ] && quiet="-S"       # progress meter only on a terminal, never in a log file
  say "downloading $label"
  if [ -s "$out.part" ]; then
    say "resuming raw/$out.part"
    curl -fL $quiet --retry 3 -C - -o "$out.part" "$url" \
      || { rm -f "$out.part"; curl -fL $quiet --retry 3 -o "$out.part" "$url"; } \
      || { rm -f "$out.part"; fail "could not download $label from $url"; }
  else
    curl -fL $quiet --retry 3 -o "$out.part" "$url" \
      || fail "could not download $label from $url (a partial raw/$out.part, if any, is resumed on the next run)"
  fi
  [ -s "$out.part" ] || { rm -f "$out.part"; fail "$url returned an empty file for $label"; }
  mv -f "$out.part" "$out"
}

# get_pdf <url> <name>: raw/<name>.pdf and its text layer raw/<name>.txt
get_pdf() {
  local url="$1" name="$2" fresh=0
  if pdf_ok "$name.pdf"; then
    say "$name.pdf already present, skipping download"
  else
    fetch "$url" "$name.pdf" "$name.pdf"; fresh=1
    pdf_ok "$name.pdf" || fail "$name.pdf from $url is not a complete PDF. Delete raw/$name.pdf and run again."
  fi
  if [ "$fresh" = 0 ] && [ -s "$name.txt" ] && [ ! "$name.pdf" -nt "$name.txt" ]; then
    say "$name.txt already present, skipping pdftotext"
  else
    say "extracting text: $name.pdf -> $name.txt"
    pdftotext -layout "$name.pdf" "$name.txt.part" || { rm -f "$name.txt.part"; fail "pdftotext failed on raw/$name.pdf"; }
    [ -s "$name.txt.part" ] || { rm -f "$name.txt.part"; fail "pdftotext produced no text from raw/$name.pdf"; }
    mv -f "$name.txt.part" "$name.txt"
  fi
}

# extracted_ok <zip> <dest dir> <member list>: every member exists under <dest dir> with the size the zip records.
# Without the zip (for example deleted to save disk space) sizes cannot be checked, so non-empty files are accepted.
extracted_ok() {
  local zip="$1" dest="$2" members="$3" listing="" m want
  if zip_ok "$zip"; then listing="$(unzip -l "$zip")"; fi
  while IFS= read -r m; do
    [ -n "$m" ] || continue
    [ -s "$dest/$m" ] || return 1
    if [ -n "$listing" ]; then
      want="$(printf '%s\n' "$listing" | awk -v m="$m" '$4 == m { print $1; exit }')"
      [ -n "$want" ] && [ "$(size_of "$dest/$m")" = "$want" ] || return 1
    fi
  done <<EOF
$members
EOF
}

# get_zip <url> <zip> <dest dir> <label> <member list> <unzip patterns...>
get_zip() {
  local url="$1" zip="$2" dest="$3" label="$4" members="$5"; shift 5
  if [ -e "$zip" ] && ! zip_ok "$zip"; then
    say "raw/$zip is incomplete or damaged, removing it"
    rm -f "$zip"
  fi
  if extracted_ok "$zip" "$dest" "$members"; then
    say "$label already present and complete, skipping"
    return 0
  fi
  [ -f "$zip" ] || fetch "$url" "$zip" "$label ($zip)"
  zip_ok "$zip" || fail "$zip from $url is not a valid zip. Delete raw/$zip and run again."
  say "unpacking $zip"
  mkdir -p "$dest"
  unzip -o -q "$zip" "$@" -d "$dest" || fail "could not unpack raw/$zip"
  extracted_ok "$zip" "$dest" "$members" || fail "raw/$zip was unpacked but expected files are missing or cut short under raw/$dest"
}

# NaCCA curriculum (Ghana), Basic 1-3 and Basic 4-6 Mathematics
get_pdf "$M13_URL" m13
get_pdf "$M46_URL" m46

# ASSISTments 2009 skill builder (corrected), ~9 MB zip, 63 MB CSV.
# If base.ustc.edu.cn is down, the same file is distributed by the EduData package:
#   pip install EduData && edudata download assistment-2009-2010-skill   (then copy the CSV to raw/)
get_zip "$A09_URL" a09.zip . "ASSISTments 2009" "$A09_FILES" 'skill_builder_data_corrected.csv'

# Eedi NeurIPS 2020, ~657 MB zip; extract only the task 3/4 files we need
get_zip "$EEDI_URL" eedi.zip eedi "Eedi NeurIPS 2020 (task 3/4)" "$EEDI_FILES" \
  'data/metadata/*_task_3_4.csv' 'data/metadata/subject_metadata.csv' 'data/train_data/train_task_3_4.csv'

echo "done"; ls -la
