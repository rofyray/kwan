#!/usr/bin/env bash
# Fetch the raw data Kwan is built and validated on, into raw/ at the repo root.
# Safe to re-run: files that are already present and complete are skipped.
#
#   bash scripts/download_data.sh                  everything below
#   bash scripts/download_data.sh ammore junyi     only the named groups
#   bash scripts/download_data.sh learner          every learner dataset, without the curriculum PDFs
#
# Groups (the learner datasets are the ones listed in data/datasets.yaml):
#   curriculum        raw/m13.pdf, raw/m46.pdf and their text    NaCCA Mathematics, Basic 1-3 and Basic 4-6
#   assistments-2009  raw/skill_builder_data_corrected.csv       ASSISTments 2009 skill builder (corrected)
#   eedi              raw/eedi/data/...                           Eedi NeurIPS 2020, task 3/4 and task 1/2
#   ammore            raw/ammore/                                 AMMORE (Rori, Rising Academies), CC BY-NC 4.0
#   assistments-2012  raw/assist2012/                             ASSISTments 2012-2013
#   assistments-2015  raw/assist2015/                             ASSISTments 2015 (100 skill builders)
#   assistments-2017  raw/assist2017/                             ASSISTments 2017 (competition data)
#   xes3g5m           raw/xes3g5m/XES3G5M/                        XES3G5M, on Google Drive (needs gdown)
#   junyi             raw/junyi/                                  Junyi Academy 2015, a 1.3 GB .rar, non-commercial
#
# XES3G5M and Junyi are fragile downloads (Google Drive, a .rar archive). If one of them cannot be
# fetched or unpacked, it is skipped with a clear message and the script still succeeds: the
# evaluation writes a "skipped" block for it. Any other failure stops the script.
set -euo pipefail

M13_URL="https://curriculumresources.edu.gh/basic/wp-content/uploads/2026/05/MATHEMATICS-BASIC-B1-B3-2.pdf"
M46_URL="https://curriculumresources.edu.gh/basic/wp-content/uploads/2026/05/MATHEMATICS-BASIC-B4-B6-1.pdf"
A09_URL="http://base.ustc.edu.cn/data/ASSISTment/2009_skill_builder_data_corrected.zip"
EEDI_URL="https://dqanonymousdata.blob.core.windows.net/neurips-public/data.zip"
AMMORE_URL="https://github.com/owenhenkel/ammore_dataset"
A12_URL="http://base.ustc.edu.cn/data/ASSISTment/2012-2013-data-with-predictions-4-final.zip"
A15_URL="http://base.ustc.edu.cn/data/ASSISTment/2015_100_skill_builders_main_problems.zip"
A17_URL="http://base.ustc.edu.cn/data/ASSISTment/anonymized_full_release_competition_dataset.zip"
XES_URL="https://github.com/ai4ed/XES3G5M"                       # the README links the Google Drive file below
XES_DRIVE_ID="1eFiIYyh5O2V90RA0brammGH6EpHvPDQe"
JUNYI_URL="http://base.ustc.edu.cn/data/JunyiAcademy_Math_Practicing_Log/junyi.rar"

A09_FILES="skill_builder_data_corrected.csv"
EEDI_FILES="data/metadata/answer_metadata_task_3_4.csv
data/metadata/question_metadata_task_3_4.csv
data/metadata/student_metadata_task_3_4.csv
data/metadata/answer_metadata_task_1_2.csv
data/metadata/question_metadata_task_1_2.csv
data/metadata/subject_metadata.csv
data/train_data/train_task_3_4.csv
data/train_data/train_task_1_2.csv"
AMMORE_FILES="AMMORE_student_answers.csv AMMORE_demographic_data.csv"
A12_FILES="2012-2013-data-with-predictions-4-final.csv"
A15_FILES="2015_100_skill_builders_main_problems.csv"
A17_FILES="anonymized_full_release_competition_dataset.csv"
XES_FILES="XES3G5M/question_level/train_valid_sequences_quelevel.csv XES3G5M/question_level/test_quelevel.csv"
JUNYI_FILES="junyi_ProblemLog_original.csv junyi_Exercise_table.csv relationship_annotation_training.csv relationship_annotation_testing.csv"

ALL_GROUPS="curriculum assistments-2009 eedi ammore assistments-2012 assistments-2015 assistments-2017 xes3g5m junyi"
LEARNER_GROUPS="assistments-2009 eedi ammore assistments-2012 assistments-2015 assistments-2017 xes3g5m junyi"
SKIPPED=""

say()  { echo "[download_data] $*"; }
fail() { echo "[download_data] ERROR: $*" >&2; exit 1; }

need() {  # need <command> <how to install it>
  command -v "$1" >/dev/null 2>&1 || fail "'$1' is not installed. $2"
}
skip() { SKIPPED="$SKIPPED $1"; echo "[download_data] SKIPPED $1: $2" >&2; }

# which groups to fetch: the arguments, or everything
GROUPS_WANTED=""
for arg in "$@"; do
  case "$arg" in
    all)     GROUPS_WANTED="$ALL_GROUPS" ;;
    learner) GROUPS_WANTED="$GROUPS_WANTED $LEARNER_GROUPS" ;;
    curriculum|assistments-2009|eedi|ammore|assistments-2012|assistments-2015|assistments-2017|xes3g5m|junyi)
             GROUPS_WANTED="$GROUPS_WANTED $arg" ;;
    -h|--help) sed -n '2,24p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *)       fail "unknown group '$arg'. Use one of: all learner $ALL_GROUPS" ;;
  esac
done
[ -n "$GROUPS_WANTED" ] || GROUPS_WANTED="$ALL_GROUPS"
want() { case " $GROUPS_WANTED " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }

need curl      "Install curl (macOS: brew install curl; Debian/Ubuntu: sudo apt-get install curl)."
need unzip     "Install unzip (macOS: brew install unzip; Debian/Ubuntu: sudo apt-get install unzip)."
if want curriculum; then
  need pdftotext "It ships with poppler (macOS: brew install poppler; Debian/Ubuntu: sudo apt-get install poppler-utils)."
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
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

# all_present <dir> <file list>: every listed file exists under <dir> and is not empty
all_present() {
  local dir="$1" f
  for f in $2; do [ -s "$dir/$f" ] || return 1; done
}

# get_git <url> <dest dir> <label> <file list>: shallow clone, once
get_git() {
  local url="$1" dest="$2" label="$3" files="$4"
  if all_present "$dest" "$files"; then
    say "$label already present and complete, skipping"
    return 0
  fi
  need git "Install git (macOS: xcode-select --install; Debian/Ubuntu: sudo apt-get install git)."
  say "cloning $label"
  rm -rf "$dest.part"
  git clone --quiet --depth 1 "$url" "$dest.part" || { rm -rf "$dest.part"; fail "could not clone $label from $url"; }
  all_present "$dest.part" "$files" || { rm -rf "$dest.part"; fail "$url was cloned but expected files are missing"; }
  rm -rf "$dest" && mv "$dest.part" "$dest"
}

# run_gdown <drive id> <out file>: gdown as a command or as a module of whichever Python has it
run_gdown() {
  local id="$1" out="$2" py
  if command -v gdown >/dev/null 2>&1; then gdown --quiet "$id" -O "$out"; return $?; fi
  for py in "${PYTHON:-}" "$REPO_ROOT/.venv/bin/python" python3 python; do
    [ -n "$py" ] && command -v "$py" >/dev/null 2>&1 || continue
    "$py" -c "import gdown" >/dev/null 2>&1 || continue
    "$py" -m gdown --quiet "$id" -O "$out"; return $?
  done
  return 127
}

# get_xes3g5m: Google Drive download with gdown, then untar. Fragile: never fatal.
get_xes3g5m() {
  local dest="xes3g5m" tgz="xes3g5m/XES3G5M.tar.gz" rc=0
  if all_present "$dest" "$XES_FILES"; then
    say "XES3G5M already present and complete, skipping"
    return 0
  fi
  mkdir -p "$dest"
  if ! { [ -s "$tgz" ] && tar -tzf "$tgz" >/dev/null 2>&1; }; then
    rm -f "$tgz" "$tgz.part"
    say "downloading XES3G5M from Google Drive (file $XES_DRIVE_ID, linked from $XES_URL)"
    run_gdown "$XES_DRIVE_ID" "$tgz.part" || rc=$?
    if [ "$rc" = 127 ]; then
      skip xes3g5m "gdown is not installed. Run: pip install 'kwan[eval]' (or pip install gdown), then run this script again."
      return 0
    fi
    if [ "$rc" != 0 ] || [ ! -s "$tgz.part" ]; then
      rm -f "$tgz.part"
      skip xes3g5m "Google Drive refused the download (often a quota limit). Try again later, or download the file by hand from the link in $XES_URL and save it as raw/$tgz."
      return 0
    fi
    mv -f "$tgz.part" "$tgz"
  fi
  say "unpacking $tgz"
  tar -xzf "$tgz" -C "$dest" || { skip xes3g5m "raw/$tgz could not be unpacked. Delete it and run this script again."; return 0; }
  all_present "$dest" "$XES_FILES" || skip xes3g5m "raw/$tgz was unpacked but the question_level files are missing."
}

# unrar_into <rar> <dest dir> <file list>: unpack only the listed files, trying each tool that reads a .rar
unrar_into() {
  local rar="$1" dest="$2" files="$3" py
  # shellcheck disable=SC2086  # the file list is split into words on purpose
  if command -v bsdtar >/dev/null 2>&1 && bsdtar -xf "$rar" -C "$dest" $files 2>/dev/null; then return 0; fi
  # shellcheck disable=SC2086
  if command -v unar   >/dev/null 2>&1 && unar -quiet -force-overwrite -no-directory -output-directory "$dest" "$rar" $files >/dev/null 2>&1; then return 0; fi
  # shellcheck disable=SC2086
  if command -v unrar  >/dev/null 2>&1 && unrar x -o+ -inul "$rar" $files "$dest/"; then return 0; fi
  # shellcheck disable=SC2086
  if command -v 7z     >/dev/null 2>&1 && 7z x -y -o"$dest" "$rar" $files >/dev/null 2>&1; then return 0; fi
  for py in "${PYTHON:-}" "$REPO_ROOT/.venv/bin/python" python3 python; do
    [ -n "$py" ] && command -v "$py" >/dev/null 2>&1 || continue
    "$py" -c "import rarfile" >/dev/null 2>&1 || continue
    "$py" -c "import sys, rarfile; rarfile.RarFile(sys.argv[1]).extractall(sys.argv[2], members=sys.argv[3].split())" \
      "$rar" "$dest" "$files" 2>/dev/null && return 0
  done
  return 1
}

# get_junyi: 1.3 GB .rar. Only the files Kwan reads are unpacked (2.7 GB); the 9.7 GB PSLC copy of the
# problem log stays in the archive. Fragile: never fatal.
get_junyi() {
  local dest="junyi" rar="junyi.rar"
  if [ -e "$dest/.extracted" ] && all_present "$dest" "$JUNYI_FILES"; then
    say "Junyi already present and complete, skipping"
    return 0
  fi
  if [ ! -s "$rar" ]; then
    say "downloading Junyi (junyi.rar, 1.3 GB)"
    ( fetch "$JUNYI_URL" "$rar" "Junyi ($rar)" ) || { skip junyi "could not download $JUNYI_URL. Try again later."; return 0; }
  fi
  say "unpacking $rar (about 2.7 GB on disk)"
  mkdir -p "$dest"
  if unrar_into "$rar" "$dest" "$JUNYI_FILES README.txt" && all_present "$dest" "$JUNYI_FILES"; then
    date > "$dest/.extracted"
  else
    skip junyi "raw/$rar could not be unpacked. Install one of: bsdtar (libarchive), unar, unrar or 7z, then run this script again. If the archive is damaged, delete raw/$rar first."
  fi
}

# NaCCA curriculum (Ghana), Basic 1-3 and Basic 4-6 Mathematics
if want curriculum; then
  get_pdf "$M13_URL" m13
  get_pdf "$M46_URL" m46
fi

# ASSISTments 2009 skill builder (corrected), ~9 MB zip, 63 MB CSV.
# If base.ustc.edu.cn is down, the same file is distributed by the EduData package:
#   pip install EduData && edudata download assistment-2009-2010-skill   (then copy the CSV to raw/)
if want assistments-2009; then
  get_zip "$A09_URL" a09.zip . "ASSISTments 2009" "$A09_FILES" 'skill_builder_data_corrected.csv'
fi

# Eedi NeurIPS 2020, ~657 MB zip; task 3/4 (1.4M answers) and task 1/2 (15.9M answers, 1.5 GB unpacked)
if want eedi; then
  get_zip "$EEDI_URL" eedi.zip eedi "Eedi NeurIPS 2020 (task 3/4 and task 1/2)" "$EEDI_FILES" \
    'data/metadata/*_task_3_4.csv' 'data/metadata/*_task_1_2.csv' 'data/metadata/subject_metadata.csv' \
    'data/train_data/train_task_3_4.csv' 'data/train_data/train_task_1_2.csv'
fi

# AMMORE (CC BY-NC 4.0: evaluation only)
if want ammore; then
  get_git "$AMMORE_URL" ammore "AMMORE" "$AMMORE_FILES"
fi

# ASSISTments 2012-2013 (576 MB zip, 3 GB CSV), 2015 (3 MB zip) and 2017 (68 MB zip)
if want assistments-2012; then
  get_zip "$A12_URL" 2012-2013-data-with-predictions-4-final.zip assist2012 "ASSISTments 2012-2013" "$A12_FILES" "$A12_FILES"
fi
if want assistments-2015; then
  get_zip "$A15_URL" 2015_100_skill_builders_main_problems.zip assist2015 "ASSISTments 2015" "$A15_FILES" "$A15_FILES"
fi
if want assistments-2017; then
  get_zip "$A17_URL" anonymized_full_release_competition_dataset.zip assist2017 "ASSISTments 2017" "$A17_FILES" "$A17_FILES"
fi

# Fragile downloads: skipped with a message when they fail
if want xes3g5m; then get_xes3g5m; fi
if want junyi;   then get_junyi;   fi

if [ -n "$SKIPPED" ]; then
  say "done, but skipped:$SKIPPED (see the messages above). Everything else is in place."
else
  say "done"
fi
ls -la
