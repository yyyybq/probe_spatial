#!/usr/bin/env bash
# Record a frozen runtime pack as a parentless commit + annotated tag code/<pack>.
# Usage: record_pack_commits_v2.sh <pack> <mode> [expected]
#   manifest     tree = exactly the files listed in runtime/<pack>/source.sha256; every hash verified
#   treedigest   tree = source/ minus __pycache__; the project tree digest must equal <expected>
#   snapshot     tree = source/ minus __pycache__; no complete identity was recorded (UNVERIFIED);
#                <expected> may name a JSON file of {absolute_path: sha256} partial hashes to check
# A private index file is used: HEAD, the real index and the working tree are untouched.
set -euo pipefail
cd /mnt/umm/users/yinbaiqiao/probe_spatial
pack=$1; mode=$2; expected=${3:-}
G="git -c user.name=yyyybq -c user.email=yinbaiqiao9@gmail.com"
src=runtime/$pack/source; tag=code/$pack
OUT=artifacts/release_audit_20261017/code_alignment/pack_commits.tsv
[[ -d $src ]] || { echo "no source for $pack"; exit 2; }
git rev-parse -q --verify "refs/tags/$tag" >/dev/null && { echo "tag $tag exists; delete it first"; exit 3; }
list=$(mktemp /tmp/psv/list.XXXXXX)
case $mode in
  manifest) sed 's/^[0-9a-f]\{64\}  //; s|^\./||' runtime/$pack/source.sha256 > "$list" ;;
  treedigest|snapshot) (cd "$src" && find . -type f ! -path "*/__pycache__/*" ! -name "*.pyc" -printf "%P\n" | sort) > "$list" ;;
  *) echo "bad mode"; exit 2 ;;
esac
idx=$(mktemp /tmp/psv/idx.XXXXXX); rm -f "$idx"
GIT_INDEX_FILE=$idx git read-tree --empty
tr '\n' '\0' < "$list" | GIT_INDEX_FILE=$idx git --work-tree="$src" update-index --add -z --stdin
tree=$(GIT_INDEX_FILE=$idx git write-tree); nfiles=$(GIT_INDEX_FILE=$idx git ls-files | wc -l); rm -f "$idx"
verdict=""
case $mode in
  manifest)
    bad=0; n=0
    while read -r sha path; do path=${path#./}; n=$((n+1))
      got=$(git cat-file -p "$tree:$path" 2>/dev/null | sha256sum | cut -c1-64) || got=missing
      [[ "$got" == "$sha" ]] || bad=$((bad+1)); done < runtime/$pack/source.sha256
    [[ $bad -eq 0 && $n -eq $nfiles ]] && verdict="PASS_MANIFEST($n files)" || verdict="FAIL_MANIFEST($bad bad of $n; tree $nfiles)"
    ident="source.sha256 sha256 $(sha256sum runtime/$pack/source.sha256 | cut -c1-64)" ;;
  treedigest)
    got=$(SOURCE="$src" python3 -c '
import os
from hashlib import sha256
from pathlib import Path
root=Path(os.environ["SOURCE"]); d=sha256()
for p in sorted(p for p in root.rglob("*") if p.is_file() and "__pycache__" not in p.parts):
    rel=p.relative_to(root).as_posix().encode(); d.update(len(rel).to_bytes(8,"big")); d.update(rel); d.update(sha256(p.read_bytes()).digest())
print(d.hexdigest())')
    [[ "$got" == "$expected" ]] && verdict="PASS_TREE_DIGEST" || verdict="FAIL_TREE_DIGEST(got $got)"
    ident="project tree digest $expected" ;;
  snapshot)
    verdict="UNVERIFIED_SNAPSHOT_20261017"; ident="no complete identity recorded at use time"
    if [[ -n $expected ]]; then
      chk=$(PACK_SRC="$src" python3 - "$expected" <<'PY'
import json, os, sys, hashlib
src = os.path.abspath(os.environ["PACK_SRC"]) + "/"
d = json.load(open(sys.argv[1]))["source_hashes"]
ok = bad = 0
for p, s in d.items():
    if p.startswith(src):
        got = hashlib.sha256(open(p, "rb").read()).hexdigest()
        ok += got == s; bad += got != s
print(f"{ok} {bad}")
PY
)
      verdict="$verdict;partial_hashes_ok=${chk% *};bad=${chk#* }"; ident="$ident; partial file hashes from $expected"
    fi ;;
esac
case $verdict in FAIL*|*bad=[1-9]*) echo "$pack: $verdict -- not tagged"; rm -f "$list"; exit 1 ;; esac
msg=$(printf "Frozen runtime pack %s\n\nTree equals runtime/%s/source as used on the cluster (mode %s).\nIdentity: %s.\nVerification: %s.\nRecorded 2026-10-17 so results produced with this pack map to a commit.\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" "$pack" "$pack" "$mode" "$ident" "$verdict")
commit=$(printf "%s\n" "$msg" | $G commit-tree "$tree")
$G tag -a "$tag" "$commit" -m "Frozen runtime pack $pack ($verdict)"
printf "%s\t%s\t%s\t%s\t%s\t%s\n" "$pack" "$commit" "$tag" "$nfiles" "$mode" "$verdict" >> "$OUT"
rm -f "$list"; echo "$pack -> ${commit:0:12} files=$nfiles $verdict"
