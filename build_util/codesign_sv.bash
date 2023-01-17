#!/usr/bin/env bash
# !!! コードサイニング証明書を取り扱うので取り扱い注意 !!!

set -eu

if [ $# -ne 1 ]; then
    echo "引数の数が一致しません"
    exit 1
fi
target_file_glob="$1"

# 指定ファイルに署名する
function codesign() {
    TARGET="$1"
    SIGNTOOL=$(find "C:/Program Files (x86)/Windows Kits/10/App Certification Kit" -name "signtool.exe" | sort -V | tail -n 1)
    powershell "& '$SIGNTOOL' sign /n 'Open Source Developer, Yuto Ashida' /fd sha1 /t http://time.certum.pl/ '$TARGET'"
    powershell "& '$SIGNTOOL' sign /n 'Open Source Developer, Yuto Ashida' /fd sha256 /td sha256 /tr http://time.certum.pl/ /as '$TARGET'"
}

# 指定ファイルが署名されているか
function is_signed() {
    TARGET="$1"
    SIGNTOOL=$(find "C:/Program Files (x86)/Windows Kits/10/App Certification Kit" -name "signtool.exe" | sort -V | tail -n 1)
    powershell "& '$SIGNTOOL' verify /pa '$TARGET'" || return 1
}

# 署名されていなければ署名
# shellcheck disable=SC2012,SC2086
ls $target_file_glob | while read -r target_file; do
    if is_signed "$target_file"; then
        echo "署名済み: $target_file"
    else
        echo "署名: $target_file"
        codesign "$target_file"
    fi
done
