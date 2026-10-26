#!/usr/bin/env bash
# Chặn push/commit ở mức git (cục bộ, không commit gì). Chạy 1 lần trong thư mục repo.
set -e
git rev-parse --git-dir >/dev/null
git remote | while read r; do git remote set-url --push "$r" no_push; done
mkdir -p .git/hooks
for h in pre-commit pre-push; do printf '#!/bin/sh\necho "Chan: khong commit/push tu dong. Con nguoi tu lam." >&2\nexit 1\n' > .git/hooks/$h; chmod +x .git/hooks/$h; done
echo "Da chan push (pushurl=no_push) va pre-commit/pre-push. Muon tu commit: bo hook tam thoi."
