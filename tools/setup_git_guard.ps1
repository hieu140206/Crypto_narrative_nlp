# Windows PowerShell. Chạy 1 lần trong thư mục repo.
git rev-parse --git-dir | Out-Null
git remote | ForEach-Object { git remote set-url --push $_ no_push }
$hooks = ".git\hooks"
New-Item -ItemType Directory -Force $hooks | Out-Null
foreach ($h in "pre-commit","pre-push") {
  "#!/bin/sh`necho 'Chan: khong commit/push tu dong. Con nguoi tu lam.' >&2`nexit 1`n" | Set-Content -NoNewline -Encoding ascii "$hooks\$h"
}
"Da chan push (pushurl=no_push) va pre-commit/pre-push. Muon tu commit: xoa hook tam thoi."
