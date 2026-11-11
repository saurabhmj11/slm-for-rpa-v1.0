@echo off
rem ====================================================================================
rem  OOA Assistant - one-click setup for a Windows laptop.
rem
rem  1. Put this file in the folder where the assistant should live, e.g.  C:\OOA
rem  2. Have these in that folder or in Downloads (loose or inside a Google Drive .zip):
rem       ooa_assistant.zip                 (from Saurabh)
rem       tokenizer.pkl, token_bytes.pt     (Drive: ooa-nanochat > nanochat_base > tokenizer)
rem       model_<N>.pt, meta_<N>.json       (Drive: ooa-nanochat > nanochat_base > chatsft_checkpoints > d8)
rem       catalog.jsonl  (optional, newer action list; Drive: ooa-nanochat > sft)
rem  3. Double-click this file. It installs Python if needed, the packages, unzips the assistant,
rem     puts the model files in place, makes a desktop shortcut and can start the assistant.
rem  Nothing is deleted: an older model in the folder is moved aside.
rem ====================================================================================
set "OOA_ROOT=%~dp0"
set "OOA_SELF=%~f0"
powershell -NoProfile -ExecutionPolicy Bypass -Command "$s = Get-Content -Raw -LiteralPath $env:OOA_SELF; $i = $s.IndexOf('#' + 'PS-START'); Invoke-Expression $s.Substring($i)"
pause
exit /b

#PS-START
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.IO.Compression.FileSystem
$root = $env:OOA_ROOT.TrimEnd('\')
if ((Split-Path $root -Leaf) -eq 'assistant' -and (Split-Path (Split-Path $root -Parent) -Leaf) -eq 'rpa model') { $root = Split-Path $root -Parent }
if ((Split-Path $root -Leaf) -eq 'rpa model') { $root = Split-Path $root -Parent }
$rpa = Join-Path $root 'rpa model'
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
function Say($t) { Write-Host $t }
function Good($t) { Write-Host $t -ForegroundColor Green }
function Bad($t) { Write-Host $t -ForegroundColor Red }
Say "OOA Assistant setup | install folder: $root"
Say ""

# ---------------------------------------------------------------- 1. Python
Say "[1/7] Python"
function Find-Python {
    $cands = @('py', 'python', (Join-Path $env:LOCALAPPDATA 'Programs\Python\Python312\python.exe'),
               (Join-Path $env:LOCALAPPDATA 'Programs\Python\Python313\python.exe'))
    foreach ($c in $cands) {
        try {
            $pa = @()
            if ($c -eq 'py') { $pa = @('-3') }
            $out = & $c @pa -c "import sys; print('%d.%d|%s' % (sys.version_info[0], sys.version_info[1], sys.executable))" 2>$null
            if ($LASTEXITCODE -eq 0 -and "$out" -match '^3\.(\d+)\|(.+)$' -and [int]$Matches[1] -ge 10) {
                $exe = $Matches[2].Trim()
                if (Test-Path -LiteralPath $exe) { return @{ Exe = $exe; Version = ("$out" -split '\|')[0] } }
            }
        } catch { }
    }
    return $null
}
$py = Find-Python
if (-not $py) {
    Say "  Python 3.10+ not found. Installing Python 3.12 with winget ..."
    try {
        winget install --id Python.Python.3.12 -e --scope user --accept-source-agreements --accept-package-agreements
    } catch { }
    $py = Find-Python
}
if (-not $py) {
    Bad "  Python could not be installed automatically."
    Say "  Install it from https://www.python.org/downloads/windows/ (tick 'Add python.exe to PATH'), then run this file again."
    return
}
$pyExe = $py.Exe                                   # full path of python.exe (also written into the launcher)
Good "  Python $($py.Version) ($pyExe)"

# ---------------------------------------------------------------- 2. Smart App Control
Say "[2/7] Windows Smart App Control"
$sac = $null
try { $sac = (Get-ItemProperty -Path 'HKLM:\SYSTEM\CurrentControlSet\Control\CI\Policy' -Name VerifiedAndReputablePolicyState -ErrorAction Stop).VerifiedAndReputablePolicyState } catch { }
if ($sac -eq 1) {
    Bad "  Smart App Control is ON. It blocks PyTorch (error WinError 4551) and the assistant cannot start."
    Say "  Turn it off: Windows Security > App & browser control > Smart App Control settings > Off, then run this file again."
    Say "  (On many Windows versions it cannot be turned back on without resetting Windows; Defender antivirus keeps working.)"
    return
} elseif ($sac -eq 2) {
    Say "  Smart App Control is in Evaluation mode; if PyTorch is blocked later, turn it Off (see above)."
} else {
    Good "  OK"
}

# ---------------------------------------------------------------- 3. find the files
Say "[3/7] Looking for ooa_assistant.zip and the model files (this folder, Downloads, Desktop, also inside .zip files)"
$searchDirs = @($root, (Join-Path $env:USERPROFILE 'Downloads'), (Join-Path $env:USERPROFILE 'Desktop'), [Environment]::GetFolderPath('Desktop')) |
    Where-Object { $_ -and (Test-Path -LiteralPath $_) } | Select-Object -Unique
$files = foreach ($d in $searchDirs) { Get-ChildItem -LiteralPath $d -Recurse -File -ErrorAction SilentlyContinue }
# skip setup leftovers, kept older models and the old catalog that ships inside the zip
$files = @($files | Where-Object { $_.FullName -notmatch '\\_ooa_setup_tmp\\|old_model_|_before_|\\data\\catalog\\' })
$cands = New-Object System.Collections.ArrayList
$want = '^(tokenizer.*\.pkl|token_bytes.*\.pt|model_\d+.*\.pt|meta_\d+.*\.json|catalog.*\.jsonl)$'
foreach ($f in $files) {
    if ($f.Name -match $want) { [void]$cands.Add([pscustomobject]@{ Name = $f.Name; Zip = $null; Entry = $null; Path = $f.FullName; Time = $f.LastWriteTime }) }
}
foreach ($zf in ($files | Where-Object { $_.Extension -eq '.zip' -and $_.Name -notlike 'ooa_assistant*' })) {
    try {
        $z = [System.IO.Compression.ZipFile]::OpenRead($zf.FullName)
        foreach ($e in $z.Entries) {
            if ($e.Name -match $want -and $e.FullName -notmatch 'old_model_|_before_|/data/catalog/') {
                [void]$cands.Add([pscustomobject]@{ Name = $e.Name; Zip = $zf.FullName; Entry = $e.FullName
                                                    Path = "$($zf.FullName) -> $($e.FullName)"; Time = $e.LastWriteTime.DateTime })
            }
        }
        $z.Dispose()
    } catch { }
}
function Read-Text($c) {
    if ($c.Zip) {
        $z = [System.IO.Compression.ZipFile]::OpenRead($c.Zip)
        try { $r = New-Object System.IO.StreamReader(($z.Entries | Where-Object { $_.FullName -eq $c.Entry } | Select-Object -First 1).Open()); $t = $r.ReadToEnd(); $r.Close(); return $t }
        finally { $z.Dispose() }
    }
    return [System.IO.File]::ReadAllText($c.Path)
}
function Step-Of($name) { if ($name -match '_(\d+)') { return [int]$Matches[1] } return -1 }
function Src-Of($c) { if ($c.Zip) { $c.Zip + '|' + [IO.Path]::GetDirectoryName($c.Entry) } else { [IO.Path]::GetDirectoryName($c.Path) } }

$zipFile = $files | Where-Object { $_.Name -like 'ooa_assistant*.zip' } | Sort-Object LastWriteTime -Descending | Select-Object -First 1
$tokPkl = $cands | Where-Object { $_.Name -like 'tokenizer*.pkl' } | Sort-Object Time -Descending | Select-Object -First 1
$tokBytes = $cands | Where-Object { $_.Name -like 'token_bytes*.pt' } | Sort-Object Time -Descending | Select-Object -First 1
$catalog = $cands | Where-Object { $_.Name -like 'catalog*.jsonl' } | Sort-Object Time -Descending | Select-Object -First 1
# the chat model: a model_<N>.pt with a meta_<N>.json that comes from chat training (not the pretrained base model)
$model = $null; $meta = $null
$metas = $cands | Where-Object { $_.Name -like 'meta_*.json' } | Sort-Object Time -Descending
foreach ($m in $metas) {
    try { $j = (Read-Text $m) | ConvertFrom-Json } catch { continue }
    if (-not ($j.user_config.PSObject.Properties.Name -contains 'mmlu_epochs')) { continue }   # base model meta: skip
    if ($j.user_config.num_iterations -lt 1000) { continue }                                     # autoresearch mini run: skip
    # same step AND preferably the same folder/zip as the meta (every full run ends at the same step number)
    $mm = $cands | Where-Object { $_.Name -like 'model_*.pt' -and (Step-Of $_.Name) -eq $j.step } |
        Sort-Object @{ Expression = { (Src-Of $_) -eq (Src-Of $m) }; Descending = $true }, @{ Expression = { $_.Time }; Descending = $true } |
        Select-Object -First 1
    if ($mm) { $model = $mm; $meta = $m; break }
}
$missing = @()
if (-not $zipFile) { $missing += 'ooa_assistant.zip' }
if (-not $tokPkl) { $missing += 'tokenizer.pkl' }
if (-not $tokBytes) { $missing += 'token_bytes.pt' }
if (-not $model) { $missing += 'model_<N>.pt + meta_<N>.json (from chatsft_checkpoints > d8)' }
foreach ($x in @(@('ooa_assistant.zip', $zipFile.FullName), @('tokenizer.pkl', $tokPkl.Path), @('token_bytes.pt', $tokBytes.Path),
                 @('model', $model.Path), @('meta', $meta.Path), @('catalog (optional)', $catalog.Path))) {
    if ($x[1]) { Say ("  found {0,-18} {1}" -f $x[0], $x[1]) }
}
if ($missing.Count) {
    foreach ($m in $missing) { Bad "  MISSING $m" }
    Say "  Nothing was changed. Download the missing files (single files from Drive, not whole folders), then run this again."
    return
}
$z = [System.IO.Compression.ZipFile]::OpenRead($zipFile.FullName)
$okZip = $z.Entries | Where-Object { $_.FullName -eq 'rpa model/assistant/app.py' }
$z.Dispose()
if (-not $okZip) { Bad "  $($zipFile.Name) is not the assistant package (no rpa model/assistant/app.py). Nothing was changed."; return }

# ---------------------------------------------------------------- 4. unzip + place files
Say "[4/7] Unzipping the assistant and placing the model files"
function Put($c, $dest) {
    if (-not $c) { return }
    if ($c.Zip) {
        $z = [System.IO.Compression.ZipFile]::OpenRead($c.Zip)
        try { [System.IO.Compression.ZipFileExtensions]::ExtractToFile(($z.Entries | Where-Object { $_.FullName -eq $c.Entry } | Select-Object -First 1), $dest, $true) }
        finally { $z.Dispose() }
    } else { Copy-Item -LiteralPath $c.Path -Destination $dest -Force }
}
# copy everything to a staging folder first: the files may already sit in the folders that are replaced below
$stage = Join-Path $root '_ooa_setup_tmp'
if (Test-Path -LiteralPath $stage) { Remove-Item -LiteralPath $stage -Recurse -Force }
New-Item -ItemType Directory -Path $stage | Out-Null
$step = '{0:D6}' -f (Step-Of $model.Name)
Say "  copying the model (about 300 MB) ..."
Put $tokPkl (Join-Path $stage 'tokenizer.pkl')
Put $tokBytes (Join-Path $stage 'token_bytes.pt')
Put $model (Join-Path $stage "model_$step.pt")
Put $meta (Join-Path $stage "meta_$step.json")
if ($catalog) { Put $catalog (Join-Path $stage 'catalog.jsonl') }
Expand-Archive -LiteralPath $zipFile.FullName -DestinationPath $root -Force
$tokDir = Join-Path $rpa 'nanochat_base\tokenizer'
$ckDir = Join-Path $rpa 'nanochat_base\chatsft_checkpoints\d8'
foreach ($d in @($tokDir, $ckDir, (Join-Path $rpa 'sft'))) { New-Item -ItemType Directory -Force -Path $d | Out-Null }
$old = @(Get-ChildItem -LiteralPath $ckDir -File -ErrorAction SilentlyContinue | Where-Object { $_.Name -match '^(model|meta|optim)_' })
$oldTok = @(Get-ChildItem -LiteralPath $tokDir -File -ErrorAction SilentlyContinue)
if ($old.Count -or $oldTok.Count) {             # the assistant loads the highest step in d8, so an old model must not stay there
    $keep = Join-Path $rpa "nanochat_base\old_model_$stamp"
    New-Item -ItemType Directory -Force -Path $keep | Out-Null
    $old | ForEach-Object { Move-Item -LiteralPath $_.FullName -Destination $keep }
    $oldTok | ForEach-Object { Copy-Item -LiteralPath $_.FullName -Destination $keep -Force }   # the old model needs its tokenizer
    Say "  previous model moved to $keep"
}
Move-Item -LiteralPath (Join-Path $stage 'tokenizer.pkl') -Destination $tokDir -Force
Move-Item -LiteralPath (Join-Path $stage 'token_bytes.pt') -Destination $tokDir -Force
Move-Item -LiteralPath (Join-Path $stage "model_$step.pt") -Destination $ckDir -Force
Move-Item -LiteralPath (Join-Path $stage "meta_$step.json") -Destination $ckDir -Force
if ($catalog) { Move-Item -LiteralPath (Join-Path $stage 'catalog.jsonl') -Destination (Join-Path $rpa 'sft') -Force }
Remove-Item -LiteralPath $stage -Recurse -Force
Good "  model step $step in place"

# ---------------------------------------------------------------- 5. packages
Say "[5/7] Installing Python packages (first time: several minutes, about 1 GB download)"
& $pyExe -m pip install --upgrade pip --quiet
& $pyExe -m pip install torch --index-url https://download.pytorch.org/whl/cpu --quiet
if ($LASTEXITCODE -ne 0) { Bad "  Installing PyTorch failed (see the messages above). Check the internet connection and run this again."; return }
& $pyExe -m pip install rustbpe tiktoken filelock jsonschema --quiet
if ($LASTEXITCODE -ne 0) { Bad "  Installing packages failed (see the messages above)."; return }
$prevEap = $ErrorActionPreference; $ErrorActionPreference = 'Continue'   # warnings on stderr must not stop the setup
$t = (& $pyExe -c "import torch; print('torch', torch.__version__)" 2>&1 | ForEach-Object { "$_" }) -join ' '
$torchExit = $LASTEXITCODE; $ErrorActionPreference = $prevEap
if ($torchExit -ne 0) {
    if ("$t" -match '4551') { Bad "  Windows blocked PyTorch (WinError 4551): turn Smart App Control Off (step 2 above) and run this again." }
    else { Bad "  PyTorch does not load: $t" }
    return
}
Good ("  " + ($t -replace '.*(torch \S+)$', '$1'))

# ---------------------------------------------------------------- 6. launcher + shortcut
Say "[6/7] Desktop shortcut"
$bat = Join-Path $rpa 'assistant\start_assistant.bat'
$txt = [System.IO.File]::ReadAllText($bat)      # use exactly the Python the packages were installed into
$txt = [regex]::Replace($txt, '(?m)^set PYTHON=.*$', ('set PYTHON=' + $pyExe.Replace('$', '$$')))
[System.IO.File]::WriteAllText($bat, $txt)
try {
    $ws = New-Object -ComObject WScript.Shell
    $lnk = $ws.CreateShortcut((Join-Path ([Environment]::GetFolderPath('Desktop')) 'OOA Assistant.lnk'))
    $lnk.TargetPath = $bat
    $lnk.WorkingDirectory = Split-Path $bat
    $lnk.Save()
    Good "  'OOA Assistant' shortcut created on the desktop"
} catch { Say "  (could not create the shortcut; start it with $bat)" }

# ---------------------------------------------------------------- 7. done
Say "[7/7] Done"
Good "Setup finished. Folder: $rpa"
Say "Start it any time with the 'OOA Assistant' desktop shortcut. The first start takes up to a minute;"
Say "when the black window says 'OOA Assistant is running', open http://127.0.0.1:8765"
$ans = Read-Host "Start the assistant now? (y/n)"
if ($ans -match '^[yY]') { Start-Process -FilePath $bat -WorkingDirectory (Split-Path $bat) }
