# Re-run every simulation study from scratch, then the theory check, figures, paper numbers and the paper.
# LLM (study 6), the ERC-8004 crawl (studies 8, 13, 14) and live twin sync are separate: see README.md.
param([int]$Workers = 5)
$ErrorActionPreference = "Stop"
$Root = Split-Path $PSScriptRoot -Parent
Set-Location $Root
$py = Join-Path $Root ".venv\Scripts\python.exe"
$code = Join-Path $Root "code\sybagent"

& $py -m pytest -q (Join-Path $code "tests")
$studies = [ordered]@{
    "sweep_attack" = "1_attack"; "sweep_defenses" = "2_defenses"; "sweep_anchors" = "3_anchor_bound";
    "sweep_adaptive" = "4_adaptive_attacks"; "sweep_audit" = "5_second_opinions"
}
foreach ($cfg in $studies.Keys) {
    $out = "results\$($studies[$cfg])"
    if (Test-Path $out) { Remove-Item -Recurse -Force $out }
    $t = Get-Date
    & $py (Join-Path $code "scripts\run_experiment.py") --sweep (Join-Path $code "configs\$cfg.yaml") --out $out --workers $Workers
    "{0} done in {1:n0}s" -f $studies[$cfg], ((Get-Date) - $t).TotalSeconds
}
$rb = "results\5b_robust_second_opinions"
if (Test-Path $rb) { Remove-Item -Recurse -Force $rb }
& $py (Join-Path $code "scripts\run_experiment.py") --sweep (Join-Path $code "configs\sweep_robust_audit_base.yaml") --out "$rb\baselines" --workers $Workers
& $py (Join-Path $code "scripts\run_experiment.py") --sweep (Join-Path $code "configs\sweep_robust_audit.yaml") --out "$rb\main" --workers $Workers
& $py analysis\validate_theory.py
# extensions: on-chain sampler (10), strategic attacker + games (11), Pareto (12)
& $py (Join-Path $code "scripts\onchain_sampler.py") --out results\10_onchain_sampler
& $py (Join-Path $code "scripts\run_strategic.py") --out results\11_strategic_attacker --workers $Workers
& $py analysis\games.py
& $py (Join-Path $code "scripts\run_experiment.py") --sweep (Join-Path $code "configs\sweep_pareto.yaml") --out results\12_pareto --workers $Workers
& $py analysis\pareto.py
# robustness (15), dynamic-programming best response, shared defenses (20, 20b), on-chain blocklist (21)
foreach ($pair in @("seeds20:seeds20", "sensitivity:sensitivity", "sensitivity_selection:selection")) {
    $c, $o = $pair.Split(":")
    & $py (Join-Path $code "scripts\run_experiment.py") --sweep (Join-Path $code "configs\sweep_$c.yaml") --out "results\15_robustness\$o" --workers $Workers
}
& $py analysis\dp_attacker.py
& $py (Join-Path $code "scripts\run_shared.py") --out results\20_shared_defense --workers $Workers
& $py (Join-Path $code "scripts\run_pooling.py") --out results\20_shared_defense\pooling --workers $Workers
& $py (Join-Path $code "scripts\onchain_blocklist.py") --out results\21_onchain_blocklist
& $py (Join-Path $code "scripts\run_verify.py") --out results\22_verifiable --workers $Workers
# study 17 needs the real-answer bundle (analysis\build_real_answers.py: MMLU + Ollama)
if (Test-Path data\real_answers\bundle.npz) {
    & $py (Join-Path $code "scripts\run_real_answers.py") --out results\17_real_answers --workers $Workers
}
# study 16 needs the real-card bundle (analysis\build_real_cards.py, Ollama) and the LLM run needs Ollama
if (Test-Path data\real_cards\bundle.npz) {
    & $py (Join-Path $code "scripts\run_experiment.py") --sweep (Join-Path $code "configs\sweep_real_cards.yaml") --out results\16_real_cards\embedding --workers $Workers
}
# studies 13-14 need the ERC-8004 crawl in data\raw\erc8004 (README: Reproducing)
if (Test-Path data\raw\erc8004\base_identities.csv) {
    & $py analysis\dense_subgraph.py
    foreach ($c in "ethereum", "base") { & $py (Join-Path $code "scripts\twin.py") backtest --chain $c }
    & $py analysis\twin_committee.py
    & $py analysis\deposit.py
}
& $py analysis\make_figures.py
& $py analysis\make_numbers.py
Push-Location paper
& (Join-Path $Root "tools\tectonic.exe") -X compile main.tex
Pop-Location
