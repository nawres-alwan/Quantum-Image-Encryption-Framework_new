# Quantum Image Encryption Framework — research code and data

هذا المستودع يجمع كود العمل والصور الأصلية والنتائج والبيانات وبرامج **جدول 7**، إضافة إلى التحليل الرياضي وتجارب Grover والدوائر المحفوظة. جميع الصور والملفات البحثية الموجودة في مجلدي العمل مضمّنة؛ استُثنيت بيئات Python وملفات التخزين المؤقت والمفاتيح الخاصة.

## Where to find the files

| Material | Location |
|---|---|
| Original image-processing code, original photographs, encrypted/decrypted exports, and diagrams | [`images encryption and decryption models/`](images%20encryption%20and%20decryption%20models/) |
| Corrected QEF–QIRBP v2 implementation, input images, 24-run data, and analysis | [`new code 19-10-2026/`](new%20code%2019-10-2026/) |
| **Table 7 generator, all 128 detailed rows, LaTeX table, and independent verifier** | [`QEF_All_Models_Source_Audit_20261019/`](new%20code%2019-10-2026/analysis/QEF_All_Models_Source_Audit_20261019/) |
| Earlier rotation-moment derivation and supporting data | [`QEF_Moment_Proof_20261019/`](new%20code%2019-10-2026/analysis/QEF_Moment_Proof_20261019/) |
| Original-operator attack-cost analysis and verification programs | [`Original_QEF_Attack_Cost_20261019/`](new%20code%2019-10-2026/analysis/Original_QEF_Attack_Cost_20261019/) |
| Coherent Grover verification oracle, tests, plots, resource tables, and QPY circuits | [`Grover_Oracle_20261021/`](new%20code%2019-10-2026/analysis/Grover_Oracle_20261021/) |

## Scientific scope

The original array-based demonstration and the repaired unitary QIRBP v2 implementation are different implementations. They are retained separately for provenance and comparison. The original scripts are historical research code, including known limitations described in the source audit.

The v2 implementation uses OS randomness through Python `secrets`, SHAKE-derived mixing schedules, and an explicit uniform privacy core. It does **not** reproduce or certify the original HCS/HG-Phi source law. Its full-image runs are classical simulations, and diagnostic PNG previews are not reversible classical ciphertexts or a security proof.

Table 7 contains **8 models × 4 images × 4 reference angle laws = 128 detailed CSV rows**, displayed as 32 model/image rows in LaTeX. Its values are conditional mathematical sample bounds; `NC` means the original source conditions have not been certified. These are not measured attack times or security-bit guarantees. The Grover experiment uses a finite public bank of 16 candidates; it does not implement a full 384-bit attack.

## Install

Use Python 3 and a new virtual environment outside the tracked source folders:

```shell
python -m venv .venv
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
```

For Table 7 and the QIRBP v2 core alone, NumPy and Pillow are sufficient. The complete requirements also cover the original plotting scripts and the Grover circuits. Per-experiment requirements are retained in their original folders.

## Verify the public package

From the repository root:

```shell
python -B verify_publication.py
```

This checks the publication hashes, Python syntax, JSON files, the four packaged input images, all eight saved profile/image sets, the 24-run records, and all 128 Table 7 rows. It also runs four independent dense-operator checks and the historical array-code regression check. Table 7 is regenerated in a temporary directory, without overwriting the published outputs. No private keys are needed.

`PUBLICATION_CHECKS.json` records the pre-publication checks. Files under `results/` and the earlier delivery manifests retain historical experiment evidence; `publication_manifest.json` is the checksum inventory for this public copy. Workstation paths in text metadata were replaced with repository-relative or historical-source references. The Python encryption algorithms and published scientific tables were not changed for packaging.

## Reproduce Table 7

```shell
python "new code 19-10-2026/analysis/QEF_All_Models_Source_Audit_20261019/calculate_all_models.py"
python "new code 19-10-2026/analysis/QEF_All_Models_Source_Audit_20261019/verify_extension.py"
```

The first program regenerates `Table7_All_Models.csv`, `Table7_All_Models.tex`, and the earlier `r=11` reference calculation. The second independently recomputes the table and checks the supporting finite reference models. These commands update their generated files; the checked-in versions remain available in Git history.

## Run fresh QIRBP v2 image experiments

```shell
cd "new code 19-10-2026"
python security_tests.py
python run_experiments.py --replicates 3 --chunk-size 2048
python analyze_results.py
```

These commands generate fresh private keys locally and update the experiment outputs. Timings and key-dependent previews will differ from the archived run. Existing keys are not silently overwritten. `REPRODUCE.ps1` defaults to fresh keys; `-ReplayRecordedKeys` and `verify_package.py` are for an offline researcher who already has the original private records. Exact replay of the archived secret-key-dependent run is not available from this public repository.

## Run the Grover oracle experiment

```shell
cd "new code 19-10-2026/analysis/Grover_Oracle_20261021"
python -X utf8 test_qef_gsa_oracle.py
python -X utf8 qef_gsa_oracle.py
```

The folder includes the original CSV/JSON evidence, PNG/SVG drawings, and actual QPY circuits, including the full oracle. Its README explains the access model, candidate register, approximation error, and gate counts.

## Publication exclusions

The repository intentionally excludes `private_keys/`, optional private state checkpoints, virtual environments, credentials, and Python caches. Fresh experiment keys remain local and are ignored by Git. All source photographs and research output files in the two supplied work folders are otherwise retained, including historical versions.

No new software or image license is assigned by this packaging step. Consult the authors before redistribution or reuse beyond the applicable rights.
