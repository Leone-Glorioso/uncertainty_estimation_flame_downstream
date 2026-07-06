#!/usr/bin/env bash
# =============================================================================
# run_experiments.sh  —  3D Face Uncertainty Pipeline · Experiment Runner
# =============================================================================
export PYTHONUNBUFFERED=1   # live log lines under SLURM / pipe redirection
#
# USAGE
#   bash run_experiments.sh                        run everything with defaults
#   bash run_experiments.sh --no-downstream        Stages 1-4 only
#   bash run_experiments.sh --downstream-only      Stage 5 only
#   bash run_experiments.sh --tune-only            Stage 0 (hyperparameter tuning) only
#   bash run_experiments.sh --with-tuning          Stage 0 then Stages 1-4-5
#   bash run_experiments.sh --downstream-tune-only Stage 6 (downstream HP tuning) only
#   bash run_experiments.sh --dataset tempeh       single dataset, Stages 1-4
#   bash run_experiments.sh --sweep                enable model × method sweep in Stages 1-4
#   bash run_experiments.sh --no-sweep             disable model × method sweep
#   bash run_experiments.sh --downstream-fusion MODE
#                                                  override downstream uncertainty injection mode.
#                                                  MODE: input | patch_embed | attn_bias |
#                                                        key_scale | value_scale | all
#                                                  CPU default: input.  GPU/HPC default: all.
#   bash run_experiments.sh --dry-run              print every command, don't execute
#   bash run_experiments.sh --help                 show this message
#
# SUBSET SELECTION FLAGS (GPU/HPC; override defaults without editing the file)
#   --models  M1,M2,...     comma-separated regressors to use (and sweep over)
#                           values: SMIRK DECA EMOCA SHeaP
#                           example: --models SMIRK,DECA,EMOCA
#   --datasets D1,D2,...    comma-separated datasets to run
#                           values: tempeh now coma utkface lfw
#                           example: --datasets tempeh,now
#   --methods M1,M2,...     comma-separated uncertainty methods (or "all" / "all_no_dropout")
#                           values: tta jacobian mahalanobis mcd sol_mcd amcd cross
#                           example: --methods tta,jacobian,cross
#                           example: --methods all_no_dropout   (skip MCD checkpoint)
#   --ds-models M1,M2,...   downstream classifier: models for confidence-map precomputation
#                           example: --ds-models SMIRK,DECA
#   --ds-methods M1,M2,...  downstream classifier: uncertainty methods for confidence maps
#                           values: tta jacobian mahalanobis mcd sol_mcd amcd
#                           (mcd/sol_mcd/amcd are SMIRK-only; others run for all ds-models)
#                           example: --ds-methods tta,jacobian,mahalanobis
#
# SWEEP DETAILS
#   With --sweep (GPU/HPC default), Stages 1-4 expand per dataset into
#   one run per MODEL (not per model×method), so each run has all its
#   applicable uncertainty methods together — producing full comparison plots:
#     • SMIRK → tta + jacobian + mahalanobis + mcd + sol_mcd + amcd
#     • DECA / EMOCA / SHeaP → tta + jacobian + mahalanobis  (no MCD)
#     • CrossMethod run with all selected models (separate, requires ≥2)
#   Default: 4 per-model runs + 1 cross = 5 runs per dataset.
#   --methods all_no_dropout → all 4 models get tta+jacobian+mahalanobis + 1 cross.
#   --models SMIRK,DECA → 2 per-model runs + 1 cross = 3 runs.
#   Running one method at a time (old behaviour) produced only the input image
#   in each output dir because comparison plots require ≥2 methods per run.
#
#   Stage 5 sweep:
#     DOWNSTREAM_SUBSETS × DOWNSTREAM_EPOCHS × DOWNSTREAM_LR
#     × DOWNSTREAM_N_TTA_VALUES × DOWNSTREAM_METHODS
#     When DOWNSTREAM_FUSION=all, each weighted variant is also split across
#     all 5 fusion modes (input/patch_embed/attn_bias/key_scale/value_scale).
#
# DEVICE MODES
#   --cpu                 CPU run, CPU-scaled defaults.
#   --gpu                 GPU run, GPU-scaled defaults.
#   --hpc [small|medium|large|full]
#                         HPC cluster run; defaults to 'large' if tier omitted.
#                         Unlocks MC-Dropout (METHODS="all") but requires the
#                         retrained SMIRK checkpoint at models/SMIRK/smirk_mcd.pth
#                         (see CLAUDE.md). Without it, set METHODS="all_no_dropout".
#
# ESTIMATED TIMES
#   CPU — one dataset run (ps=5, tta=10)         : ~15 min
#   CPU — one downstream run (subset=25, ep=5)   : ~50 min
#   GPU — one dataset run (ps=10, tta=10)        : ~3 min
#   GPU — one downstream run (subset=200, ep=10) : ~25 min
#   GPU — full model×method sweep (16×datasets)  : ~50 min
#
# EXAMPLE COMMANDS
#   # CPU — all methods + downstream (scaled down)
#   bash run_experiments.sh --cpu
#
#   # GPU — everything: all models, all methods including MCD, all fusion modes
#   bash run_experiments.sh --gpu
#
#   # GPU — skip MCD (no checkpoint), only SMIRK+DECA, only tempeh dataset
#   bash run_experiments.sh --gpu --methods all_no_dropout --models SMIRK,DECA --datasets tempeh
#
#   # GPU — Stages 1-4 sweep only, no downstream
#   bash run_experiments.sh --gpu --no-downstream
#
#   # GPU — downstream only with specific models and fusion mode
#   bash run_experiments.sh --gpu --downstream-only --ds-models SMIRK,EMOCA --downstream-fusion attn_bias
#
#   # HPC — ultimate full run (all models, all methods, all datasets, all fusion modes)
#   bash run_experiments.sh --hpc large
# =============================================================================


# =============================================================================
# ① DEVICE MODE
# =============================================================================

# Three mutually exclusive modes — set at runtime, do not edit this line.
#   --cpu                 Force CPU; applies CPU-scaled defaults.
#   --gpu                 CUDA GPU; applies GPU-scaled defaults.
#   --hpc [small|medium|large|full|all]
#                         CUDA GPU + HPC-scaled defaults.  Tier defaults to
#                         'large' when omitted.  'all' runs every tier in
#                         sequence (small→medium→large→full).  Requires the
#                         MCD checkpoint.
DEVICE_MODE="cpu"

# Pre-scan args so the defaults block below sees the correct mode.
_prescan_args=("$@")
for (( _i=0; _i<${#_prescan_args[@]}; _i++ )); do
    _prescan_arg="${_prescan_args[$_i]}"
    _prescan_next="${_prescan_args[$(( _i+1 ))]:-}"
    case "$_prescan_arg" in
        --cpu) DEVICE_MODE="cpu" ;;
        --gpu) DEVICE_MODE="gpu" ;;
        --hpc)
            case "$_prescan_next" in
                small|medium|large|full|all) DEVICE_MODE="hpc_${_prescan_next}" ;;
                *) DEVICE_MODE="hpc_large" ;;
            esac
            ;;
    esac
done

# --hpc all: re-invoke this script once per tier in order, then exit.
if [[ "$DEVICE_MODE" == "hpc_all" ]]; then
    _remaining_args=()
    _skip_next=false
    for _a in "$@"; do
        if [[ "$_skip_next" == true ]]; then _skip_next=false; continue; fi
        if [[ "$_a" == "--hpc" ]];       then _skip_next=true; continue;  fi
        _remaining_args+=("$_a")
    done
    for _tier in small medium large full; do
        echo ""
        echo "============================================================"
        echo "  --hpc all: launching tier $_tier"
        echo "============================================================"
        bash "$0" --hpc "$_tier" "${_remaining_args[@]}"
    done
    exit 0
fi

# Derived for backward-compat with cpu_flag logic below.
USE_GPU=false
[[ "$DEVICE_MODE" != "cpu" ]] && USE_GPU=true


# =============================================================================
# ② AUTO-SCALED DEFAULTS  (applied when the config arrays below are unchanged)
#    Override any of these freely — they are just starting points.
# =============================================================================

if [[ "$DEVICE_MODE" != "cpu" ]]; then
    # ── GPU defaults ──────────────────────────────────────────────────────────
    _DEF_DATASETS="tempeh now coma utkface lfw"
    # All four regressors; EMOCA is included so cross-method uses the full ensemble
    # and EMOCA gets its own per-model sweep runs.
    _DEF_MODELS="SMIRK DECA EMOCA SHeaP"
    # "all" unlocks MCD/SOL-MCD/A-MCD — requires smirk_checkpoint_data/trained.pt.
    # If you don't have it, change this to "all_no_dropout".
    _DEF_METHODS="all"
    _DEF_PARTITION_SIZES="10"
    _DEF_N_TTA="10"
    _DEF_N_MCD="15"          # MC-Dropout forward passes (methods 2, 6, 7)
    _DEF_N_JACOBIAN="8"
    _DEF_N_MAHAL_REF="20"    # Mahalanobis reference images (non-overlapping w/ test split)
    # Downstream
    # EMOCA included: its emotion-discriminative training makes it a strong
    # candidate for uncertainty-guided expression classification.
    _DEF_DS_MODELS="SMIRK DECA EMOCA SHeaP"   # all 4 models on GPU
    _DEF_DS_METHODS="tta jacobian mahalanobis mcd sol_mcd amcd"  # all applicable methods on GPU/HPC
    _DEF_DS_SUBSETS="200"
    _DEF_DS_N_TTA="5"
    _DEF_DS_EPOCHS="10"
    _DEF_DS_LR="2e-4"
    _DEF_DS_BATCH="64"
    _DEF_DS_BACKBONE="vit_b_16"         # stronger backbone on GPU
    _DEF_DOWNSTREAM_FUSION="all"        # run all 5 injection modes on GPU
else
    # ── CPU defaults ──────────────────────────────────────────────────────────
    _DEF_DATASETS="tempeh now coma utkface lfw"
    _DEF_MODELS="SMIRK DECA SHeaP"
    _DEF_METHODS="all"                  # smirk_checkpoint_data/trained.pt present
    _DEF_PARTITION_SIZES="5"
    _DEF_N_TTA="10"
    _DEF_N_MCD="15"                     # only used if METHODS includes mcd/sol_mcd/amcd
    _DEF_N_JACOBIAN="8"
    _DEF_N_MAHAL_REF="20"               # Mahalanobis reference images (non-overlapping w/ test split)
    # Downstream
    _DEF_DS_MODELS="SMIRK"              # limit to 1 model to keep CPU time manageable
    _DEF_DS_METHODS="tta"
    _DEF_DS_SUBSETS="25"
    _DEF_DS_N_TTA="3"
    _DEF_DS_EPOCHS="20"
    _DEF_DS_LR="2e-4"
    _DEF_DS_BATCH="32"
    _DEF_DS_BACKBONE="vit_b_32"         # lightest backbone on CPU
    _DEF_DOWNSTREAM_FUSION="input"      # pixel masking only on CPU (no ViT-internal ablation)
fi

# =============================================================================
# ② b  HPC TIERS  (activated by --hpc small | medium | large | full)
#    Overrides all Stages 1-4 and downstream parameters in one flag.
#    All four tiers require --hpc (implied GPU) and the MCD checkpoint.
#
#    hpc_small   ViT-B/16  (86 M params, 16-px patches)   — fast baseline
#    hpc_medium  ViT-L/16  (307 M params, 16-px patches)  — balanced
#    hpc_large   ViT-H/14  (632 M params, 14-px patches)  — high capacity
#    hpc_full    ViT-H/14  (632 M params, 14-px patches)  — largest safe sample
#
#    N_TTA is raised per tier: Stages 1-4 batch eval uses the full n_tta value
#    (main.py:1119, uncapped). The gallery was previously capped at 5 — that
#    cap has been removed so gallery figures also show full-resolution TTA.
#
#    PARTITION_SIZES ceiling: a single value is shared across every dataset,
#    including the smallest GT-bearing ones (now=352, tempeh=500 images).
#    The Mahalanobis loader draws partition_size + n_mahal_ref + 5 images in
#    one non-overlapping pool; hpc_full's combined draw (285) stays safely
#    under now's 352-image ceiling. "full" means "largest sample size that
#    still guarantees the non-overlapping test/reference split."
# =============================================================================
case "$DEVICE_MODE" in
    hpc_small)
        _DEF_DS_BACKBONE="vit_b_16"
        _DEF_DS_BATCH="64"
        _DEF_DS_SUBSETS="100"
        _DEF_DS_EPOCHS="15"
        _DEF_PARTITION_SIZES="20"
        _DEF_N_TTA="20"
        _DEF_N_JACOBIAN="20"
        _DEF_N_MCD="30"
        _DEF_N_MAHAL_REF="20"
        _DEF_DOWNSTREAM_FUSION="all"
        ;;
    hpc_medium)
        _DEF_DS_BACKBONE="vit_l_16"
        _DEF_DS_BATCH="32"
        _DEF_DS_SUBSETS="200"
        _DEF_DS_EPOCHS="12"
        _DEF_PARTITION_SIZES="50"
        _DEF_N_TTA="30"
        _DEF_N_JACOBIAN="40"
        _DEF_N_MCD="50"
        _DEF_N_MAHAL_REF="40"
        _DEF_DOWNSTREAM_FUSION="all"
        ;;
    hpc_large)
        _DEF_DS_BACKBONE="vit_b_16"   # vit_l_16 (307M) → vit_b_16 (86M): 3× faster training
        _DEF_DS_BATCH="32"
        _DEF_DS_SUBSETS="1429"        # 1429/class × 7 = 10,003 train; test uncapped (full 3,589)
        _DEF_DS_EPOCHS="10"
        _DEF_PARTITION_SIZES="100"
        _DEF_N_TTA="40"
        _DEF_N_JACOBIAN="60"
        _DEF_N_MCD="75"
        _DEF_N_MAHAL_REF="60"
        _DEF_DOWNSTREAM_FUSION="all"
        ;;
    hpc_full)
        _DEF_DS_BACKBONE="vit_l_16"   # vit_h_14 OOMs on A100; use vit_l_16 (224px, 307M params)
        _DEF_DS_BATCH="32"
        _DEF_DS_SUBSETS="0"
        _DEF_DS_EPOCHS="10"
        _DEF_PARTITION_SIZES="200"
        _DEF_N_TTA="50"
        _DEF_N_JACOBIAN="100"
        _DEF_N_MCD="100"
        _DEF_N_MAHAL_REF="80"
        _DEF_DOWNSTREAM_FUSION="all"
        ;;
    cpu|gpu)
        ;;   # no overrides — keep defaults from block ②
    *)
        echo "Unknown mode: '$DEVICE_MODE'. Use --cpu, --gpu, or --hpc [small|medium|large|full]"
        exit 1
        ;;
esac


# =============================================================================
# ③ CONFIGURATION — edit this block to tune experiments
#    Change the array contents.  Single-element = no sweep.
#    Multi-element = sweep one run per element (Cartesian with other arrays).
# =============================================================================

# ── Stages 1-4 ───────────────────────────────────────────────────────────────

# Datasets.  Each entry → one output directory per hyperparameter combination.
#   tempeh  images + FLAME GT  → full suite: Spearman ρ, AUSE, ECE, NLL, s2m
#   now     images + raw scans → scan-to-mesh only
#   coma    FLAME GT, no images → geometric stats + Mahalanobis reference data
#   utkface images, no GT      → uncertainty maps + gallery (no metric scores)
#   lfw     images, no GT      → uncertainty maps + gallery (no metric scores)
read -ra DATASETS        <<< "$_DEF_DATASETS"

# Models loaded for inference and cross-method disagreement.
read -ra MODELS          <<< "$_DEF_MODELS"

# Uncertainty methods.
#   all_no_dropout  TTA + CrossMethod + Jacobian + Mahalanobis (no checkpoint needed)
#   all             all 7 methods including MCD/SOL-MCD/A-MCD  (GPU + MCD ckpt only)
#   tta crossmethod jacobian mahalanobis mcd sol_mcd amcd      (individual)
METHODS="$_DEF_METHODS"

# Partition sizes: number of dataset images for batch eval + gallery + Mahal ref.
# Sweep example: PARTITION_SIZES=( 5 10 20 )
read -ra PARTITION_SIZES <<< "$_DEF_PARTITION_SIZES"

# TTA forward passes for Stage 3 and per-image gallery (gallery capped at 5).
read -ra N_TTA_VALUES    <<< "$_DEF_N_TTA"

# MC-Dropout passes (methods 2 / 6 / 7).  Ignored when METHODS=all_no_dropout.
read -ra N_MCD_VALUES    <<< "$_DEF_N_MCD"

# Jacobian random-projection directions.  Higher = smoother map, slower.
read -ra N_JACOBIAN_VALUES <<< "$_DEF_N_JACOBIAN"

# Mahalanobis reference image count (non-overlapping w/ test partition; see
# main.py's _load_partition_data). Bounded by the smallest GT dataset's size
# minus partition_size — see the HPC preset comment above for the math.
N_MAHAL_REF="$_DEF_N_MAHAL_REF"

# Primary model for single-model methods (TTA, Jacobian, Mahalanobis).
PRIMARY_MODEL="SMIRK"

# Featured test image for Stages 2-3 (single-image heatmaps + mesh panels).
#
# Leave empty (recommended) → the pipeline auto-selects the first image from
# each dataset's own partition, so TEMPEH runs show a TEMPEH face, NoW runs
# show a NoW face, etc.  CoMA has no images, so Stages 2-3 are cleanly skipped.
#
# Set a specific path only if you want to force ONE fixed face across all
# datasets (useful for a controlled side-by-side comparison).
# Example: IMAGE_PATH="./datasets/utkface/100_0_0_20170112213500903.jpg.chip.jpg"
IMAGE_PATH=""

# ── Stage 5 (downstream expression classifier) ───────────────────────────────

# Set false to skip Stage 5 entirely.
RUN_DOWNSTREAM=true

# Models used for confidence-map precomputation.  Each model × method pair
# trains a separate plain + weighted classifier comparison.
# CPU: keep to 1 model.  GPU: all 3 is fine.
read -ra DOWNSTREAM_MODELS   <<< "$_DEF_DS_MODELS"

# Method used to build per-pixel confidence maps.
#   tta       — TTA variance projected to 2D  (CPU-friendly)
#   jacobian  — Jacobian sensitivity map      (GPU recommended)
# Multiple values → one downstream run per value.
read -ra DOWNSTREAM_METHODS  <<< "$_DEF_DS_METHODS"

# Train images per class (test set always uncapped — full test for statistical power).
#   25   → 175 train  + full test  (~50 min CPU  /  ~5 min GPU)
#   100  → 700 train  + full test  (~3.5 hr CPU  /  ~20 min GPU)
#   1429 → 10003 train + full test (hpc_large default — ~2.5h maps + ~1h train)
#   0    → full dataset — hours on CPU, not recommended
read -ra DOWNSTREAM_SUBSETS  <<< "$_DEF_DS_SUBSETS"

# TTA passes for confidence-map precomputation (independent of N_TTA_VALUES).
read -ra DOWNSTREAM_N_TTA_VALUES <<< "$_DEF_DS_N_TTA"

# Training epochs.
read -ra DOWNSTREAM_EPOCHS   <<< "$_DEF_DS_EPOCHS"

# Initial learning rate.
# Good sweep: DOWNSTREAM_LR=( 1e-4 2e-4 5e-4 )
read -ra DOWNSTREAM_LR       <<< "$_DEF_DS_LR"

# Batch size.
DOWNSTREAM_BATCH_SIZE="$_DEF_DS_BATCH"

# Backbone architecture.
#   vit_b_32  87M params, 32-px patches — CPU-friendly (default CPU)
#   vit_b_16  86M params, 16-px patches — GPU recommended (default GPU)
#   vit_l_16  307M params              — GPU required (default HPC; fits A100 at batch 32)
#   vit_h_14  632M params              — OOMs on A100 at 518px; only use if VRAM>80GB
#   resnet50  25M params               — fast, weaker accuracy
#   resnet18  11M params               — fastest, weakest
DOWNSTREAM_BACKBONE="$_DEF_DS_BACKBONE"

# Downstream uncertainty injection mode (see --downstream_fusion in main.py).
#   input       — multiply image by (1 − α·U_2D) before backbone  [CPU default]
#   patch_embed — scale conv_proj patch tokens after embedding
#   attn_bias   — subtract α·U_j from pre-softmax attention logits (PyTorch ≥ 2.0)
#   key_scale   — scale MHA key projections (uncertain patches attract less attention)
#   value_scale — scale MHA value projections (uncertain patches contribute less)
#   all         — run all 5 modes as separate sub-experiments  [GPU/HPC default]
DOWNSTREAM_FUSION="$_DEF_DOWNSTREAM_FUSION"
DOWNSTREAM_FUSION_ALPHA="1.0"

# Regularization — critical for preventing overfitting on small datasets.
#   Weight decay: 0.05 is standard for ViT fine-tuning (DeiT/MAE convention).
#   Patience: early-stop after N epochs without test-acc improvement.
#   Unfreeze blocks: number of trailing ViT transformer blocks opened in Stage 2.
#     (-1 = all; 6 = last 6 of 24 for ViT-L; reduces fine-tune params ~5×).
#   Mixup alpha: Beta(α,α) mixing coefficient (0 = disabled).
DOWNSTREAM_WEIGHT_DECAY="0.05"
DOWNSTREAM_PATIENCE="3"
DOWNSTREAM_UNFREEZE_BLOCKS="12"   # 6/24 was too few — underfitting; 12 gives more capacity
DOWNSTREAM_MIXUP_ALPHA="0.1"      # 0.2 was too strong with partial freeze; 0.1 is gentler
DOWNSTREAM_CURRICULUM_START="0.5" # start Stage 2 seeing 50% of training data, grow to 100%

# Emotion dataset root — point this at your organised FER2013 directory.
# Download: kaggle datasets download -d msambare/fer2013
# Organise:  python scripts/organise_fer2013.py --src /path/to/fer2013 --dst /path/to/datasets/fer2013
# Must contain train/{anger,disgust,fear,happy,neutral,sad,surprise}/ and test/(same).
EMOTION_DB_ROOT="./datasets/fer2013"

# =============================================================================
# Model × uncertainty-method sweep (opt-in via --sweep on any device mode)
# =============================================================================
#
# When SWEEP_METHOD_MODEL=true, Stages 1-4 expand into one run per MODEL:
#
#   Each model in SWEEP_MODELS gets ONE invocation with ALL its applicable
#   uncertainty methods together — exactly like the CPU run, just one model
#   at a time so comparison plots (sparsification curves, method tables,
#   heatmap grids) are fully generated per model.
#
#   SMIRK  → tta + jacobian + mahalanobis + mcd + sol_mcd + amcd  (if SWEEP_METHODS=all)
#   DECA   → tta + jacobian + mahalanobis                           (no MCD checkpoint)
#   EMOCA  → tta + jacobian + mahalanobis
#   SHeaP  → tta + jacobian + mahalanobis
#   Cross  → all SWEEP_MODELS simultaneously (separate run)
#
# Default totals: 4 per-model runs + 1 cross = 5 runs per dataset.
#
# Running methods one-at-a-time (old behaviour) produces only the raw input
# image in each output directory because comparison plots need ≥2 methods.
#
# Override with --sweep / --no-sweep on the command line.
# Sweep mode is opt-in only via --sweep; default matches the CPU run
# (all models, all methods, one python3 main.py call per dataset).
SWEEP_METHOD_MODEL=false

# Models and methods used inside the sweep.
# These default to MODELS/METHODS but can be overridden independently
# via --models / --methods on the CLI without changing the regular run config.
SWEEP_MODELS=("${MODELS[@]}")
SWEEP_METHODS="$METHODS"

# Delete and regenerate confidence maps even when cached .npy files exist.
# Leave false unless you change n_tta/subset/pipeline code and need fresh maps.
# Maps are stored in MAPS_CACHE_DIR (stable across runs) so setting this true
# is rarely needed.
FORCE_RECOMPUTE=false

# Stable directory for pre-computed confidence maps.
# Stored outside the timestamped output_dir so maps are reused across runs.
# On HPC set this to an absolute path on shared storage.
MAPS_CACHE_DIR="/leonardo_work/EUHPC_D34_205/cv26_team3/datasets/maps"

# Root directory for all output folders.
# All run outputs land inside ./figures/output_<RUN_ID>/ — see below.
# Override this base if you want a different top-level parent (e.g. /scratch/outputs).
_FIGURES_BASE="./figures"

# ── Stage 0 (hyperparameter tuning) ─────────────────────────────────────────

# Set true to run hyperparameter tuning before Stages 1-4.
# Tuning is a one-time setup step; once hyperparams.json is saved you can
# disable this and pass the optimal values via N_TTA_VALUES etc.
# Default: false (tuning is opt-in; use --tune-only or --with-tuning to enable).
RUN_TUNING=false

# ── Stage 6 (downstream hyperparameter tuning) ───────────────────────────────

# Set true to run downstream classifier hyperparameter tuning.
# Use --downstream-tune-only to run only this stage.
# IMPORTANT: Run --stage downstream (Stage 5) first to generate confidence maps.
#            Stage 6 auto-discovers maps from DOWNSTREAM_TUNE_MAPS_DIR or from
#            the Stage-5 output inside OUTPUT_ROOT/downstream_{N}cls_*/downstream/maps/.
# Default: false.
RUN_DOWNSTREAM_TUNE=false

# Output directory for downstream tuning results.  Re-derived after CLI
# parsing once the unique RUN_ID is known.  To override, set this variable
# after the CLI parsing block below (search for "Re-derive after CLI").
DOWNSTREAM_TUNE_OUTPUT_DIR=""   # placeholder — set after RUN_ID is built

# Directory containing pre-computed confidence-map trees.
# Each sub-directory must be a model×method combo (e.g. SMIRK_tta/) that
# mirrors the RAF-DB image tree with .npy files.
# Leave empty → auto-discover from DOWNSTREAM_TUNE_OUTPUT_DIR/downstream/maps/.
DOWNSTREAM_TUNE_MAPS_DIR=""

# Random seeds per config (2 CPU / 3 GPU gives reliable estimates with fast runs).
DOWNSTREAM_TUNE_N_SEEDS=""   # empty → use default (2 CPU, 3 GPU)

# Max training epochs per trial (60 CPU / 30 GPU; early stopping usually kicks
# in well before this ceiling).
DOWNSTREAM_TUNE_MAX_EPOCHS=""   # empty → use default

# Early-stopping patience (12 CPU / 7 GPU).
DOWNSTREAM_TUNE_PATIENCE=""     # empty → use default

# Number of search trials for Phase 1 (100 CPU / 40 GPU with feature caching).
# Ignored when DOWNSTREAM_TUNE_SEARCH=grid.
DOWNSTREAM_TUNE_N_TRIALS=""     # empty → use default

# Search strategy: auto | tpe | random | grid
#   auto   → Optuna TPE when available, otherwise log-uniform random (recommended)
#   tpe    → Optuna Tree-structured Parzen Estimator (pip install optuna)
#   random → log-uniform random sampling (no extra deps)
#   grid   → 48-point grid (CPU only, original behaviour)
DOWNSTREAM_TUNE_SEARCH=""       # empty → auto

# Dataset to draw GT-paired images from for tuning.
# Must have gt_vertices: tempeh, coma, now (not utkface or lfw).
TUNE_DATASET="tempeh"

# Number of GT-paired images to use for tuning (5–10 is enough; more = slower).
#   CPU: 8 images, non-MCD only   → ~6 min
#   CPU: 8 images, all 7 methods  → ~13 min  (MCD signal check adds ~0.5 min)
#   GPU: 8 images, all 7 methods  → ~1 min
TUNE_N_IMAGES=8

# Which uncertainty methods to include in the search.
# Non-MCD (safe on CPU): tta cross jacobian mahalanobis
# MCD (needs smirk_checkpoint_data/trained.pt): mcd sol_mcd amcd
# smirk_checkpoint_data/trained.pt is present — all 7 methods are enabled.
# If the MCD checkpoint is missing, the signal check will skip mcd/sol_mcd/amcd
# automatically; the other four methods will still be tuned.
TUNE_METHODS="tta cross jacobian mahalanobis mcd sol_mcd amcd"

# Optimisation objective: spearman_rho (rank correlation, default) or ause.
TUNE_OBJECTIVE="spearman_rho"


# =============================================================================
# DO NOT EDIT BELOW — internal logic
# =============================================================================

set -euo pipefail

# ── CLI argument parsing ──────────────────────────────────────────────────────
MODE="all"            # all | no_downstream | downstream_only | tune_only | with_tuning | downstream_tune_only
FILTER_DATASET=""     # non-empty → only that dataset in Stages 1-4
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --cpu)  DEVICE_MODE="cpu"; USE_GPU=false;                  shift ;;
        --gpu)  DEVICE_MODE="gpu"; USE_GPU=true;                 shift ;;
        --hpc)
            USE_GPU=true
            case "${2:-}" in
                small|medium|large|full|all) DEVICE_MODE="hpc_${2}"; shift 2 ;;
                *) DEVICE_MODE="hpc_large";                           shift   ;;
            esac
            ;;
        --no-downstream)          MODE="no_downstream";         shift ;;
        --downstream-only)        MODE="downstream_only";       shift ;;
        --tune-only)              MODE="tune_only";              shift ;;
        --with-tuning)            MODE="with_tuning";            shift ;;
        --downstream-tune-only)   MODE="downstream_tune_only";  shift ;;
        --dataset)                FILTER_DATASET="${2:-}";      shift 2 ;;
        --sweep)                  SWEEP_METHOD_MODEL=true;       shift ;;
        --no-sweep)               SWEEP_METHOD_MODEL=false;      shift ;;
        --downstream-fusion)
            case "${2:-}" in
                input|patch_embed|attn_bias|key_scale|value_scale|all)
                    DOWNSTREAM_FUSION="${2}"; shift 2 ;;
                *)
                    echo "Unknown --downstream-fusion value: '${2:-}'"
                    echo "  Valid: input  patch_embed  attn_bias  key_scale  value_scale  all"
                    exit 1 ;;
            esac
            ;;
        # ── Subset selection flags ───────────────────────────────────────────────
        # Accept comma-separated or space-separated lists.
        # Examples:
        #   --models SMIRK,DECA,EMOCA
        #   --datasets tempeh,now
        #   --methods tta,jacobian,mahalanobis
        #   --ds-models SMIRK,DECA
        #   --ds-methods tta
        --models)
            IFS=',' read -ra MODELS <<< "${2:-}"
            SWEEP_MODELS=("${MODELS[@]}")
            shift 2 ;;
        --datasets)
            IFS=',' read -ra DATASETS <<< "${2:-}"
            shift 2 ;;
        --methods)
            # Accept "all", "all_no_dropout", or comma-separated method names.
            METHODS="${2:-all}"
            SWEEP_METHODS="$METHODS"
            shift 2 ;;
        --ds-models)
            IFS=',' read -ra DOWNSTREAM_MODELS <<< "${2:-}"
            shift 2 ;;
        --ds-methods)
            IFS=',' read -ra DOWNSTREAM_METHODS <<< "${2:-}"
            shift 2 ;;
        --dry-run)                DRY_RUN=true;                  shift ;;
        --help) sed -n '3,40p' "$0"; exit 0 ;;
        *) echo "Unknown option: $1  (use --help)"; exit 1 ;;
    esac
done

# ── Unique run identifier + output root ──────────────────────────────────────
# Built here, after CLI parsing, so DEVICE_MODE, MODE, and SWEEP_METHOD_MODEL
# all reflect any overrides the user passed on the command line.
#
# Format: output_YYYYMMDD_HHMMSS_<device>_<mode>[_sweep]
# Examples:
#   output_20260630_143022_gpu_all_sweep
#   output_20260630_143022_hpc_large_no_ds_sweep
#   output_20260630_143022_cpu_all
_run_ts="$(date +%Y%m%d_%H%M%S)"
_run_dev="${DEVICE_MODE}"    # cpu / gpu / hpc_small / hpc_medium / hpc_large / hpc_full
case "$MODE" in
    all)                   _run_mode_short="all"       ;;
    no_downstream)         _run_mode_short="no_ds"     ;;
    downstream_only)       _run_mode_short="ds_only"   ;;
    tune_only)             _run_mode_short="tune"       ;;
    with_tuning)           _run_mode_short="with_tune" ;;
    downstream_tune_only)  _run_mode_short="ds_tune"   ;;
    *)                     _run_mode_short="$MODE"      ;;
esac
_sweep_tag=""
[[ "$SWEEP_METHOD_MODEL" == true ]] && _sweep_tag="_sweep"
RUN_ID="${_run_ts}_${_run_dev}_${_run_mode_short}${_sweep_tag}"
OUTPUT_ROOT="${_FIGURES_BASE}/output_${RUN_ID}"

# Re-derive after CLI — paths that referenced the old OUTPUT_ROOT placeholder.
DOWNSTREAM_TUNE_OUTPUT_DIR="${OUTPUT_ROOT}/tuning/downstream"

# ── Device flag ───────────────────────────────────────────────────────────────
cpu_flag=()
[[ "$USE_GPU" == false ]] && cpu_flag=( --cpu )

# ── Force-recompute flag ──────────────────────────────────────────────────────
force_args=()
[[ "$FORCE_RECOMPUTE" == true ]] && force_args=( --force_recompute_maps )

# ── Image-path flag (non-downstream runs only) ────────────────────────────────
# When IMAGE_PATH is empty the flag is omitted entirely, letting the pipeline
# auto-select the first image from each dataset's own partition.
# When IMAGE_PATH is set, that specific image is used for every dataset run.
img_args=()
[[ -n "$IMAGE_PATH" ]] && img_args=( --image_path "$IMAGE_PATH" )

# ── Pretty-printer / executor ─────────────────────────────────────────────────
run() {
    if [[ "$DRY_RUN" == true ]]; then
        printf '\n[DRY-RUN]'
        local first=true
        for arg in "$@"; do
            if [[ "$first" == true ]]; then
                printf ' %s' "$arg"; first=false
            else
                printf ' \\\n    %s' "$arg"
            fi
        done
        printf '\n'
    else
        printf '\n▶  '
        printf '%s ' "$@"
        printf '\n\n'
        "$@"
    fi
}

# ── Experiment count ──────────────────────────────────────────────────────────
n_dataset_runs=0
if [[ "$MODE" != "downstream_only" ]]; then
    for _d  in "${DATASETS[@]}"; do
        [[ -n "$FILTER_DATASET" && "$_d" != "$FILTER_DATASET" ]] && continue
        for _ps  in "${PARTITION_SIZES[@]}";   do
        for _tta in "${N_TTA_VALUES[@]}";       do
        for _jac in "${N_JACOBIAN_VALUES[@]}";  do
            (( n_dataset_runs++ )) || true
        done; done; done
    done
fi

n_ds_runs=0
if [[ "$MODE" != "no_downstream" && "$RUN_DOWNSTREAM" == true ]]; then
    for _ in "${DOWNSTREAM_SUBSETS[@]}";      do
    for _ in "${DOWNSTREAM_EPOCHS[@]}";       do
    for _ in "${DOWNSTREAM_LR[@]}";           do
    for _ in "${DOWNSTREAM_N_TTA_VALUES[@]}"; do
    for _ in "${DOWNSTREAM_METHODS[@]}";      do
        (( n_ds_runs++ )) || true
    done; done; done; done; done
fi

_run_tuning_flag=false
[[ "$RUN_TUNING" == true || "$MODE" == "tune_only" || "$MODE" == "with_tuning" ]] \
    && _run_tuning_flag=true

_run_ds_tune_flag=false
[[ "$RUN_DOWNSTREAM_TUNE" == true || "$MODE" == "downstream_tune_only" ]] \
    && _run_ds_tune_flag=true

echo "============================================================"
echo "  3D Face Uncertainty — Experiment Runner"
printf  "  Device            : %s\n" "$DEVICE_MODE"
echo   "  Mode              : $MODE"
echo   "  Dry run           : $DRY_RUN"
echo   "  Run tuning        : $_run_tuning_flag"
echo   "  Datasets          : ${DATASETS[*]}"
echo   "  Models            : ${MODELS[*]}"
echo   "  Methods           : $METHODS"
echo   "  Dataset runs      : $n_dataset_runs"
echo   "  Model×method sweep: $SWEEP_METHOD_MODEL"
if [[ "$SWEEP_METHOD_MODEL" == true ]]; then
echo   "    Sweep models    : ${SWEEP_MODELS[*]}"
echo   "    Sweep methods   : $SWEEP_METHODS"
fi
echo   "  Downstream runs   : $n_ds_runs"
echo   "  DS models         : ${DOWNSTREAM_MODELS[*]}"
echo   "  DS methods        : ${DOWNSTREAM_METHODS[*]}"
echo   "  Downstream fusion : $DOWNSTREAM_FUSION  (α=$DOWNSTREAM_FUSION_ALPHA)"
echo   "  Downstream tune   : $_run_ds_tune_flag"
echo   "  Output root       : $OUTPUT_ROOT"
echo "============================================================"


# =============================================================================
# STAGE 0  (hyperparameter tuning — opt-in)
# =============================================================================
if [[ "$_run_tuning_flag" == true ]]; then
    echo ""
    echo "────────────────────────────────────────────────────────────"
    echo "  Stage 0 — Hyperparameter Tuning"
    echo "  Dataset     : $TUNE_DATASET"
    echo "  Images      : $TUNE_N_IMAGES"
    echo "  Methods     : $TUNE_METHODS"
    echo "  Objective   : $TUNE_OBJECTIVE"
    echo "────────────────────────────────────────────────────────────"

    tune_out="${OUTPUT_ROOT}/tuning/uncertainty"

    # shellcheck disable=SC2206
    tune_methods_arr=( $TUNE_METHODS )

    # CrossMethod disagreement requires all four regressors; load them all when requested
    if echo "$TUNE_METHODS" | grep -qw "cross"; then
        # shellcheck disable=SC2206
        _tune_models_arr=( SMIRK DECA EMOCA SHeaP )
    else
        _tune_models_arr=( "$PRIMARY_MODEL" )
    fi

    run python3 main.py \
        --stage          tune \
        "${cpu_flag[@]}" \
        --dataset        "$TUNE_DATASET" \
        --models         "${_tune_models_arr[@]}" \
        --primary_model  "$PRIMARY_MODEL" \
        --tune_n_images  "$TUNE_N_IMAGES" \
        --tune_objective "$TUNE_OBJECTIVE" \
        --tune_methods   "${tune_methods_arr[@]}" \
        --output_dir     "$tune_out"

    echo ""
    echo "  Tuning complete.  JSON → ${tune_out}/hyperparams.json"
    echo "  Review that file and update N_TTA_VALUES / N_JACOBIAN_VALUES above"
    echo "  with the optimal values before running Stages 1-4."
fi

# Exit early when only tuning was requested.
if [[ "$MODE" == "tune_only" ]]; then
    echo ""
    echo "============================================================"
    echo "  Tuning-only run complete."
    echo "  Results in: $OUTPUT_ROOT/tuning"
    echo "============================================================"
    exit 0
fi



# =============================================================================
# STAGES 1–4  (non-downstream)
# =============================================================================
if [[ "$MODE" != "downstream_only" && "$MODE" != "downstream_tune_only" ]]; then

    echo ""
    echo "────────────────────────────────────────────────────────────"
    echo "  Stages 1-4  (${#DATASETS[@]} dataset(s))"
    echo "────────────────────────────────────────────────────────────"

    for dataset in "${DATASETS[@]}"; do
        [[ -n "$FILTER_DATASET" && "$dataset" != "$FILTER_DATASET" ]] && continue

        for ps   in "${PARTITION_SIZES[@]}"; do
        for n_tta in "${N_TTA_VALUES[@]}"; do
        for n_jac in "${N_JACOBIAN_VALUES[@]}"; do
        for n_mcd in "${N_MCD_VALUES[@]}"; do

            n_mr="$N_MAHAL_REF"

            # Shared parameter-sweep suffix for the output directory name.
            _param_suffix=""
            [[ "${#PARTITION_SIZES[@]}"   -gt 1 ]] && _param_suffix+="_ps${ps}"
            [[ "${#N_TTA_VALUES[@]}"      -gt 1 ]] && _param_suffix+="_tta${n_tta}"
            [[ "${#N_JACOBIAN_VALUES[@]}" -gt 1 ]] && _param_suffix+="_jac${n_jac}"
            [[ "${#N_MCD_VALUES[@]}"      -gt 1 ]] && _param_suffix+="_mcd${n_mcd}"
            [[ "${#PARTITION_SIZES[@]}"  -eq 1 ]]  && _param_suffix+="_ps${ps}"

            if [[ "$SWEEP_METHOD_MODEL" == true ]]; then
                # ── Parse SWEEP_METHODS into per-method boolean flags ──────────
                # SWEEP_METHODS may be: "all" | "all_no_dropout" | comma/space list
                _sw_tta=false _sw_jac=false _sw_mah=false
                _sw_mcd=false _sw_sol=false _sw_amcd=false _sw_cross=false

                case "$SWEEP_METHODS" in
                    all)
                        _sw_tta=true; _sw_jac=true; _sw_mah=true
                        _sw_mcd=true; _sw_sol=true; _sw_amcd=true; _sw_cross=true ;;
                    all_no_dropout)
                        _sw_tta=true; _sw_jac=true; _sw_mah=true; _sw_cross=true ;;
                    *)
                        # Comma- or space-separated list of individual method names.
                        # shellcheck disable=SC2206
                        _sm_arr=( ${SWEEP_METHODS//,/ } )
                        for _m in "${_sm_arr[@]}"; do
                            case "$_m" in
                                tta)               _sw_tta=true   ;;
                                jacobian)          _sw_jac=true   ;;
                                mahalanobis)       _sw_mah=true   ;;
                                mcd)               _sw_mcd=true   ;;
                                sol_mcd)           _sw_sol=true   ;;
                                amcd|amcd)        _sw_amcd=true  ;;
                                cross|crossmethod) _sw_cross=true ;;
                            esac
                        done ;;
                esac

                # ── One run per model, ALL applicable methods together ─────────
                # This mirrors the CPU run structure: every method requested for
                # that model is passed in one --methods call so comparison plots
                # (sparsification curves, method tables, heatmap grids) are built.
                # Running methods one-at-a-time produces only single-image outputs
                # because the visualiser needs ≥2 methods to build comparisons.
                for _spm in "${SWEEP_MODELS[@]}"; do
                    # Build the method list for this specific model.
                    _model_methods=()
                    [[ "$_sw_tta"  == true ]] && _model_methods+=( tta )
                    [[ "$_sw_jac"  == true ]] && _model_methods+=( jacobian )
                    [[ "$_sw_mah"  == true ]] && _model_methods+=( mahalanobis )
                    # MCD variants: SMIRK only (retrained checkpoint required).
                    if [[ "$_spm" == "SMIRK" ]]; then
                        [[ "$_sw_mcd"  == true ]] && _model_methods+=( mcd )
                        [[ "$_sw_sol"  == true ]] && _model_methods+=( sol_mcd )
                        [[ "$_sw_amcd" == true ]] && _model_methods+=( amcd )
                    fi

                    # Skip this model if no methods are enabled.
                    [[ "${#_model_methods[@]}" -eq 0 ]] && continue

                    tag="${dataset}_${_spm}${_param_suffix}"
                    run python3 main.py \
                        --stage          no_downstream \
                        "${cpu_flag[@]}" \
                        --dataset        "$dataset" \
                        --models         "$_spm" \
                        --methods        "${_model_methods[@]}" \
                        --partition_size "$ps" \
                        --n_tta          "$n_tta" \
                        --n_mcd          "$n_mcd" \
                        --n_jacobian     "$n_jac" \
                        --n_mahal_ref    "$n_mr" \
                        --primary_model  "$_spm" \
                        "${img_args[@]}" \
                        --output_dir     "${OUTPUT_ROOT}/${tag}"
                done

                # ── CrossMethod: all SWEEP_MODELS simultaneously ──────────────
                # Requires ≥2 models; skip silently when only 1 is selected.
                # Cross is always its own run — it needs all models loaded at once.
                if [[ "$_sw_cross" == true && "${#SWEEP_MODELS[@]}" -ge 2 ]]; then
                    _cross_tag="${dataset}_cross${_param_suffix}"
                    run python3 main.py \
                        --stage          no_downstream \
                        "${cpu_flag[@]}" \
                        --dataset        "$dataset" \
                        --models         "${SWEEP_MODELS[@]}" \
                        --methods        cross \
                        --partition_size "$ps" \
                        --n_tta          "$n_tta" \
                        --n_mcd          "$n_mcd" \
                        --n_jacobian     "$n_jac" \
                        --n_mahal_ref    "$n_mr" \
                        --primary_model  "${SWEEP_MODELS[0]}" \
                        "${img_args[@]}" \
                        --output_dir     "${OUTPUT_ROOT}/${_cross_tag}"
                fi

            else
                # ── Standard single-config run ────────────────────────────────
                tag="${dataset}${_param_suffix}"
                run python3 main.py \
                    --stage          no_downstream \
                    "${cpu_flag[@]}" \
                    --dataset        "$dataset" \
                    --models         "${MODELS[@]}" \
                    --methods        "$METHODS" \
                    --partition_size "$ps" \
                    --n_tta          "$n_tta" \
                    --n_mcd          "$n_mcd" \
                    --n_jacobian     "$n_jac" \
                    --n_mahal_ref    "$n_mr" \
                    --primary_model  "$PRIMARY_MODEL" \
                    "${img_args[@]}" \
                    --output_dir     "${OUTPUT_ROOT}/${tag}"
            fi

        done; done; done; done  # n_mcd / n_jac / n_tta / ps
    done  # dataset
fi


# =============================================================================
# STAGE 5  (downstream classifier)
# =============================================================================
if [[ "$MODE" != "no_downstream" && "$MODE" != "downstream_tune_only" && "$RUN_DOWNSTREAM" == true ]]; then

    echo ""
    echo "────────────────────────────────────────────────────────────"
    echo "  Stage 5 — downstream classifier  ($n_ds_runs run(s))"
    echo "────────────────────────────────────────────────────────────"

    for subset   in "${DOWNSTREAM_SUBSETS[@]}";      do
    for epochs   in "${DOWNSTREAM_EPOCHS[@]}";        do
    for lr       in "${DOWNSTREAM_LR[@]}";            do
    for ds_n_tta in "${DOWNSTREAM_N_TTA_VALUES[@]}";  do
    for ds_method in "${DOWNSTREAM_METHODS[@]}";      do

        # Output directory name.
        tag="downstream_${subset}cls"
        [[ "${#DOWNSTREAM_EPOCHS[@]}"       -gt 1 ]] && tag+="_${epochs}ep"
        [[ "${#DOWNSTREAM_LR[@]}"           -gt 1 ]] && tag+="_lr${lr}"
        [[ "${#DOWNSTREAM_N_TTA_VALUES[@]}" -gt 1 ]] && tag+="_dtta${ds_n_tta}"
        [[ "${#DOWNSTREAM_METHODS[@]}"      -gt 1 ]] && tag+="_${ds_method}"
        # Always include epochs when only one value (informative).
        [[ "${#DOWNSTREAM_EPOCHS[@]}" -eq 1 ]]       && tag+="_${epochs}ep"

        run python3 main.py \
            --stage               downstream \
            "${cpu_flag[@]}" \
            --models              "${DOWNSTREAM_MODELS[@]}" \
            --methods             all_no_dropout \
            --downstream_methods  "$ds_method" \
            --downstream_train_subset "$subset" \
            --downstream_test_subset  0 \
            --downstream_epochs   "$epochs" \
            --downstream_lr       "$lr" \
            --downstream_n_tta    "$ds_n_tta" \
            --downstream_backbone "$DOWNSTREAM_BACKBONE" \
            --downstream_batch_size "$DOWNSTREAM_BATCH_SIZE" \
            --downstream_fusion   "$DOWNSTREAM_FUSION" \
            --downstream_fusion_alpha "$DOWNSTREAM_FUSION_ALPHA" \
            --downstream_weight_decay "$DOWNSTREAM_WEIGHT_DECAY" \
            --downstream_patience "$DOWNSTREAM_PATIENCE" \
            --downstream_unfreeze_blocks "$DOWNSTREAM_UNFREEZE_BLOCKS" \
            --downstream_mixup_alpha "$DOWNSTREAM_MIXUP_ALPHA" \
            --downstream_curriculum \
            --downstream_curriculum_start "$DOWNSTREAM_CURRICULUM_START" \
            --raf_db_root         "$EMOTION_DB_ROOT" \
            --maps_cache_dir      "$MAPS_CACHE_DIR" \
            --output_dir          "${OUTPUT_ROOT}/${tag}" \
            "${force_args[@]}"

    done; done; done; done; done
fi


# =============================================================================
# STAGE 6  (downstream hyperparameter tuning — opt-in)
# =============================================================================
if [[ "$_run_ds_tune_flag" == true ]]; then

    echo ""
    echo "────────────────────────────────────────────────────────────"
    echo "  Stage 6 — Downstream Hyperparameter Tuning"
    echo "  Output dir   : $DOWNSTREAM_TUNE_OUTPUT_DIR"
    if [[ -n "$DOWNSTREAM_TUNE_MAPS_DIR" ]]; then
        echo "  Maps dir     : $DOWNSTREAM_TUNE_MAPS_DIR"
    else
        echo "  Maps dir     : auto-discover from $DOWNSTREAM_TUNE_OUTPUT_DIR/downstream/maps/"
    fi
    printf  "  Device       : %s\n" "$DEVICE_MODE"
    echo "────────────────────────────────────────────────────────────"

    # Build optional flag arrays.
    _dt_seeds_args=()
    [[ -n "$DOWNSTREAM_TUNE_N_SEEDS" ]] \
        && _dt_seeds_args=( --downstream_tune_n_seeds "$DOWNSTREAM_TUNE_N_SEEDS" )

    _dt_epochs_args=()
    [[ -n "$DOWNSTREAM_TUNE_MAX_EPOCHS" ]] \
        && _dt_epochs_args=( --downstream_tune_max_epochs "$DOWNSTREAM_TUNE_MAX_EPOCHS" )

    _dt_patience_args=()
    [[ -n "$DOWNSTREAM_TUNE_PATIENCE" ]] \
        && _dt_patience_args=( --downstream_tune_patience "$DOWNSTREAM_TUNE_PATIENCE" )

    _dt_maps_args=()
    [[ -n "$DOWNSTREAM_TUNE_MAPS_DIR" ]] \
        && _dt_maps_args=( --downstream_maps_dir "$DOWNSTREAM_TUNE_MAPS_DIR" )

    _dt_ntrials_args=()
    [[ -n "$DOWNSTREAM_TUNE_N_TRIALS" ]] \
        && _dt_ntrials_args=( --downstream_tune_n_trials "$DOWNSTREAM_TUNE_N_TRIALS" )

    _dt_search_args=()
    [[ -n "$DOWNSTREAM_TUNE_SEARCH" ]] \
        && _dt_search_args=( --downstream_tune_search "$DOWNSTREAM_TUNE_SEARCH" )

    run python3 main.py \
        --stage               downstream_tune \
        "${cpu_flag[@]}" \
        --raf_db_root         "${_DEF_DS_MODELS:+./datasets/raf-db}" \
        --downstream_backbone "$DOWNSTREAM_BACKBONE" \
        --downstream_subset   "${DOWNSTREAM_SUBSETS[0]}" \
        --output_dir          "$DOWNSTREAM_TUNE_OUTPUT_DIR" \
        "${_dt_maps_args[@]}" \
        "${_dt_seeds_args[@]}" \
        "${_dt_epochs_args[@]}" \
        "${_dt_patience_args[@]}" \
        "${_dt_ntrials_args[@]}" \
        "${_dt_search_args[@]}"

    echo ""
    echo "  Downstream tuning complete."
    echo "  Results → ${DOWNSTREAM_TUNE_OUTPUT_DIR}/results.json"

    if [[ "$MODE" == "downstream_tune_only" ]]; then
        echo ""
        echo "============================================================"
        echo "  Downstream-tune-only run complete."
        echo "============================================================"
        exit 0
    fi
fi


echo ""
echo "============================================================"
echo "  All experiments complete."
echo "  Results in: $OUTPUT_ROOT"
echo "============================================================"
