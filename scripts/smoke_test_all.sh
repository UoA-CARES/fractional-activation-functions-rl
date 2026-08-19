#!/usr/bin/env bash

set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

OUT="smoke_test_all"
LOG="$OUT/smoke_test.log"

rm -rf "$OUT"
mkdir -p "$OUT"

PASS=0
FAIL=0

run_test () {
    NAME="$1"
    shift

    echo
    echo "============================================================"
    echo "TEST: $NAME"
    echo "============================================================"

    if "$@" 2>&1 | tee -a "$LOG"; then
        echo "PASS: $NAME"
        PASS=$((PASS + 1))
    else
        echo "FAIL: $NAME"
        FAIL=$((FAIL + 1))
    fi
}


echo "============================================================" | tee "$LOG"
echo " Fractional-RL Complete Smoke Test" | tee -a "$LOG"
echo "============================================================" | tee -a "$LOG"


# ------------------------------------------------------------
# 1. Unit / integration tests
# ------------------------------------------------------------

run_test \
    "pytest full suite" \
    pytest -q


# ------------------------------------------------------------
# 2. MuJoCo baseline tests
# ------------------------------------------------------------

for ALGO in TD3 SAC
do
    run_test \
        "$ALGO + ReLU + HalfCheetah-v4" \
        python scripts/train.py \
            --algo "$ALGO" \
            --env HalfCheetah-v4 \
            --activation ReLU \
            --layers 2 \
            --placement all_both \
            --seed 10 \
            --steps 20 \
            --start-steps 5 \
            --batch-size 4 \
            --buffer-size 100 \
            --train-every 1 \
            --gradient-steps 1 \
            --eval-every 20 \
            --eval-episodes 1 \
            --output "$OUT"
done


# ------------------------------------------------------------
# 3. Every official fractional activation × TD3/SAC
# ------------------------------------------------------------

ACTIVATIONS=(
    FReLU
    FLReLU
    FPReLU
    FSwish
    FGELU
)

for ALGO in TD3 SAC
do
    for ACT in "${ACTIVATIONS[@]}"
    do
        run_test \
            "$ALGO + $ACT + HalfCheetah-v4" \
            python scripts/train.py \
                --algo "$ALGO" \
                --env HalfCheetah-v4 \
                --activation "$ACT" \
                --alpha 0.2 \
                --layers 2 \
                --placement all_both \
                --seed 10 \
                --steps 20 \
                --start-steps 5 \
                --batch-size 4 \
                --buffer-size 100 \
                --train-every 1 \
                --gradient-steps 1 \
                --eval-every 20 \
                --eval-episodes 1 \
                --output "$OUT"
    done
done


# ------------------------------------------------------------
# 4. Placement logic
# ------------------------------------------------------------

PLACEMENTS=(
    all_both
    all_actor
    all_critic
    first_both
    first_actor
    first_critic
)

for PLACE in "${PLACEMENTS[@]}"
do
    run_test \
        "TD3 + FReLU + placement=$PLACE" \
        python scripts/train.py \
            --algo TD3 \
            --env HalfCheetah-v4 \
            --activation FReLU \
            --alpha 0.2 \
            --layers 2 \
            --placement "$PLACE" \
            --seed 10 \
            --steps 12 \
            --start-steps 5 \
            --batch-size 4 \
            --buffer-size 100 \
            --train-every 1 \
            --gradient-steps 1 \
            --eval-every 12 \
            --eval-episodes 1 \
            --output "$OUT"
done


# ------------------------------------------------------------
# 5. One-hidden-layer configuration
# ------------------------------------------------------------

run_test \
    "TD3 + FReLU + 1 layer" \
    python scripts/train.py \
        --algo TD3 \
        --env HalfCheetah-v4 \
        --activation FReLU \
        --alpha 0.2 \
        --layers 1 \
        --placement all_both \
        --seed 10 \
        --steps 12 \
        --start-steps 5 \
        --batch-size 4 \
        --buffer-size 100 \
        --train-every 1 \
        --gradient-steps 1 \
        --eval-every 12 \
        --eval-episodes 1 \
        --output "$OUT"


# ------------------------------------------------------------
# 6. DMC environment family
# ------------------------------------------------------------

run_test \
    "DMC TD3 + FReLU + walker-walk" \
    python scripts/train.py \
        --algo TD3 \
        --env walker-walk \
        --activation FReLU \
        --alpha 0.2 \
        --layers 2 \
        --placement all_both \
        --seed 10 \
        --steps 20 \
        --start-steps 5 \
        --batch-size 4 \
        --buffer-size 100 \
        --train-every 1 \
        --gradient-steps 1 \
        --eval-every 20 \
        --eval-episodes 1 \
        --output "$OUT"


run_test \
    "DMC SAC + FGELU + walker-walk" \
    python scripts/train.py \
        --algo SAC \
        --env walker-walk \
        --activation FGELU \
        --alpha 0.2 \
        --layers 2 \
        --placement all_both \
        --seed 10 \
        --steps 20 \
        --start-steps 5 \
        --batch-size 4 \
        --buffer-size 100 \
        --train-every 1 \
        --gradient-steps 1 \
        --eval-every 20 \
        --eval-episodes 1 \
        --output "$OUT"


# ------------------------------------------------------------
# 7. Verify generated artifacts
# ------------------------------------------------------------

echo
echo "============================================================"
echo "GENERATED ARTIFACT CHECK"
echo "============================================================"

CONFIGS=$(find "$OUT" -name config.json | wc -l)
TRAINS=$(find "$OUT" -name train.csv | wc -l)
EVALS=$(find "$OUT" -name eval.csv | wc -l)
MODELS=$(find "$OUT" -name model_checkpoint.pth | wc -l)

echo "config.json files : $CONFIGS"
echo "train.csv files   : $TRAINS"
echo "eval.csv files    : $EVALS"
echo "checkpoints       : $MODELS"


echo
echo "============================================================"
echo "FINAL RESULT"
echo "============================================================"
echo "PASSED: $PASS"
echo "FAILED: $FAIL"
echo "LOG:    $LOG"
echo "============================================================"

if [ "$FAIL" -eq 0 ]; then
    echo "ALL FRACTIONAL-RL SMOKE TESTS PASSED"
    exit 0
else
    echo "ONE OR MORE TESTS FAILED"
    exit 1
fi
