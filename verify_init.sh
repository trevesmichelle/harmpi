#!/bin/bash
# verify_init.sh — pre-submit sanity check for HARMPI torus config
# Validates decs.h + init.c + run_harmpi.sh against Chashkina et al. 2021 fiducial.
# Usage:  ./verify_init.sh   (run from harmpi/ directory)
# Exit:   0 = pass (or warnings only), 1 = at least one failure, 2 = setup error

set -u

GREEN='\033[0;32m'; RED='\033[0;31m'; YELLOW='\033[0;33m'
BOLD='\033[1m';    NC='\033[0m'

PASS=0; FAIL=0; WARN=0

check_eq() {
    local label="$1" actual="$2" expected="$3"
    if [[ "$actual" == "$expected" ]]; then
        printf "  ${GREEN}PASS${NC}  %-30s = %s\n" "$label" "$actual"
        PASS=$((PASS+1))
    else
        printf "  ${RED}FAIL${NC}  %-30s = %s  ${RED}(expected: %s)${NC}\n" "$label" "$actual" "$expected"
        FAIL=$((FAIL+1))
    fi
}

check_one_of() {
    local label="$1" actual="$2"; shift 2
    local opts=("$@")
    for opt in "${opts[@]}"; do
        if [[ "$actual" == "$opt" ]]; then
            printf "  ${GREEN}PASS${NC}  %-30s = %s\n" "$label" "$actual"
            PASS=$((PASS+1)); return
        fi
    done
    printf "  ${RED}FAIL${NC}  %-30s = %s  ${RED}(expected: %s)${NC}\n" "$label" "$actual" "${opts[*]}"
    FAIL=$((FAIL+1))
}

warn_msg() {
    printf "  ${YELLOW}WARN${NC}  %-30s = %s  ${YELLOW}(%s)${NC}\n" "$1" "$2" "$3"
    WARN=$((WARN+1))
}

info_msg() {
    printf "  ${BOLD}info${NC}  %-30s = %s\n" "$1" "$2"
}

# Required files
for f in decs.h init.c run_harmpi.sh; do
    if [[ ! -f "$f" ]]; then
        printf "${RED}ERROR: $f not found. Run from harmpi/ directory.${NC}\n"
        exit 2
    fi
done

# Extract "var = value" RHS from a line range (whitespace stripped)
extract() {
    local file="$1" var="$2" lo="$3" hi="$4"
    sed -n "${lo},${hi}p" "$file" \
        | grep -E "^\s*${var}\s*=" \
        | head -1 \
        | sed -E "s|^\s*${var}\s*=\s*||; s|\s*;.*$||; s/\s+//g"
}

# Get the raw assignment line (for display + regex on its content)
extract_line() {
    local file="$1" var="$2" lo="$3" hi="$4"
    sed -n "${lo},${hi}p" "$file" \
        | grep -E "^\s*${var}\s*=" \
        | head -1 \
        | sed -E 's|^\s*||; s|\s*//.*$||; s|\s*;\s*$||'
}

printf "\n${BOLD}=== HARMPI Torus Config Verification ===${NC}\n"
printf "Reference: Chashkina et al. 2021 (arXiv:2106.15738), 2D fiducial run\n\n"

# -------------------------------------------------------
# decs.h: problem selection + per-tile resolution
# -------------------------------------------------------
printf "${BOLD}[decs.h: problem & tile resolution]${NC}\n"

WP=$(grep -E "^\s*#define\s+WHICHPROBLEM\s" decs.h | awk '{print $3}')
check_eq "WHICHPROBLEM" "$WP" "TORUS_PROBLEM"

TLO=$(grep -n "WHICHPROBLEM\s*==\s*TORUS_PROBLEM" decs.h | head -1 | cut -d: -f1)
if [[ -z "$TLO" ]]; then
    printf "  ${RED}ERROR: cannot locate TORUS_PROBLEM #elif block in decs.h${NC}\n"
    exit 2
fi
THI=$((TLO+15))

read_define() {
    sed -n "${TLO},${THI}p" decs.h \
        | grep -E "^\s*#define\s+$1\s" | head -1 \
        | awk '{print $3}' | tr -d '()'
}

N1=$(read_define N1); N2=$(read_define N2); N3=$(read_define N3)
info_msg "N1 (per-tile)" "$N1"
info_msg "N2 (per-tile)" "$N2"
info_msg "N3 (per-tile)" "$N3"

# -------------------------------------------------------
# run_harmpi.sh: tile decomposition + MPI cores
# -------------------------------------------------------
printf "\n${BOLD}[run_harmpi.sh: tile decomposition]${NC}\n"

MPI_LINE=$(grep -E "mpirun.*\./harm" run_harmpi.sh | head -1)
NP=$(echo "$MPI_LINE" | sed -E 's/.*-np\s+([0-9]+).*/\1/')
TX=$(echo "$MPI_LINE" | sed -E 's|.*\./harm\s+([0-9]+)\s+([0-9]+)\s+([0-9]+).*|\1|')
TY=$(echo "$MPI_LINE" | sed -E 's|.*\./harm\s+([0-9]+)\s+([0-9]+)\s+([0-9]+).*|\2|')
TZ=$(echo "$MPI_LINE" | sed -E 's|.*\./harm\s+([0-9]+)\s+([0-9]+)\s+([0-9]+).*|\3|')

info_msg "MPI processes (-np)" "$NP"
info_msg "Tile decomposition" "${TX} x ${TY} x ${TZ}"

PROD=$((TX*TY*TZ))
check_eq "Tiles * decomp == -np" "$PROD" "$NP"

# Global resolution = per-tile x decomposition (this is the actual physics grid)
GN1=$((N1*TX)); GN2=$((N2*TY)); GN3=$((N3*TZ))
printf "\n${BOLD}[Global resolution = per-tile x decomposition]${NC}\n"
check_eq "Global N1" "$GN1" "512"
check_eq "Global N2" "$GN2" "512"
check_eq "Global N3" "$GN3" "1"

# -------------------------------------------------------
# init.c: TORUS_PROBLEM init() block (lines 1-300; other problems live below 670)
# -------------------------------------------------------
LO=1; HI=300
printf "\n${BOLD}[init.c — TORUS init() block, lines ${LO}-${HI}]${NC}\n"

A=$(extract    init.c "a"    $LO $HI)
RIN=$(extract  init.c "rin"  $LO $HI)
RMAX=$(extract init.c "rmax" $LO $HI)
GAM=$(extract  init.c "gam"  $LO $HI)
BETA=$(extract init.c "beta" $LO $HI)
R0=$(extract   init.c "R0"   $LO $HI)
ROUT=$(extract init.c "Rout" $LO $HI)
TF=$(extract   init.c "tf"   $LO $HI)
DTD=$(extract  init.c "DTd"  $LO $HI)
DTR=$(extract  init.c "DTr"  $LO $HI)

# Physics
check_eq     "a (spin)"             "$A"    "0.9"
check_eq     "rin (torus inner)"    "$RIN"  "15."
check_eq     "rmax (pressure max)"  "$RMAX" "32."
check_eq     "gam (adiabatic)"      "$GAM"  "5./3."
check_eq     "beta_min (B-field)"   "$BETA" "100."

# Grid
check_eq     "R0 (grid offset)"     "$R0"   "0.0"
check_one_of "Rout (outer bound.)"  "$ROUT" "1e5" "1.e5" "1E5" "100000." "100000.0"

# Rin should be inside horizon: pattern like 0.8x*(1+sqrt(1-a*a))
RIN_GRID_LINE=$(extract_line init.c "Rin" $LO $HI)
if echo "$RIN_GRID_LINE" | grep -qE "0\.[89][0-9]*\*\(1\..*sqrt"; then
    printf "  ${GREEN}PASS${NC}  %-30s : %s\n" "Rin (grid, inside horizon)" "$RIN_GRID_LINE"
    PASS=$((PASS+1))
else
    warn_msg "Rin (grid)" "$RIN_GRID_LINE" "expected form: 0.87*(1+sqrt(1-a*a))"
fi

# Run length: paper runs much longer, but 500 is fine for a first physics check
if [[ "$TF" == "500.0" || "$TF" == "500." ]]; then
    warn_msg "tf (final time)" "$TF" "OK for first run; paper goes much longer"
else
    info_msg "tf (final time)" "$TF"
fi
info_msg "DTd (dump cadence)"    "$DTD M"
info_msg "DTr (restart cadence)" "$DTR M"

# -------------------------------------------------------
# Summary
# -------------------------------------------------------
printf "\n${BOLD}=== Summary ===${NC}\n"
printf "  ${GREEN}Pass: %d${NC}   ${YELLOW}Warn: %d${NC}   ${RED}Fail: %d${NC}\n" "$PASS" "$WARN" "$FAIL"

if [[ $FAIL -gt 0 ]]; then
    printf "\n${RED}${BOLD}DO NOT SUBMIT — fix failures first.${NC}\n\n"
    exit 1
elif [[ $WARN -gt 0 ]]; then
    printf "\n${YELLOW}${BOLD}Warnings only — review, then:${NC} make clean && make && qsub run_harmpi.sh\n\n"
    exit 0
else
    printf "\n${GREEN}${BOLD}All checks passed.${NC} Safe to: make clean && make && qsub run_harmpi.sh\n\n"
    exit 0
fi
