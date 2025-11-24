#!/bin/bash
# ==========================================
# 출력 스타일 설정 (색상 및 구분선)
# ==========================================
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# 헤더 출력 함수
print_header() {
    echo -e "\n${BLUE}############################################################${NC}"
    echo -e "${YELLOW}👉 Running Experiment: ${BOLD}$1${NC}"
    echo -e "${BLUE}############################################################${NC}\n"
}



# ==========================================
# 실험 시작
# ==========================================

# 0. Linear Regression Baseline
print_header "0. Linear Regression Baseline"
python linear_regression.py

# 1. No Covariates, No Cross Learning
print_header "1. No Covariates, No Cross Learning"
python chronos_run.py --use_chronos

# 2. With Covariates, No Cross Learning
print_header "2. With Covariates, No Cross Learning"
python chronos_run.py --use_chronos --use_covariates

# 3. With Covariates, With Cross Learning
print_header "3. With Covariates, With Cross Learning"
python chronos_run.py --use_chronos --use_covariates --use_cross_learning

# 4. Fine-tuning (No Cross Learning)
print_header "4. Fine-tuning - No Cross Learning"
python chronos_run.py --use_chronos --fine_tune --use_covariates

# 5. Fine-tuning (With Cross Learning)
print_header "5. Fine-tuning - With Cross Learning"
python chronos_run.py --use_chronos --fine_tune --use_covariates --use_cross_learning

# 6. Continual Pretrained Model
print_header "6. Continual Pretrained Model + Fine-tuning"
python chronos_run.py --use_chronos --use_covariates --continual_pretrain --fine_tune

# 종료 메시지
echo -e "\n${GREEN}${BOLD}✅ All experiments completed successfully!${NC}\n"