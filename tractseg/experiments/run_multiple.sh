#!/bin/bash
set -e  # stop on error

#sbatch --job-name=TractSeg20_888_DiceLoss_Adam_lr5 ~/runner.sh
#sbatch --job-name=TractSeg20_888_DiceLoss_Adamax ~/runner.sh
#sbatch --job-name=TractSeg20_888_SampDiceLoss_Adam ~/runner.sh

#sbatch --job-name=EndingsSeg_270g_125mm ~/runner.sh
#sbatch --job-name=EndingsSeg_12g90g270g_125mm_DAugAll ~/runner.sh
#sbatch --job-name=TractSeg_12g90g270g_125mm_DAugAll ~/runner.sh
#sbatch --job-name=TractSeg_12g90g270g_125mm_DAugAll_DiceLoss ~/runner.sh


