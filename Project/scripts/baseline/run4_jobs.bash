jid1=$(sbatch train.sbatch | awk '{print $4}')
jid2=$(sbatch --dependency=afterany:$jid1 train.sbatch | awk '{print $4}')
jid3=$(sbatch --dependency=afterany:$jid2 train.sbatch | awk '{print $4}')
jid4=$(sbatch --dependency=afterany:$jid3 train.sbatch | awk '{print $4}')
jid5=$(sbatch --dependency=afterany:$jid4 train.sbatch | awk '{print $4}')
jid6=$(sbatch --dependency=afterany:$jid5 train.sbatch | awk '{print $4}')
jid7=$(sbatch --dependency=afterany:$jid6 train.sbatch | awk '{print $4}')
jid8=$(sbatch --dependency=afterany:$jid7 train.sbatch | awk '{print $4}')
jid9=$(sbatch --dependency=afterany:$jid8 train.sbatch | awk '{print $4}')

echo "$jid1 -> $jid2 -> $jid3 -> $jid4 -> $jid5 -> $jid6 -> $jid7 -> $jid8 -> $jid9"

#93290 -> 93291 -> 93292 -> 93293 -> 93294 -> 93295 -> 93296 -> 93297 -> 93298