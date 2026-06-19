jid1=$(sbatch train.sbatch | awk '{print $4}') 
jid2=$(sbatch --dependency=afterok:$jid1 train.sbatch | awk '{print $4}') 
jid3=$(sbatch --dependency=afterok:$jid2 train.sbatch | awk '{print $4}') 
jid4=$(sbatch --dependency=afterok:$jid3 train.sbatch | awk '{print $4}') 
echo "$jid1 -> $jid2 -> $jid3 -> $jid4"


#93221 -> 93222 -> 93223 -> 93224