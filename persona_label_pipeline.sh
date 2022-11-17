gpu=1
bz=512
threshold=False

for split in train valid test
do
    echo $split
    # echo 'get persona label...'
    # python cal_persona_label.py ../../datasets/${split}_self_original.txt predictions-11-16/${split}_entailment_scores.bin $bz $gpu
    echo 'get labeled data...'
    # python get_persona_labeled_dataset.py ../../datasets/${split}_self_original.txt predictions/${split}/entailment_scores.bin predictions/${split}/output-wo-th
    python get_persona_labeled_dataset.py ../../datasets/${split}_self_original.txt predictions/${split}/entailment_scores.bin predictions/${split}/output-reverse-th10 $threshold
    echo ''
done