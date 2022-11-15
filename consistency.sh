date="2022-11-08"
# date="2022-11-10"
decoding_strategy="top10_top0.9_T0.9"

exp_name='baseline'
# exp_name='baseline_shuffle'
# exp_name='baseline_wo_typeId'
# exp_name='baseline_wo_typeId_all_seq_loss'

# exp_name='joint_decoding'
# exp_name='joint_decoding_w_groundtruth_label'

for order in normal_ord pos_ord neg_ord lex_pos_ord lex_neg_ord pos_maj3 pos_maj10 neg_maj3 neg_maj10 
do
echo $order
ref=/misc/kfdata01/kf_grp/lchen/ACL23/Nov/decoding_results/debug_${exp_name}_GP2-pretrain-step-7000_${order}_${decoding_strategy}_${date}_pred_response
python cal_consistency_score.py $ref ${exp_name}_${order}_${decoding_strategy}
done