# case 1
model_id="meta-llama/Llama-3.2-1B-Instruct"
model_ref_id="Llama-3.2-1B-Instruct" # for tee.
device_names=("Tesla_V100-SXM2-32GB")  
device_numbers=(1)  # define device numbers as a list of integers

prompt_length=128
output_token=200
seed=200
gb_size=32
theta=100 # how much concerned about quality
GROUPSIZE=1

llmpq-algo --model_id "meta-llama/Llama-3.2-1B-Instruct" \
 --device_names "${device_names[@]}" \
 --device_numbers "${device_numbers[@]}" \
 --ilp_seed $seed \
 --fit --s $prompt_length --n $output_token \
 --theta $theta --global_bz $gb_size --ilp_time_limit 60 --group_size $GROUPSIZE --fit --debug \
 --fname-suffix "group_$GROUP_SIZE" 2>&1 | tee "${ABLATION_FOLDER}${model_ref_id}_GROUPSIZE${GROUPSIZE}"