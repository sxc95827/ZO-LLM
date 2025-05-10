# /media/SSD5/personal/wzy/models/opt-13b/
# /media/SSD5/personal/may_the_force_be_with_you/huggingface_model/meta-llama/Meta-Llama-3-8B-Instruct/
# /media/SSD5/personal/wzy/models/opt-1.3b/
# /media/SSD5/personal/may_the_force_be_with_you/huggingface_model/Mamba2InLlama_1/

#python run.py --lora \
#--model_name=/media/SSD5/personal/may_the_force_be_with_you/huggingface_model/Mamba2InLlama_1/  \
#--task_name=SST2 \
#--output_dir=/media/SSD7/personal/wangziyi/SST2-lora-$TAG \
#--num_train_epochs=5 \
#--per_device_train_batch_size=16 \
#--load_best_model_at_end \
#--evaluation_strategy=steps \
#--save_strategy=steps \
#--save_total_limit=1 \
#--max_steps=20000 \
#--logging_steps=10 \
#--num_eval=500 \
#--num_train=1000 \
#--num_dev=100 \
#--train_as_classification=True \
#--perturbation_mode=two_side \
#--trainer=zo_sign_opt \
#--train_set_seed=0 \
#--lr_scheduler_type=constant \
#--eval_steps=500 \
#--save_steps=500 \
#--learning_rate=0.001 \
#--zo_eps=0.01 \
#--load_float16

export CUDA_VISIBLE_DEVICES=0
export WANDB_MODE=disabled
python run.py --lora \
--model_name=/media/SSD6/personal/wzy/models/opt-1.3b/  \
--task_name=SST2 \
--output_dir=/media/SSD6/personal/wzy/code/ZO-LLM-Norman/zo-bench/SST2-lora-$TAG \
--num_train_epochs=5 \
--per_device_train_batch_size=16 \
--load_best_model_at_end \
--evaluation_strategy=steps \
--save_strategy=steps \
--save_total_limit=1 \
--max_steps=20000 \
--logging_steps=10 \
--num_eval=1000 \
--num_train=1000 \
--num_dev=500 \
--train_as_classification=True \
--perturbation_mode=two_side \
--trainer=zo_sgd \
--train_set_seed=0 \
--lr_scheduler_type=constant \
--eval_steps=500 \
--save_steps=500 \
--learning_rate=0.0001 \
--zo_eps=0.01 \
--load_float16 \
--save_model






# python run.py --lora \
# --model_name=/media/SSD5/personal/may_the_force_be_with_you/huggingface_model/Mamba2InLlama_1/ \
# --task_name=Copa \
# --output_dir=/media/SSD7/personal/wangziyi/SST2-lora-$TAG \
# --num_train_epochs=5 \
# --no_reparam \
# --per_device_train_batch_size=16 \
# --load_best_model_at_end \
# --evaluation_strategy=steps \
# --save_strategy=steps \
# --save_total_limit=1 \
# --max_steps=20000 \
# --logging_steps=10 \
# --num_eval=1000 \
# --num_train=1000 \
# --num_dev=100 \
# --train_as_classification=False \
# --perturbation_mode=two_side \
# --trainer=zo_sgd \
# --train_set_seed=0 \
# --lr_scheduler_type=constant \
# --eval_steps=100 \
# --save_steps=500 \
# --learning_rate=0.0001 \
# --zo_eps=0.01 \
# --weight_decay=0 \
# --load_float16

# python run_s.py --lora \
# --model_name=/media/SSD5/personal/may_the_force_be_with_you/huggingface_model/Mamba2InLlama_1/  \
# --task_name=WinoGrande \
# --output_dir=/media/SSD7/personal/wangziyi/SST2-lora-$TAG \
# --num_train_epochs=5 \
# --no_reparam \
# --per_device_train_batch_size=16 \
# --load_best_model_at_end \
# --evaluation_strategy=steps \
# --save_strategy=steps \
# --save_total_limit=1 \
# --max_steps=20000 \
# --logging_steps=10 \
# --num_eval=1000 \
# --num_train=1000 \
# --num_dev=100 \
# --train_as_classification=False \
# --perturbation_mode=two_side \
# --trainer=zo_sign_opt \
# --train_set_seed=0 \
# --lr_scheduler_type=constant \
# --eval_steps=100 \
# --save_steps=10000 \
# --learning_rate=0.0001 \
# --zo_eps=0.01 \
# --weight_decay=0 \
# --load_float16
