dataset_name='coco' # 'aokvqa' 'gqa'
image_folder='../data/val2014' # '/workspace/data/gqa'
for seed in 1 2 3
do
for type in 'adversarial'
do
python run_llava.py \
--question-file ../data/POPE/${dataset_name}/${dataset_name}_pope_${type}.json \
--image-folder ${image_folder} \
--answers-file ../output/llava_${dataset_name}_pope_${type}_answers_agla_seed${seed}.jsonl \
--use_agla \
--alpha 2 \
--beta 0.5 \
--seed $seed

python eval_pope.py --gt_files ../data/POPE/coco/coco_pope_${type}.json --gen_files ../output/llava_${dataset_name}_pope_${type}_answers_agla_seed${seed}.jsonl

done
done
