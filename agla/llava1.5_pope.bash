dataset_name='coco' # 'aokvqa' 'gqa'
image_folder='../data/val2014' # '/workspace/data/gqa'
for seed in 42424
for alpha in .5 1. 1.5 2.
do
for beta in .2 .4 .6
do
for type in 'random'
do
python run_llava.py \
--question-file ../data/POPE/${dataset_name}/${dataset_name}_pope_${type}.json \
--image-folder ${image_folder} \
--answers-file ../output/llava_${dataset_name}_pope_${type}_answers_agla_seed${seed}_alpha_${alpha}_beta_${beta}.jsonl \
--use_agla \
--alpha $alpha \
--beta $beta \
--seed $seed

python eval_pope.py --gt_files ../data/POPE/coco/coco_pope_random.json --gen_files ../output/llava_${dataset_name}_pope_${type}_answers_agla_seed${seed}_alpha_${alpha}_beta_${beta}.jsonl

done
done
done
done
