target_domain=${1}
experiment=${2}

python /content/Activation-Shaping-AML/main.py \
--experiment=${experiment} \
--experiment_name=${experiment}/${target_domain}/ \
--dataset_args="{'root': '/content/Activation-Shaping-AML/data/PACS', 'source_domain': 'art_painting', 'target_domain': '${target_domain}'}" \
--batch_size=128 \
--num_workers=5 \
--grad_accum_steps=1