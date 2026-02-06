python -m domainbed.scripts.sweep delete_incomplete \
       --data_dir=/data/luhongtao/dataset \
       --output_dir=./output/pacs1/results \
       --command_launcher local \
       --algorithms ERM \
       --datasets PACS \
       --n_hparams 5\
       --n_trials 1