# Domain shift
CUDA_VISIBLE_DEVICES=1 python main.py /data/luhongtao/dataset -d PACS           --log exp0 --task domain_shift --targets 0 -b 12 --lr 5e-6 --epochs 10 --lam 0.3 --beta 0.5
CUDA_VISIBLE_DEVICES=1 python main.py /data/luhongtao/dataset -d PACS           --log exp0_1 --task domain_shift --targets 1 -b 12 --lr 5e-6 --epochs 10 --lam 0.3 --beta 0.5
CUDA_VISIBLE_DEVICES=1 python main.py /data/luhongtao/dataset -d PACS           --log exp0_2 --task domain_shift --targets 2 -b 12 --lr 5e-6 --epochs 10 --lam 0.3 --beta 0.5
CUDA_VISIBLE_DEVICES=1 python main.py /data/luhongtao/dataset -d PACS           --log exp0_3 --task domain_shift --targets 3 -b 12 --lr 5e-6 --epochs 10 --lam 0.3 --beta 0.5

# CUDA_VISIBLE_DEVICES=1 python main.py /data/luhongtao/dataset -d VLCS           --log exp1 --task domain_shift --targets 0 -b 12 --lr 5e-6 --epochs 10 --lam 0.3 --beta 0.5
# CUDA_VISIBLE_DEVICES=1 python main.py /data/luhongtao/dataset -d VLCS           --log exp1_1 --task domain_shift --targets 1 -b 12 --lr 5e-6 --epochs 10 --lam 0.3 --beta 0.5
CUDA_VISIBLE_DEVICES=1 python main.py /data/luhongtao/dataset -d VLCS           --log exp1_2 --task domain_shift --targets 2 -b 12 --lr 5e-6 --epochs 10 --lam 0.3 --beta 0.5
# CUDA_VISIBLE_DEVICES=1 python main.py /data/luhongtao/dataset -d VLCS           --log exp1_3 --task domain_shift --targets 3 -b 12 --lr 5e-6 --epochs 10 --lam 0.3 --beta 0.5

# CUDA_VISIBLE_DEVICES=1 python main.py /data/luhongtao/dataset -d OfficeHome     --log exp2 --task domain_shift --targets 0 -b 12 --lr 1e-5 --epochs 10 --lam 0.3 --beta 0.5
# CUDA_VISIBLE_DEVICES=1 python main.py /data/luhongtao/dataset -d OfficeHome     --log exp2_1 --task domain_shift --targets 1 -b 12 --lr 1e-5 --epochs 10 --lam 0.3 --beta 0.5
# CUDA_VISIBLE_DEVICES=1 python main.py /data/luhongtao/dataset -d OfficeHome     --log exp2_2 --task domain_shift --targets 2 -b 12 --lr 1e-5 --epochs 10 --lam 0.3 --beta 0.5
CUDA_VISIBLE_DEVICES=1 python main.py /data/luhongtao/dataset -d OfficeHome     --log exp2_3 --task domain_shift --targets 3 -b 12 --lr 1e-5 --epochs 10 --lam 0.3 --beta 0.5

# CUDA_VISIBLE_DEVICES=1 python main.py /data/luhongtao/dataset -d TerraIncognita --log exp3 --task domain_shift --targets 0 -b 12 --lr 5e-6 --epochs 10 --lam 0.3 --beta 0.5
# CUDA_VISIBLE_DEVICES=1 python main.py /data/luhongtao/dataset -d TerraIncognita --log exp3_1 --task domain_shift --targets 1 -b 12 --lr 5e-6 --epochs 10 --lam 0.3 --beta 0.5
# CUDA_VISIBLE_DEVICES=1 python main.py /data/luhongtao/dataset -d TerraIncognita --log exp3_2 --task domain_shift --targets 2 -b 12 --lr 5e-6 --epochs 10 --lam 0.3 --beta 0.5
CUDA_VISIBLE_DEVICES=0 python main.py /data/luhongtao/dataset -d TerraIncognita --log exp3_3 --task domain_shift --targets 3 -b 12 --lr 5e-6 --epochs 10 --lam 0.3 --beta 0.5

# CUDA_VISIBLE_DEVICES=2 python main.py /data/luhongtao/dataset -d DomainNet      --log exp4 --task domain_shift --targets 0 -b 12 --lr 1e-5 --epochs 20 --lam 0.3 --beta 0.5
# CUDA_VISIBLE_DEVICES=1 python main.py /data/luhongtao/dataset -d DomainNet      --log exp4_1 --task domain_shift --targets 1 -b 12 --lr 1e-5 --epochs 20 --lam 0.3 --beta 0.5
# CUDA_VISIBLE_DEVICES=0 python main.py /data/luhongtao/dataset -d DomainNet      --log exp4_2 --task domain_shift --targets 2 -b 12 --lr 1e-5 --epochs 20 --lam 0.3 --beta 0.5
# CUDA_VISIBLE_DEVICES=2 python main.py /data/luhongtao/dataset -d DomainNet      --log exp4_3 --task domain_shift --targets 3 -b 12 --lr 1e-5 --epochs 20 --lam 0.3 --beta 0.5
# CUDA_VISIBLE_DEVICES=2 python main.py /data/luhongtao/dataset -d DomainNet      --log exp4_4 --task domain_shift --targets 4 -b 12 --lr 1e-5 --epochs 20 --lam 0.3 --beta 0.5
# CUDA_VISIBLE_DEVICES=0 python main.py /data/luhongtao/dataset -d DomainNet      --log exp4_5 --task domain_shift --targets 5 -b 12 --lr 1e-5 --epochs 20 --lam 0.3 --beta 0.5

# Open class
# CUDA_VISIBLE_DEVICES=0 python main.py /data/dataset/CLIP/ -d ImageNet            --task open_class --n-shot 16 -b 8 --lr 5e-6 --epochs 10 --lam 0.3 --beta 0.1
# CUDA_VISIBLE_DEVICES=1 python main.py /data/dataset/CLIP/ -d Caltech101          --task open_class --n-shot 16 -b 32 --lr 5e-6 --epochs 1  --lam 0.3 --beta 0.1
# CUDA_VISIBLE_DEVICES=2 python main.py /data/dataset/CLIP/ -d OxfordPets          --task open_class --n-shot 16 -b 36 --lr 5e-6 --epochs 1  --lam 0.3 --beta 0.1
# CUDA_VISIBLE_DEVICES=0 python main.py /data/dataset/CLIP/ -d StanfordCars        --task open_class --n-shot 16 -b 36 --lr 5e-6 --epochs 5  --lam 0.3 --beta 0.1
# CUDA_VISIBLE_DEVICES=0 python main.py /data/dataset/CLIP/ -d OxfordFlowers       --task open_class --n-shot 16 -b 36 --lr 5e-6 --epochs 1  --lam 0.3 --beta 0.1
# CUDA_VISIBLE_DEVICES=0 python main.py /data/dataset/CLIP/ -d Food101             --task open_class --n-shot 16 -b 36 --lr 5e-6 --epochs 1  --lam 0.3 --beta 0.1
# CUDA_VISIBLE_DEVICES=0 python main.py /data/dataset/CLIP/ -d FGVCAircraft        --task open_class --n-shot 16 -b 36 --lr 5e-6 --epochs 10 --lam 0.3 --beta 0.1
# CUDA_VISIBLE_DEVICES=0 python main.py /data/dataset/CLIP/ -d SUN397              --task open_class --n-shot 16 -b 36 --lr 5e-6 --epochs 5  --lam 0.3 --beta 0.1
# CUDA_VISIBLE_DEVICES=0 python main.py /data/dataset/CLIP/ -d DescribableTextures --task open_class --n-shot 16 -b 36 --lr 5e-6 --epochs 1  --lam 0.3 --beta 0.1
# CUDA_VISIBLE_DEVICES=0 python main.py /data/dataset/CLIP/ -d EuroSAT             --task open_class --n-shot 16 -b 36 --lr 5e-6 --epochs 1  --lam 0.3 --beta 0.1
# CUDA_VISIBLE_DEVICES=0 python main.py /data/dataset/CLIP/ -d UCF101              --task open_class --n-shot 16 -b 36 --lr 5e-6 --epochs 2  --lam 0.3 --beta 0.1

# CUDA_VISIBLE_DEVICES=0 python main.py /data/dataset/CLIP/ -d ImageNet            --task open_class --n-shot 16 -b 12 --lr 5e-6 --epochs 10 --lam 0.3 --beta 0.1 --log ImageNet
# CUDA_VISIBLE_DEVICES=1 python main.py /data/dataset/CLIP/ -d Caltech101          --task open_class --n-shot 16 -b 36 --lr 5e-6 --epochs 1  --lam 0.3 --beta 0.1 --log Caltech101
# CUDA_VISIBLE_DEVICES=2 python main.py /data/dataset/CLIP/ -d OxfordPets          --task open_class --n-shot 16 -b 36 --lr 5e-6 --epochs 1  --lam 0.3 --beta 0.1 --log OxfordPets
# CUDA_VISIBLE_DEVICES=1 python main.py /data/dataset/CLIP/ -d StanfordCars        --task open_class --n-shot 16 -b 36 --lr 5e-6 --epochs 5  --lam 0.3 --beta 0.1 --log StanfordCars
# CUDA_VISIBLE_DEVICES=2 python main.py /data/dataset/CLIP/ -d OxfordFlowers       --task open_class --n-shot 16 -b 36 --lr 5e-6 --epochs 1  --lam 0.3 --beta 0.1 --log OxfordFlowers
# CUDA_VISIBLE_DEVICES=0 python main.py /data/dataset/CLIP/ -d Food101             --task open_class --n-shot 16 -b 36 --lr 5e-6 --epochs 1  --lam 0.3 --beta 0.1 --log Food101
# CUDA_VISIBLE_DEVICES=2 python main.py /data/dataset/CLIP/ -d FGVCAircraft        --task open_class --n-shot 16 -b 36 --lr 5e-6 --epochs 10 --lam 0.3 --beta 0.1 --log FGVCAircraft
# CUDA_VISIBLE_DEVICES=0 python main.py /data/dataset/CLIP/ -d SUN397              --task open_class --n-shot 16 -b 36 --lr 5e-6 --epochs 5  --lam 0.3 --beta 0.1 --log SUN397
# CUDA_VISIBLE_DEVICES=2 python main.py /data/dataset/CLIP/ -d DescribableTextures --task open_class --n-shot 16 -b 36 --lr 5e-6 --epochs 1  --lam 0.3 --beta 0.1 --log DescribableTextures
# CUDA_VISIBLE_DEVICES=0 python main.py /data/dataset/CLIP/ -d EuroSAT             --task open_class --n-shot 16 -b 36 --lr 5e-6 --epochs 1  --lam 0.3 --beta 0.1 --log EuroSAT
# CUDA_VISIBLE_DEVICES=2 python main.py /data/dataset/CLIP/ -d UCF101              --task open_class --n-shot 16 -b 36 --lr 5e-6 --epochs 2  --lam 0.3 --beta 0.1 --log UCF101
# # In-the-wild
# python main.py DomainBed/domainbed/data/ -d OfficeHome --task in_the_wild --targets 0 -b 12 --lr 3e-6 --epochs 10 --lam 0.1 --beta 0.1
# python main.py DomainBed/domainbed/data/ -d OfficeHome --task in_the_wild --targets 1 -b 12 --lr 3e-6 --epochs 10 --lam 0.1 --beta 0.1
# python main.py DomainBed/domainbed/data/ -d OfficeHome --task in_the_wild --targets 2 -b 12 --lr 3e-6 --epochs 10 --lam 0.1 --beta 0.1
# python main.py DomainBed/domainbed/data/ -d OfficeHome --task in_the_wild --targets 3 -b 12 --lr 3e-6 --epochs 10 --lam 0.1 --beta 0.1

# python main.py DomainBed/domainbed/data/ -d DomainNet  --task in_the_wild --targets 0 -b 12 --lr 5e-6 --epochs 20 --lam 0.1 --beta 0.5
# python main.py DomainBed/domainbed/data/ -d DomainNet  --task in_the_wild --targets 1 -b 12 --lr 5e-6 --epochs 20 --lam 0.1 --beta 0.5
# python main.py DomainBed/domainbed/data/ -d DomainNet  --task in_the_wild --targets 2 -b 12 --lr 5e-6 --epochs 20 --lam 0.1 --beta 0.5
# python main.py DomainBed/domainbed/data/ -d DomainNet  --task in_the_wild --targets 3 -b 12 --lr 5e-6 --epochs 20 --lam 0.1 --beta 0.5
# python main.py DomainBed/domainbed/data/ -d DomainNet  --task in_the_wild --targets 4 -b 12 --lr 5e-6 --epochs 20 --lam 0.1 --beta 0.5
# python main.py DomainBed/domainbed/data/ -d DomainNet  --task in_the_wild --targets 5 -b 12 --lr 5e-6 --epochs 20 --lam 0.1 --beta 0.5