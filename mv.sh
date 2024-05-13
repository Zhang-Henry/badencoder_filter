# mkdir -p ../IMPERATIVE/data/cifar10
# mkdir -p ../IMPERATIVE/data/stl10
# mkdir -p ../IMPERATIVE/data/gtsrb
# mkdir -p ../IMPERATIVE/data/svhn
# cp -r data/cifar10/*.npz ../IMPERATIVE/data/cifar10
# cp -r data/stl10/*.npz ../IMPERATIVE/data/stl10
# cp -r data/gtsrb/*.npz ../IMPERATIVE/data/gtsrb
# cp -r data/svhn/*.npz ../IMPERATIVE/data/svhn
# cp -r reference ../IMPERATIVE/

# cp -r pytorch_ssim ../IMPERATIVE/

# rsync -av --exclude='optimize_filter/trigger' --exclude='optimize_filter/logs' optimize_filter ../IMPERATIVE/


# mkdir ../IMPERATIVE/output


# cp -r output/cifar10/clean_encoder ../IMPERATIVE/output/cifar10/
# cp -r output/stl10/clean_encoder ../IMPERATIVE/output/stl10/
# cp -r output/imagenet/clean_encoder ../IMPERATIVE/output/imagenet/


# cp  -r data/imagenet/train ../IMPERATIVE/data/imagenet/
# cp  -r data/imagenet/test ../IMPERATIVE/data/imagenet/


# mkdir -p ../BadEncoder/data/cifar10
# mkdir -p ../BadEncoder/data/stl10
# mkdir -p ../BadEncoder/data/gtsrb
# mkdir -p ../BadEncoder/data/svhn
mkdir -p ../BadEncoder/output/CLIP/backdoor
# cp -r data/cifar10/*.npz ../BadEncoder/data/cifar10
# cp -r data/stl10/*.npz ../BadEncoder/data/stl10
# cp -r data/gtsrb/*.npz ../BadEncoder/data/gtsrb
# cp -r data/svhn/*.npz ../BadEncoder/data/svhn
cp -r output/CLIP/backdoor ../BadEncoder/output/CLIP/

# mkdir -p ../BadEncoder/output/cifar10/
# mkdir -p ../BadEncoder/output/stl10/
# mkdir -p ../BadEncoder/output/imagenet/
# mkdir -p ../BadEncoder/output/CLIP/
# cp -r output/cifar10/clean_encoder ../BadEncoder/output/cifar10/
# cp -r output/stl10/clean_encoder ../BadEncoder/output/stl10/
# cp -r output/imagenet/clean_encoder ../BadEncoder/output/imagenet/
# cp -r output/CLIP/clean_encoder ../BadEncoder/output/CLIP/