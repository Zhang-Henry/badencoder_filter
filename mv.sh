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


# mkdir -p ../DRUPE/data/cifar10
# mkdir -p ../DRUPE/data/stl10
# mkdir -p ../DRUPE/data/gtsrb
# mkdir -p ../DRUPE/data/svhn
# cp -r data/cifar10/*.npz ../DRUPE/data/cifar10
# cp -r data/stl10/*.npz ../DRUPE/data/stl10
# cp -r data/gtsrb/*.npz ../DRUPE/data/gtsrb
# cp -r data/svhn/*.npz ../DRUPE/data/svhn


mkdir -p ../DRUPE/output/cifar10/clean_encoder
mkdir -p ../DRUPE/output/stl10/clean_encoder
mkdir -p ../DRUPE/output/imagenet/clean_encoder
mkdir -p ../DRUPE/output/CLIP/clean_encoder
cp -r output/cifar10/clean_encoder ../DRUPE/output/cifar10/
cp -r output/stl10/clean_encoder ../DRUPE/output/stl10/
cp -r output/imagenet/clean_encoder ../DRUPE/output/imagenet/
cp -r output/CLIP/clean_encoder ../DRUPE/output/CLIP/