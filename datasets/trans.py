from torchvision import transforms
from .noise import *
from lightly.transforms import utils
import cv2

def add_salt_and_pepper_noise_HSV(image, amount=0.2):
    image = np.array(image)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # 添加噪声到 HSV 图像
    h, w, _ = hsv_image.shape
    num_pixels = int(amount * h * w)

    # 随机选择添加噪声的像素
    for _ in range(num_pixels):
        i = np.random.randint(0, h)
        j = np.random.randint(0, w)
        channel = np.random.choice([0, 1, 2])  # H, S, or V
        hsv_image[i, j, channel] = np.random.choice([0, 255])  # 黑或白

    rgb_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)
    return torch.from_numpy(rgb_image)




def add_poisson_noise_HSV(image, scale=10):
    image = np.array(image)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # 添加泊松噪声到 V 通道
    hsv_image = hsv_image.astype(np.float32)
    h, s, v = cv2.split(hsv_image)
    noise = np.random.poisson(v / scale)  # 基于明度值生成噪声
    v = np.clip(v + noise, 0, 255)

    # 合并并转换回 RGB
    hsv_noisy = cv2.merge([h, s, v]).astype(np.uint8)
    rgb_noisy = cv2.cvtColor(hsv_noisy, cv2.COLOR_HSV2RGB)
    return torch.from_numpy(rgb_noisy)


test_transform_cifar10 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

test_transform_cifar10_SSL = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(utils.IMAGENET_NORMALIZE["mean"], utils.IMAGENET_NORMALIZE["std"])])

test_transform_cifar10_GaussianBlur = transforms.Compose([
    transforms.GaussianBlur(kernel_size=7),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])


test_transform_cifar10_JPEGcompression = transforms.Compose([
    lambda x: JPEGcompression(x, quality=20),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])


test_transform_cifar10_salt_and_pepper_noise = transforms.Compose([
    lambda x: add_salt_and_pepper_noise(x, 0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])


test_transform_cifar10_salt_and_pepper_noise_HSV = transforms.Compose([
    lambda x: add_salt_and_pepper_noise_HSV(x, amount=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
])


test_transform_cifar10_poisson_noise = transforms.Compose([
    lambda x: add_poisson_noise(x, scale=10),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])



test_transform_cifar10_poisson_noise_HSV = transforms.Compose([
    lambda x: add_poisson_noise_HSV(x, scale=10),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
])


test_transform_stl10_GaussianBlur = transforms.Compose([
    transforms.GaussianBlur(kernel_size=7),
    transforms.ToTensor(),
    transforms.Normalize([0.44087798, 0.42790666, 0.38678814], [0.25507198, 0.24801506, 0.25641308])])

test_transform_stl10_JPEGcompression = transforms.Compose([
    lambda x: JPEGcompression(x, quality=2),
    transforms.ToTensor(),
    transforms.Normalize([0.44087798, 0.42790666, 0.38678814], [0.25507198, 0.24801506, 0.25641308])])


test_transform_stl10_salt_and_pepper_noise = transforms.Compose([
    lambda x: add_salt_and_pepper_noise(x, 0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.44087798, 0.42790666, 0.38678814], [0.25507198, 0.24801506, 0.25641308])])

test_transform_stl10_salt_and_pepper_noise_HSV = transforms.Compose([
    lambda x: add_salt_and_pepper_noise_HSV(x, 0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.44087798, 0.42790666, 0.38678814], [0.25507198, 0.24801506, 0.25641308])])


test_transform_stl10_poisson_noise = transforms.Compose([
    lambda x: add_poisson_noise(x, scale=5),
    transforms.ToTensor(),
    transforms.Normalize([0.44087798, 0.42790666, 0.38678814], [0.25507198, 0.24801506, 0.25641308])])

test_transform_stl10_poisson_noise_HSV = transforms.Compose([
    lambda x: add_poisson_noise_HSV(x, scale=5),
    transforms.ToTensor(),
    transforms.Normalize([0.44087798, 0.42790666, 0.38678814], [0.25507198, 0.24801506, 0.25641308])])


test_transform_stl10 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.44087798, 0.42790666, 0.38678814], [0.25507198, 0.24801506, 0.25641308])])


test_transform_imagenet = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(utils.IMAGENET_NORMALIZE["mean"], utils.IMAGENET_NORMALIZE["std"])
    # transforms.Normalize([0.4850, 0.4560, 0.4060], [0.2290, 0.2240, 0.2250])
    ])

test_transform_CLIP = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),])