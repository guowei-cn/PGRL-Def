from PIL import Image
from torchvision import transforms

from lib.augmentation import EdgePreservingBlur, AddGaussianNoisePIL, AddWhiteNoisePIL

if __name__ == '__main__':
    image1 = Image.open('/storageA/david_projects/DefTimeSeries/poisonDataset/blto/poisonTest/airplane/1001.png')
    transform_f = AddWhiteNoisePIL(noise_range=15.) # AddGaussianNoisePIL(mean=0.0, std=5, p=1) #EdgePreservingBlur(diameter=7, sigma_color=75, sigma_space=75) # AddGaussianNoisePIL(mean=0.0, std=10, p=1) # EdgePreservingBlur(diameter=5, sigma_color=75, sigma_space=150) # transforms.GaussianBlur(kernel_size=5)
    transform_f_2 = EdgePreservingBlur(diameter=10, sigma_color=75, sigma_space=75)
    image2 = transform_f(image1)
    image3 = transform_f_2(image2)
    width1, height1 = image1.size
    width2, height2 = image2.size
    width3, height3 = image3.size
    # Calculate the total width and the max height for the combined image
    total_width = width1 + width2 + width3
    max_height = max(height1, height2)

    # Create a new blank image with the calculated size
    new_image = Image.new("RGB", (total_width, max_height))

    # Paste the images side by side in the new image
    new_image.paste(image1, (0, 0))
    new_image.paste(image2, (width1, 0))
    new_image.paste(image3, (width1 + width3, 0))
    # Save the result
    new_image.save("filter_img.png")
