import numpy as np
from PIL import Image

# img = Image.open("pic.jpg")
# img = np.array(img)
# print(img)
# print(img.shape)

# img = np.random.randint(0, 255, 90000).reshape(300, 300)
# img = Image.fromarray(img, "L")
# img.show()

img = np.random.randint(0, 255, 270000).reshape((300, 300, 3))
img = Image.fromarray(img, "RGB")
img.show()
