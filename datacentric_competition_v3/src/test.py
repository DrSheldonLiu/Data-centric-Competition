import numpy as np
from PIL import Image

import matplotlib.pyplot as plt

x = np.random.rand(250, 500)
print('image shape is ', x.shape)
plt.figure()
plt.imshow(x * 255)
img_ = Image.fromarray(x * 255, 'L')
img_.save('test_img.png')
# plt.savefig('test_img.png')
img = Image.open('test_img.png')
print(np.array(img).shape)
plt.show()