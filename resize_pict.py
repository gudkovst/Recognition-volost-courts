import matplotlib.pyplot as plt
import glob

SIZE = 200
path = r"C:\\Users\\User\\Desktop\\test-pictures\\resize\\"

for filename in glob.glob(path + '*.png'):
    print(filename)
    img = plt.imread(filename)
    size = img.shape
    shift0 = (size[0] - SIZE) // 2
    shift1 = (size[1] - SIZE) // 2
    res_img = img[shift0:shift0+SIZE, shift1:shift1+SIZE, :]
    plt.imsave(filename, res_img)
    
