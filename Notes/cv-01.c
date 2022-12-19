# 1.1 图像的读取
imgFile = "../images/imgLena.tif"  # 读取文件的路径
img1 = cv2.imread(imgFile, flags=1)  # flags=1 读取彩色图像(BGR)
img2 = cv2.imread(imgFile, flags=0)  # flags=0 读取为灰度图像

# 1.2 从网络读取图像
import urllib.request as request
response = request.urlopen("https://profile.csdnimg.cn/8/E/F/0_youcans")
imgUrl = cv2.imdecode(np.array(bytearray(response.read()), dtype=np.uint8), -1)

# 1.3 读取中文路径的图像
imgFile = "../images/测试图01.png"  # 带有中文的文件路径和文件名
# imread() 不支持中文路径和文件名，读取失败，但不会报错!
# img = cv2.imread(imgFile, flags=1)
# 使用 imdecode 可以读取带有中文的文件路径和文件名
img = cv2.imdecode(np.fromfile(imgFile, dtype=np.uint8), -1)

# 1.4 图像的保存
imgFile = "../images/logoCV.png"  # 读取文件的路径
img3 = cv2.imread(imgFile, flags=1)  # flags=1 读取彩色图像(BGR)

saveFile = "../images/imgSave.png"  # 保存文件的路径
# cv2.imwrite(saveFile, img3, [int(cv2.IMWRITE_PNG_COMPRESSION), 8])  # 保存图像文件, 设置压缩比为 8
cv2.imwrite(saveFile, img3)  # 保存图像文件

# 1.5 保存中文路径的图像
imgFile = "../images/logoCV.png"  # 读取文件的路径
img3 = cv2.imread(imgFile, flags=1)  # flags=1 读取彩色图像(BGR)

saveFile = "../images/测试图02.jpg"  # 带有中文的保存文件路径
# cv2.imwrite(saveFile, img3)  # imwrite 不支持中文路径和文件名，读取失败，但不会报错!
img_write = cv2.imencode(".jpg", img3)[1].tofile(saveFile)

# 1.6 图像的显示(cv2.imshow)
imgFile = "../images/imgLena.tif"  # 读取文件的路径
img1 = cv2.imread(imgFile, flags=1)  # flags=1 读取彩色图像(BGR)
img2 = cv2.imread(imgFile, flags=0)  # flags=0 读取为灰度图像

cv2.imshow("Demo1", img1)  # 在窗口 "Demo1" 显示图像 img1
cv2.imshow("Demo2", img2)  # 在窗口 "Demo2" 显示图像 img2
key = cv2.waitKey(0)  # 等待按键命令, 1000ms 后自动关闭

# 1.7 图像显示(按指定大小的窗口显示图像)
imgFile = "../images/imgLena.tif"  # 读取文件的路径
img1 = cv2.imread(imgFile, flags=1)  # flags=1 读取彩色图像(BGR)

cv2.namedWindow("Demo3", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Demo3", 400, 300)
cv2.imshow("Demo3", img1)  # 在窗口 "Demo3" 显示图像 img1
key = cv2.waitKey(0)  # 等待按键命令, 1000ms 后自动关闭

# 1.8 图像显示(多个图像组合显示)
imgFile1 = "../images/imgLena.tif"  # 读取文件的路径
img1 = cv2.imread(imgFile1, flags=1)  # flags=1 读取彩色图像(BGR)
imgFile2 = "../images/imgGaia.tif"  # 读取文件的路径
img2 = cv2.imread(imgFile2, flags=1)  # # flags=1 读取彩色图像(BGR)

imgStack = np.hstack((img1, img2))  # 相同大小图像水平拼接
cv2.imshow("Demo4", imgStack)  # 在窗口 "Demo4" 显示图像 imgStack
key = cv2.waitKey(0)  # 等待按键命令, 1000ms 后自动关闭


matplotlib.pyplot.imshow(img[, cmap])
# 1.10 图像显示(plt.imshow)
imgFile = "../images/imgLena.tif"  # 读取文件的路径
img1 = cv2.imread(imgFile, flags=1)  # flags=1 读取彩色图像(BGR)

imgRGB = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)  # 图片格式转换：BGR(OpenCV) -> RGB(PyQt5)
img2 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)  # 图片格式转换：BGR(OpenCV) -> Gray

plt.rcParams['font.sans-serif'] = ['FangSong']  # 支持中文标签
plt.subplot(221), plt.title("1. RGB 格式(mpl)"), plt.axis('off')
plt.imshow(imgRGB)  # matplotlib 显示彩色图像(RGB格式)
plt.subplot(222), plt.title("2. BGR 格式(OpenCV)"), plt.axis('off')
plt.imshow(img1)    # matplotlib 显示彩色图像(BGR格式)
plt.subplot(223), plt.title("3. 设置 Gray 参数"), plt.axis('off')
plt.imshow(img2, cmap='gray')  # matplotlib 显示灰度图像，设置 Gray 参数
plt.subplot(224), plt.title("4. 未设置 Gray 参数"), plt.axis('off')
plt.imshow(img2)  # matplotlib 显示灰度图像，未设置 Gray 参数
plt.show()

# 1.11 图像数组的属性
imgFile = "../images/imgLena.tif"  # 读取文件的路径
img1 = cv2.imread(imgFile, flags=1)  # flags=1 读取彩色图像(BGR)
img2 = cv2.imread(imgFile, flags=0)  # flags=0 读取为灰度图像
# cv2.imshow("Demo1", img1)  # 在窗口显示图像
# key = cv2.waitKey(0)  # 等待按键命令

# 维数(ndim), 形状(shape), 元素总数(size), 元素类型(dtype)
print("Ndim of img1(BGR): {}, img2(Gray): {}".format(img1.ndim, img2.ndim))  # number of rows, columns and channels
print("Shape of img1(BGR): {}, img2(Gray): {}".format(img1.shape, img2.shape))  # number of rows, columns and channels
print("Size of img1(BGR): {}, img2(Gray): {}".format(img1.size, img2.size))  # size = rows * columns * channels
print("Dtype of img1(BGR): {}, img2(Gray): {}".format(img1.dtype, img2.dtype))  # uint8

//本例程的运行结果如下：
// Ndim of img1(BGR): 3, img2(Gray): 2
// Shape of img1(BGR): (512, 512, 3), img2(Gray): (512, 512)
// Size of img1(BGR): 786432, img2(Gray): 262144
// Dtype of img1(BGR): uint8, img2(Gray): uint8

# 1.13 Numpy 获取和修改像素值
img1 = cv2.imread("../images/imgLena.tif", flags=1)  # flags=1 读取彩色图像(BGR)
x, y = 10, 10  # 指定像素位置 x, y

# (1) 直接访问数组元素，获取像素值(BGR)
pxBGR = img1[x,y]  # 访问数组元素[x,y], 获取像素 [x,y] 的值
print("x={}, y={}\nimg[x,y] = {}".format(x,y,img1[x,y]))
# (2) 直接访问数组元素，获取像素通道的值
print("img[{},{},ch]:".format(x,y))
for i in range(3):
	print(img1[x, y, i], end=' ')  # i=0,1,2 对应 B,G,R 通道
# (3) img.item() 访问数组元素，获取像素通道的值
print("\nimg.item({},{},ch):".format(x,y))
for i in range(3):
	print(img1.item(x, y, i), end=' ')  # i=0,1,2 对应 B,G,R 通道

# (4) 修改像素值：img.itemset() 访问数组元素，修改像素通道的值
ch, newValue = 0, 255
print("\noriginal img[x,y] = {}".format(img1[x,y]))
img1.itemset((x, y, ch), newValue)  # 将 [x,y,channel] 的值修改为 newValue
print("updated img[x,y] = {}".format(img1[x,y]))

// 本例程的运行结果如下：
// x=10, y=10
// img[x,y] = [113 131 226]
// img[10,10,ch]:  113 131 226 
// img.item(10,10,ch):  113 131 226 
// original img[x,y] = [113 131 226]
// updated  img[x,y] = [255 131 226]


# 1.14 Numpy 创建图像
# 创建彩色图像(RGB)
# (1) 通过宽度高度值创建多维数组
height, width, channels = 400, 300, 3  # 行/高度, 列/宽度, 通道数
imgEmpty = np.empty((height, width, channels), np.uint8)  # 创建空白数组
imgBlack = np.zeros((height, width, channels), np.uint8)  # 创建黑色图像 RGB=0
imgWhite = np.ones((height, width, channels), np.uint8) * 255  # 创建白色图像 RGB=255
# (2) 创建相同形状的多维数组
img1 = cv2.imread("../images/imgLena.tif", flags=1)  # flags=1 读取彩色图像(BGR)
imgBlackLike = np.zeros_like(img1)  # 创建与 img1 相同形状的黑色图像
imgWhiteLike = np.ones_like(img1) * 255  # 创建与 img1 相同形状的白色图像
# (3) 创建彩色随机图像 RGB=random
import os
randomByteArray = bytearray(os.urandom(height * width * channels))
flatNumpyArray = np.array(randomByteArray)
imgRGBRand = flatNumpyArray.reshape(height, width, channels)
# (4) 创建灰度图像
imgGrayWhite = np.ones((height, width), np.uint8) * 255  # 创建白色图像 Gray=255
imgGrayBlack = np.zeros((height, width), np.uint8)  # 创建黑色图像 Gray=0
imgGrayEye = np.eye(width)  # 创建对角线元素为1 的单位矩阵
randomByteArray = bytearray(os.urandom(height*width))
flatNumpyArray = np.array(randomByteArray)
imgGrayRand = flatNumpyArray.reshape(height, width)  # 创建灰度随机图像 Gray=random

print("Shape of image: gray {}, RGB {}".format(imgGrayRand.shape, imgRGBRand.shape))
cv2.imshow("DemoGray", imgGrayRand)  # 在窗口显示 灰度随机图像
cv2.imshow("DemoRGB", imgRGBRand)  # 在窗口显示 彩色随机图像
cv2.imshow("DemoBlack", imgBlack)  # 在窗口显示 黑色图像
key = cv2.waitKey(0)  # 等待按键命令
// numpy.empty(shape[, dtype, order]) # 返回一个指定形状和类型的空数组
// numpy.zeros(shape[, dtype, order]) # 返回一个指定形状和类型的全零数组
// numpy.ones(shape[, dtype, order]) # 返回一个指定形状和类型的全一数组
// numpy.empty_like(img) # 返回一个与图像 img 形状和类型相同的空数组
// numpy.zeros_like(img) # 返回一个与图像 img 形状和类型相同的全零数组
// numpy.ones_like(img) # 返回一个与图像 img 形状和类型相同的全一数组



