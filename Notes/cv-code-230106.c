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

// arr = numpy.copy(img) # 返回一个复制的图像
# 1.15 图像的复制
img1 = cv2.imread("../images/imgLena.tif", flags=1)  # flags=1 读取彩色图像(BGR)
img2 = img1.copy()
print("img2=img1.copy(), img2 is img1?", img2 is img1)
for col in range(100):
	for row in range(100):
		img2[col, row, :] = 0

img3 = cv2.imread("../images/imgLena.tif", flags=1)  # flags=1 读取彩色图像(BGR)
img4 = img3
print("img4=img3, img4 is img3?", img4 is img3)
for col in range(100):
	for row in range(100):
		img4[col, row, :] = 0

cv2.imshow("Demo1", img1)  # 在窗口显示图像
cv2.imshow("Demo2", img2)  # 在窗口显示图像
cv2.imshow("Demo3", img3)  # 在窗口显示图像
cv2.imshow("Demo4", img4)  # 在窗口显示图像
key = cv2.waitKey(0)  # 等待按键命令

// retval = img[y:y+h, x:x+w].copy()
# 1.16 图像的裁剪
img1 = cv2.imread("../images/imgLena.tif", flags=1)  # flags=1 读取彩色图像(BGR)
xmin, ymin, w, h = 180, 190, 200, 200  # 矩形裁剪区域 (ymin:ymin+h, xmin:xmin+w) 的位置参数
imgCrop = img1[ymin:ymin+h, xmin:xmin+w].copy()  # 切片获得裁剪后保留的图像区域
cv2.imshow("DemoCrop", imgCrop)  # 在窗口显示 彩色随机图像
key = cv2.waitKey(0)  # 等待按键命令

// cv2.selectROI(windowName, img, showCrosshair=None, fromCenter=None):
# 1.17 图像的裁剪 (ROI)
img1 = cv2.imread("../images/imgLena.tif", flags=1)  # flags=1 读取彩色图像(BGR)
roi = cv2.selectROI(img1, showCrosshair=True, fromCenter=False)
xmin, ymin, w, h = roi  # 矩形裁剪区域 (ymin:ymin+h, xmin:xmin+w) 的位置参数
imgROI = img1[ymin:ymin+h, xmin:xmin+w].copy()  # 切片获得裁剪后保留的图像区域
cv2.imshow("DemoRIO", imgROI)
cv2.waitKey(0)

// retval = numpy.hstack((img1, img2, …)) # 水平拼接
// retval = numpy.vstack((img1, img2, …)) # 垂直拼接
# 1.18 图像拼接
img1 = cv2.imread("../images/imgLena.tif")  # 读取彩色图像(BGR)
img2 = cv2.imread("../images/logoCV.png")  # 读取彩色图像(BGR)
img1 = cv2.resize(img1, (400, 400))
img2 = cv2.resize(img2, (300, 400))
img3 = cv2.resize(img2, (400, 300))
imgStackH = np.hstack((img1, img2))  # 高度相同图像可以横向水平拼接
imgStackV = np.vstack((img1, img3))  # 宽度相同图像可以纵向垂直拼接

print("Horizontal stack:\nShape of img1, img2 and imgStackH: ", img1.shape, img2.shape, imgStackH.shape)
print("Vertical stack:\nShape of img1, img3 and imgStackV: ", img1.shape, img3.shape, imgStackV.shape)
cv2.imshow("DemoStackH", imgStackH)  # 在窗口显示图像 imgStackH
cv2.imshow("DemoStackV", imgStackV)  # 在窗口显示图像 imgStackV
key = cv2.waitKey(0)  # 等待按键命令

// Horizontal stack:
// Shape of img1, img2 and imgStackH:  (400, 400, 3) (400, 300, 3) (400, 700, 3)
// Vertical stack:
// Shape of img1, img3 and imgStackV:  (400, 400, 3) (300, 400, 3) (700, 400, 3)

// cv2.split(img[, mv]) -> retval # 图像拆分为 BGR 通道
# 1.19 图像拆分通道
img1 = cv2.imread("../images/imgB1.jpg", flags=1)  # flags=1 读取彩色图像(BGR)
cv2.imshow("BGR", img1)  # BGR 图像

# BGR 通道拆分
bImg, gImg, rImg = cv2.split(img1)  # 拆分为 BGR 独立通道
cv2.imshow("rImg", rImg)  # 直接显示红色分量 rImg 显示为灰度图像

# 将单通道扩展为三通道
imgZeros = np.zeros_like(img1)  # 创建与 img1 相同形状的黑色图像
imgZeros[:,:,2] = rImg  # 在黑色图像模板添加红色分量 rImg
cv2.imshow("channel R", imgZeros)  # 扩展为 BGR 通道

print(img1.shape, rImg.shape, imgZeros.shape)
cv2.waitKey(0)
cv2.destroyAllWindows()  # 释放所有窗口
// 本例程的运行结果如下：
// (512, 512, 3) (512, 512) (512, 512, 3)
// 1.彩色图像 img1 的形状为 (512, 512, 3)，拆分的 R 通道 rImg 的形状为 (512, 512)。
// 2.用 imshow 显示 rImg，将被视为 (512, 512) 形状的灰度图像显示，不能显示为红色通道。
// 3.对 rImg 增加 B、G 两个通道值（置 0）转换为 BGR格式，再用 imshow 才能显示红色通道的颜色。

# 1.20 图像拆分通道 (Numpy切片)
img1 = cv2.imread("../images/imgB1.jpg", flags=1)  # flags=1 读取彩色图像(BGR)

# 获取 B 通道
bImg = img1.copy()  # 获取 BGR
bImg[:, :, 1] = 0  # G=0
bImg[:, :, 2] = 0  # R=0

# 获取 G 通道
gImg = img1.copy()  # 获取 BGR
gImg[:, :, 0] = 0  # B=0
gImg[:, :, 2] = 0  # R=0

# 获取 R 通道
rImg = img1.copy()  # 获取 BGR
rImg[:, :, 0] = 0  # B=0
rImg[:, :, 1] = 0  # G=0

# 消除 B 通道
grImg = img1.copy()  # 获取 BGR
grImg[:, :, 0] = 0  # B=0

plt.subplot(221), plt.title("1. B channel"), plt.axis('off')
bImg = cv2.cvtColor(bImg, cv2.COLOR_BGR2RGB)  # 图片格式转换：BGR(OpenCV) -> RGB(PyQt5)
plt.imshow(bImg)  # matplotlib 显示 channel B
plt.subplot(222), plt.title("2. G channel"), plt.axis('off')
gImg = cv2.cvtColor(gImg, cv2.COLOR_BGR2RGB)
plt.imshow(gImg)  # matplotlib 显示 channel G
plt.subplot(223), plt.title("3. R channel"), plt.axis('off')
rImg = cv2.cvtColor(rImg, cv2.COLOR_BGR2RGB)
plt.imshow(rImg)  # matplotlib 显示 channel R
plt.subplot(224), plt.title("4. GR channel"), plt.axis('off')
grImg = cv2.cvtColor(grImg, cv2.COLOR_BGR2RGB)
plt.imshow(grImg)  # matplotlib 显示 channel GR
plt.show()

//函数 cv2.merge() 将 B、G、R 单通道合并为 3 通道 BGR 彩色图像
//cv2.merge(mv[, dst]) -> retval # BGR 通道合并
# 1.21 图像通道的合并
img1 = cv2.imread("../images/imgB1.jpg", flags=1)  # flags=1 读取彩色图像(BGR)
bImg, gImg, rImg = cv2.split(img1)  # 拆分为 BGR 独立通道
# cv2.merge 实现图像通道的合并
imgMerge = cv2.merge([bImg, gImg, rImg])
cv2.imshow("cv2Merge", imgMerge)
# Numpy 拼接实现图像通道的合并
imgStack = np.stack((bImg, gImg, rImg), axis=2)
cv2.imshow("npStack", imgStack)
print(imgMerge.shape, imgStack.shape)
print("imgMerge is imgStack?", np.array_equal(imgMerge, imgStack))
cv2.waitKey(0)
cv2.destroyAllWindows()  # 释放所有窗口
// (512, 512, 3) (512, 512, 3)
// imgMerge is imgStack? True

//函数 cv2.add() 用于图像的加法运算。
// OpenCV 加法和 numpy 加法之间有区别：cv2.add() 是饱和运算（相加后如大于 255 则结果为 255），而 Numpy 加法是模运算(255+64)%255=64。
// 使用 cv2.add() 函数对两张图片相加时，图片的大小和类型（通道数）必须相同。
// 使用 cv2.add() 函数对一张图像与一个标量相加，标量是指一个 1x3 的 numpy 数组，相加后图像整体发白。

# 1.22 图像的加法 (cv2.add)
img1 = cv2.imread("../images/imgB1.jpg")  # 读取彩色图像(BGR)
img2 = cv2.imread("../images/imgB3.jpg")  # 读取彩色图像(BGR)

imgAddCV = cv2.add(img1, img2)  # OpenCV 加法: 饱和运算
imgAddNP = img1 + img2  # # Numpy 加法: 模运算

plt.subplot(221), plt.title("1. img1"), plt.axis('off')
plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))  # 显示 img1(RGB)
plt.subplot(222), plt.title("2. img2"), plt.axis('off')
plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))  # 显示 img2(RGB)
plt.subplot(223), plt.title("3. cv2.add(img1, img2)"), plt.axis('off')
plt.imshow(cv2.cvtColor(imgAddCV, cv2.COLOR_BGR2RGB))  # 显示 imgAddCV(RGB)
plt.subplot(224), plt.title("4. img1 + img2"), plt.axis('off')
plt.imshow(cv2.cvtColor(imgAddNP, cv2.COLOR_BGR2RGB))  # 显示 imgAddNP(RGB)
plt.show()

# 1.23 图像的加法 (与标量相加)
img1 = cv2.imread("../images/imgB1.jpg")  # 读取彩色图像(BGR)
img2 = cv2.imread("../images/imgB3.jpg")  # 读取彩色图像(BGR)

Value = 100  # 常数
# Scalar = np.array([[50., 100., 150.]])  # 标量
Scalar = np.ones((1, 3), dtype="float") * Value  # 标量
imgAddV = cv2.add(img1, Value)  # OpenCV 加法: 图像 + 常数
imgAddS = cv2.add(img1, Scalar)  # OpenCV 加法: 图像 + 标量

print("Shape of scalar", Scalar)
for i in range(1, 6):
	x, y = i*10, i*10
	print("(x,y)={},{}, img1:{}, imgAddV:{}, imgAddS:{}"
		  .format(x,y,img1[x,y],imgAddV[x,y],imgAddS[x,y]))

plt.subplot(131), plt.title("1. img1"), plt.axis('off')
plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))  # 显示 img1(RGB)
plt.subplot(132), plt.title("2. img + constant"), plt.axis('off')
plt.imshow(cv2.cvtColor(imgAddV, cv2.COLOR_BGR2RGB))  # 显示 imgAddV(RGB)
plt.subplot(133), plt.title("3. img + scalar"), plt.axis('off')
plt.imshow(cv2.cvtColor(imgAddS, cv2.COLOR_BGR2RGB))  # 显示 imgAddS(RGB)
plt.show()

//结果
// Shape of scalar [[150. 150. 150.]]
// (x,y)=10,10, img1:[ 9  9 69], imgAddV:[159   9  69], imgAddS:[159 159 219]
// (x,y)=20,20, img1:[  3 252 255], imgAddV:[153 252 255], imgAddS:[153 255 255]
// (x,y)=30,30, img1:[  1 255 254], imgAddV:[151 255 254], imgAddS:[151 255 255]
// (x,y)=40,40, img1:[  1 255 254], imgAddV:[151 255 254], imgAddS:[151 255 255]
// (x,y)=50,50, img1:[  1 255 255], imgAddV:[151 255 255], imgAddS:[151 255 255]
// 注意 cv2.add() 对图像与标量相加时，“常数” 与 “标量” 的区别：
// 将图像与一个常数 value 相加，只是将 B 通道即蓝色分量与常数相加，而 G、R 通道的数值不变，因此图像发蓝。
// 将图像与一个标量 scalar 相加，“标量” 是指一个 1x3 的 numpy 数组，此时 B/G/R 通道分别与数组中对应的常数相加，因此图像发白。

//cv2.addWeighted(src1, alpha, src2, beta, gamma[, dst[, dtype]]) → dst 
// dst = src1 * alpha + src2 * beta + gamma
# 1.24 图像的混合(加权加法)
img1 = cv2.imread("../images/imgGaia.tif")  # 读取图像 imgGaia
img2 = cv2.imread("../images/imgLena.tif")  # 读取图像 imgLena

imgAddW1 = cv2.addWeighted(img1, 0.2, img2, 0.8, 0)  # 加权相加, a=0.2, b=0.8
imgAddW2 = cv2.addWeighted(img1, 0.5, img2, 0.5, 0)  # 加权相加, a=0.5, b=0.5
imgAddW3 = cv2.addWeighted(img1, 0.8, img2, 0.2, 0)  # 加权相加, a=0.8, b=0.2

plt.subplot(131), plt.title("1. a=0.2, b=0.8"), plt.axis('off')
plt.imshow(cv2.cvtColor(imgAddW1, cv2.COLOR_BGR2RGB))  # 显示 img1(RGB)
plt.subplot(132), plt.title("2. a=0.5, b=0.5"), plt.axis('off')
plt.imshow(cv2.cvtColor(imgAddW2, cv2.COLOR_BGR2RGB))  # 显示 imgAddV(RGB)
plt.subplot(133), plt.title("3. a=0.8, b=0.2"), plt.axis('off')
plt.imshow(cv2.cvtColor(imgAddW3, cv2.COLOR_BGR2RGB))  # 显示 imgAddS(RGB)
plt.show()

// 使用函数 cv2.add()、cv2.addWeight() 对两张图片相加时，图片的大小和类型（通道数）必须相同。
// 对于不同尺寸的图像加法，将小图叠加到大图的指定位置，可以按扩展例程 1.25 处理。
# 1.25 不同尺寸的图像加法
imgL = cv2.imread("../images/imgB2.jpg")  # 读取大图
imgS = cv2.imread("../images/logoCV.png")  # 读取小图 (LOGO)

x,y = 300,50  # 叠放位置
W1, H1 = imgL.shape[1::-1]  # 大图尺寸
W2, H2 = imgS.shape[1::-1]  # 小图尺寸
if (x + W2) > W1: x = W1 - W2  # 调整图像叠放位置，避免溢出
if (y + H2) > H1: y = H1 - H2

imgCrop = imgL[y:y + H2, x:x + W2]  # 裁剪大图，与小图 imgS 的大小相同
imgAdd = cv2.add(imgCrop, imgS)  # cv2 加法，裁剪图与小图叠加
alpha, beta, gamma = 0.2, 0.8, 0.0  # 加法权值
imgAddW = cv2.addWeighted(imgCrop, alpha, imgS, beta, gamma)  # 加权加法，裁剪图与小图叠加

imgAddM = np.array(imgL)
imgAddM[y:y + H2, x:x + W2] = imgAddW  # 用叠加小图替换原图 imgL 的叠放位置

cv2.imshow("imgAdd", imgAdd)
cv2.imshow("imgAddW", imgAddW)
cv2.imshow("imgAddM", imgAddM)
cv2.waitKey(0)

// cv2.addWeighted(src1, alpha, src2, beta, gamma[, dst[, dtype]]) → dst
// dst = src1 * alpha + src2 * beta + gamma
# 1.26 两张图像的的渐变切换 (改变加权叠加的权值)
img1 = cv2.imread("../images/imgLena.tif")  # 读取图像 imgLena
img2 = cv2.imread("../images/imgGiga.jpg")  # 读取彩色图像(BGR)
wList = np.arange(0.0, 1.0, 0.05)  # start, end, step
for w in wList:
	imgAddW = cv2.addWeighted(img1, w, img2, (1 - w), 0)
	cv2.imshow("imgAddWeight", imgAddW)
	cv2.waitKey(100)

// imgAddMask1 是标准的掩模加法，在窗口区域将 img1 与 img2 进行饱和加法，其它区域为黑色遮蔽。imgAddMask2 中加法运算的第二图像是全黑图像（数值为 0），掩模加法的结果是从第一图像中提取遮蔽窗口，该操作生成的图像是从原图中提取感兴趣区域（ROI）、黑色遮蔽其它区域。
# 1.27 图像的加法 (掩模 mask)
img1 = cv2.imread("../images/imgLena.tif")  # 读取彩色图像(BGR)
img2 = cv2.imread("../images/imgB3.jpg")  # 读取彩色图像(BGR)

Mask = np.zeros((img1.shape[0], img1.shape[1]), dtype=np.uint8)  # 返回与图像 img1 尺寸相同的全零数组
xmin, ymin, w, h = 180, 190, 200, 200  # 矩形裁剪区域 (ymin:ymin+h, xmin:xmin+w) 的位置参数
Mask[ymin:ymin+h, xmin:xmin+w] = 255  # 掩模图像，ROI 为白色，其它区域为黑色
print(img1.shape, img2.shape, Mask.shape)

imgAddMask1 = cv2.add(img1, img2, mask=Mask)  # 带有掩模 mask 的加法
imgAddMask2 = cv2.add(img1, np.zeros(np.shape(img1), dtype=np.uint8), mask=Mask)  # 提取 ROI

cv2.imshow("MaskImage", Mask)  # 显示掩模图像 Mask
cv2.imshow("MaskAdd", imgAddMask1)  # 显示掩模加法结果 imgAddMask1
cv2.imshow("MaskROI", imgAddMask2)  # 显示从 img1 提取的 ROI
key = cv2.waitKey(0)  # 等待按键命令

# 1.28 图像的加法 (圆形和其它形状的遮罩)
img1 = cv2.imread("../images/imgLena.tif")  # 读取彩色图像(BGR)
img2 = cv2.imread("../images/imgB3.jpg")  # 读取彩色图像(BGR)

Mask1 = np.zeros((img1.shape[0], img1.shape[1]), dtype=np.uint8)  # 返回与图像 img1 尺寸相同的全零数组
Mask2 = Mask1.copy()
cv2.circle(Mask1, (285, 285), 110, (255, 255, 255), -1)  # -1 表示实心
cv2.ellipse(Mask2, (285, 285), (100, 150), 0, 0, 360, 255, -1)  # -1 表示实心

imgAddMask1 = cv2.add(img1, np.zeros(np.shape(img1), dtype=np.uint8), mask=Mask1)  # 提取圆形 ROI
imgAddMask2 = cv2.add(img1, np.zeros(np.shape(img1), dtype=np.uint8), mask=Mask2)  # 提取椭圆 ROI

cv2.imshow("circularMask", Mask1)  # 显示掩模图像 Mask
cv2.imshow("circularROI", imgAddMask1)  # 显示掩模加法结果 imgAddMask1
cv2.imshow("ellipseROI", imgAddMask2)  # 显示掩模加法结果 imgAddMask2
key = cv2.waitKey(0)  # 等待按键命令

cv.bitwise_and(src1, src2[, dst[, mask]] → dst  # 位操作: 与
cv.bitwise_or(src1, src2[, dst[, mask]] → dst  # 位操作: 或
cv.bitwise_xor(src1, src2[, dst[, mask]] → dst  # 位操作: 与或
cv.bitwise_not(src1, src2[, dst[, mask]] → dst  # 位操作: 非（取反）


# 1.30 图像的叠加

img1 = cv2.imread("../images/imgLena.tif")  # 读取彩色图像(BGR)
img2 = cv2.imread("../images/logoCV.png")  # 读取 CV Logo

x, y = (0, 10)  # 图像叠加位置
W1, H1 = img1.shape[1::-1]
W2, H2 = img2.shape[1::-1]
if (x + W2) > W1: x = W1 - W2
if (y + H2) > H1: y = H1 - H2
print(W1,H1,W2,H2,x,y)
imgROI = img1[y:y+H2, x:x+W2]  # 从背景图像裁剪出叠加区域图像

img2Gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)  # img2: 转换为灰度图像
ret, mask = cv2.threshold(img2Gray, 175, 255, cv2.THRESH_BINARY)  # 转换为二值图像，生成遮罩，LOGO 区域黑色遮盖
maskInv = cv2.bitwise_not(mask)  # 按位非(黑白转置)，生成逆遮罩，LOGO 区域白色开窗，LOGO 以外区域黑色

# mask 黑色遮盖区域输出为黑色，mask 白色开窗区域与运算（原图像素不变）
img1Bg = cv2.bitwise_and(imgROI, imgROI, mask=mask)  # 生成背景，imgROI 的遮罩区域输出黑色
img2Fg = cv2.bitwise_and(img2, img2, mask=maskInv)  # 生成前景，LOGO 的逆遮罩区域输出黑色
# img1Bg = cv2.bitwise_or(imgROI, imgROI, mask=mask)  # 生成背景，与 cv2.bitwise_and 效果相同
# img2Fg = cv2.bitwise_or(img2, img2, mask=maskInv)  # 生成前景，与 cv2.bitwise_and 效果相同
# img1Bg = cv2.add(imgROI, np.zeros(np.shape(img2), dtype=np.uint8), mask=mask)  # 生成背景，与 cv2.bitwise 效果相同
# img2Fg = cv2.add(img2, np.zeros(np.shape(img2), dtype=np.uint8), mask=maskInv)  # 生成背景，与 cv2.bitwise 效果相同
imgROIAdd = cv2.add(img1Bg, img2Fg)  # 前景与背景合成，得到裁剪部分的叠加图像
imgAdd = img1.copy()
imgAdd[y:y+H2, x:x+W2] = imgROIAdd  # 用叠加图像替换背景图像中的叠加位置，得到叠加 Logo 合成图像

plt.figure(figsize=(9,6))
titleList = ["1. imgGray", "2. imgMask", "3. MaskInv", "4. img2FG", "5. img1BG", "6. imgROIAdd"]
imageList = [img2Gray, mask, maskInv, img2Fg, img1Bg, imgROIAdd]
for i in range(6):
	plt.subplot(2,3,i+1), plt.title(titleList[i]), plt.axis('off')
	if (imageList[i].ndim==3):  # 彩色图像 ndim=3
		plt.imshow(cv2.cvtColor(imageList[i], cv2.COLOR_BGR2RGB))  # 彩色图像需要转换为 RGB 格式
	else:  # 灰度图像 ndim=2
		plt.imshow(imageList[i], 'gray')
plt.show()
cv2.imshow("imgAdd", imgAdd)  # 显示叠加图像 imgAdd
key = cv2.waitKey(0)  # 等待按键命令

# 1.31 图像添加文字
img1 = cv2.imread("../images/imgLena.tif")  # 读取彩色图像(BGR)
text = "OpenCV2021, youcans@xupt"
fontList = [cv2.FONT_HERSHEY_SIMPLEX,
			cv2.FONT_HERSHEY_SIMPLEX,
			cv2.FONT_HERSHEY_PLAIN,
			cv2.FONT_HERSHEY_DUPLEX,
			cv2.FONT_HERSHEY_COMPLEX,
			cv2.FONT_HERSHEY_TRIPLEX,
			cv2.FONT_HERSHEY_COMPLEX_SMALL,
			cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
			cv2.FONT_HERSHEY_SCRIPT_COMPLEX,
			cv2.FONT_ITALIC]
fontScale = 1  # 字体缩放比例
color = (255, 255, 255)  # 字体颜色
for i in range(10):
	pos = (10, 50*(i+1))
	imgPutText = cv2.putText(img1, text, pos, fontList[i], fontScale, color)

cv2.imshow("imgPutText", imgPutText)  # 显示叠加图像 imgAdd
key = cv2.waitKey(0)  # 等待按键命令

# 1.32 图像中添加中文文字
imgBGR = cv2.imread("../images/imgLena.tif")  # 读取彩色图像(BGR)

from PIL import Image, ImageDraw, ImageFont
if (isinstance(imgBGR, np.ndarray)):  # 判断是否 OpenCV 图片类型
	imgPIL = Image.fromarray(cv2.cvtColor(imgBGR, cv2.COLOR_BGR2RGB))
text = "OpenCV2021, 中文字体"
pos = (50, 20)  # (left, top)，字符串左上角坐标
color = (255, 255, 255)  # 字体颜色
textSize = 40
drawPIL = ImageDraw.Draw(imgPIL)
fontText = ImageFont.truetype("font/simsun.ttc", textSize, encoding="utf-8")
drawPIL.text(pos, text, color, font=fontText)
imgPutText = cv2.cvtColor(np.asarray(imgPIL), cv2.COLOR_RGB2BGR)

cv2.imshow("imgPutText", imgPutText)  # 显示叠加图像 imgAdd
key = cv2.waitKey(0)  # 等待按键命令














