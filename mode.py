import os
import cv2

filename = '/home/sakuma/Developments/python/receipt.jpg'
input_file = None
input_file_gray = None
input_file_blur = None
input_file_thresh = None
dirname = 'images'

def read_image():
    global filename, input_file
    # 画像を読み込む
    input_file = cv2.imread(filename)

def convert_to_gray():
    global input_file, input_file_gray
    input_file_gray = cv2.cvtColor(input_file, cv2.COLOR_BGR2GRAY)

def blur_image():
    global input_file_gray, input_file_blur
    input_file_blur = cv2.GaussianBlur(input_file_gray, (15, 15), 2)

def thresh():
    global input_file_blur, input_file_thresh
    # ret2, img_otsu = cv2.threshold(input_file_blur, 0, 255, cv2.THRESH_OTSU)
    input_file_thresh = cv2.adaptiveThreshold(input_file_blur, 255, 1, 1, 11, 2)
    cv2.imwrite('img_gauss.jpg', input_file_thresh)

def extract_contour():
    global input_file_thresh, input_file, dirname
    # 輪郭を抽出
    # contours = cv2.findContours(input_file_thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
    contours = cv2.findContours(input_file_thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
    # cnts, hierarchy = cv2.findContours(img_dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # インデックス
    index = 0
    # フォルダが存在しない場合
    if not os.path.exists(dirname):
        os.mkdir(dirname)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if not 5 < w < 1000:
            continue
        if not 5 < h < 1000:
            continue
        red = (0, 0, 255)
        # 赤色の枠を追加
        cv2.rectangle(input_file, (x, y), (x+w, y+h), red, 2)
        # 右上の座標
        x2 = x+w
        # 左下の座標
        y2 = y+h
        # 画像を切り出す
        cut_img = input_file[y:y2,x:x2]
        # 切り出した画像の名前を取得
        cut_img_name = 'cut' + str(index) + '.jpg'
        # 画像を保存
        # cv2.imwrite(os.path.join(dirname, cut_img_name), cut_img)
        index += 1
    # 画像を保存
    cv2.imwrite('result.jpg', input_file)

# 画像を読み込む
read_image()
# グレースケールに変換
convert_to_gray()
# 画像をぼかす
blur_image()
# 二値化
thresh()
# 輪郭を抽出
extract_contour()