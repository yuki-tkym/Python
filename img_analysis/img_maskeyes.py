import matplotlib.pyplot as plt
import cv2

# cascasde検出器（目）を作成
cascade_file = "haarcascade_eye.xml"
cascade = cv2.CascadeClassifier(cascade_file)

# 読み込んだ画像→グレースケール変換
in_dir = "images/input"
out_dir = "images/output"
f_name = "tomoko_t"
f_type = "jpg"
# 認識する目の最小のサイズ
e_minsize = (20,20)
img = cv2.imread(in_dir + "/" + f_name + "." + f_type)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 目の認識実行
eyes_list = cascade.detectMultiScale(img_gray, minSize = e_minsize)

# 結果出力
if len(eyes_list) == 0:
    print("検出なし")
    quit()

# 目のマスキング
for (x, y, w, h) in eyes_list:
    print("目の座標（x座標,y座標,幅,高さ）：",x ,y ,w ,h)
    black = (0 ,0 ,0)
    cv2.rectangle(img, (x, y),(x + w, y + h), black,-1)

# マスキング済みの画像の出力
cv2.imwrite(out_dir + "/" + f_name +"_maskeyes.png", img)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

plt.show()
