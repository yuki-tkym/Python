import cv2

# cascasde検出器（目）を作成
cascade_file = "haarcascade_eye.xml"
cascade = cv2.CascadeClassifier(cascade_file)

# 認識する目の最小のサイズ
e_minsize = (20,20)

# 画像取得
video_size = (500,300) # 画面サイズ
capture = cv2.VideoCapture(0)

# WMVファイル動画書き出し用のオブジェクト生成
format = cv2.VideoWriter_fourcc('W', 'M', 'V', '1')
fps = 20.0
size = (500,300)
writer = cv2.VideoWriter('videos/maskeyes.wmv',format,fps, size)

while True :
    _, frame = capture.read()
    frame = cv2.resize(frame,video_size)
    f_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 目の認識実行
    eyes_list = cascade.detectMultiScale(f_gray, minSize = e_minsize)
    # 目のマスキング
    for (x, y, w, h) in eyes_list:
        black = (0 ,0 ,0)
        cv2.rectangle(frame, (x, y),(x + w, y + h), black,-1)
    # 動画書き込み
    writer.write(frame)
    # マスキング済みの画像の出力
    cv2.imshow("MaskEyes Camera",frame)
    # Enterキークリックで終了
    if cv2.waitKey(1) == 13: break

capture.release()
cv2.destroyAllWindows()
