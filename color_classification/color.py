from calendar import c
from tkinter import Y
import numpy as np
import cv2
from sklearn.cluster import KMeans
import sys
sys.path.append('/user/app/color_classification')
from lab_converter import ColorConverter



def execute(img_path, q):
    converter = ColorConverter('/user/app/color_classification/final_color_list_demo.txt')
    # 이미지 로드
    # image = cv2.imread(img_path)  # test image
    print(img_path)
    image = cv2.imread(img_path)  # test image

    # 배경색 제거를 위한 이미지 자르기
    # print(image.shape)
    h, w, _ = image.shape # _ ->채널
    image = image.copy() # 원본 소스방해 x  
    image = image[int(h*0.2) : int(h*0.8), int(w*0.2) : int(w*0.8)] # [시작 height : 끝 height, 시작 width : 끝 width]


    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #opencv에서는 BRG matpolotlib에서는 rgb순서
    # height와 width를 통합하여 하나의 array로 만듬
    image = image.reshape((image.shape[0] * image.shape[1], 3))
    # print(image.shape)
    # print(image)
    # print(image[1])

    ##### k-mean 알고리즘(비지도 학습에서 가장 보편적인 알고리즘)으로 이미지 학습 ######
    # k개의 데이터 평균을 만들어 데이터를 클러스팅하는 알고리즘
    k = 5 # 추출할 클러스터 개수
    # k-mean 알고리즘을 이용한 비지도 학습 실행
    clt = KMeans(n_clusters=k).fit(image)
    # print(type(clt))
    # 생성한 k개의 cluster 각각의 중심값이 담긴 리스트
    centers = clt.cluster_centers_
    # print("centers:")
    # print(centers)

    # 클러스터로 추출한 컬러가 (k개의 클러스터 중) 차지하는 비율을 반환
    def get_weights(clt):
        # grab the number of different clusters and create a histogram
        # based on the number of pixels assigned to each cluster
        # print(clt.labels_)
        numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
        # print("numlabels:")
        # print(numLabels)
        # print("clt.labels:")
        # print(clt.labels_)
        (hist, _) = np.histogram(clt.labels_, bins=numLabels)

        # normalize the histogram, such that it sums to one
        histogram = hist.astype("float")
        histogram /= hist.sum()

        # return the histogram
        return histogram


    # 각 클러스터의 가중치로 사용할 것임
    # W의 모든 값을 더하면 1임
    W = get_weights(clt)
    # print("W")
    # print(W)
    # W를 내림차순으로 정렬하기 위해 {index:W} 딕셔너리를 만듬
    w_dict = {}
    for w in range(k):
        w_dict[w] = W[w]

    # 내림차순 정렬
    w_dict = sorted(w_dict.items(), reverse=True, key=lambda item: item[1])

    # baseline% 이상 차지하는 색상이 있을 경우 -> 해당 색상 하나만 출력
    # 그 외 -> 상위 2개의 색상 모두 출
    # ([r,g,b], W) 튜플의 리스트를 리턴함
    def get_dominent_rgb(baseline):
        # w_dict[0]은 가장 비율이 높은 색의 튜플, (해당 클러스터의 인덱스, 해당 클러스터의 비율)임
        clt_index, w = w_dict[0]

        rgb_tuple_list = [(centers[clt_index], w)]
        # 가장 큰 비율이 baseline 이상이면 하나의 색만 리턴
        if w >= baseline:
            # rgb_tuple_list에는 하나의 ([r,g,b], w) 튜플만 들어있음
            return rgb_tuple_list

        # baseline 이상인 것이 없으면 상위 2개의 색 리턴
        clt_index, w = w_dict[1]
        rgb_tuple_list.append((centers[clt_index], w))

        return rgb_tuple_list

    # 색상의 평균값 출력
    # 0.6(=baseline) 이상 차지하는 색상이 있으면 그 색상 하나만 리턴
    # --> 0.4로 변경
    # rgb_tuple_list = get_dominent_rgb(0.6)
    rgb_tuple_list = get_dominent_rgb(0.7)

    rgb, w1 = rgb_tuple_list[0]
    color_name1 = converter.rgb_to_name(rgb)
    ret_string = color_name1 + '색 '
    if len(rgb_tuple_list) == 1:
        q.put(ret_string + str(int( (round(w1, 2) * 100) )) + '퍼, ')
        return q

    else:
        rgb, w2 = rgb_tuple_list[1]
        color_name2 = converter.rgb_to_name(rgb)
        if color_name1 == color_name2:
            w = w1 + w2
            ret_string = ret_string + str(int( (round(w, 2) * 100) )) + '퍼, '
        else:
            ret_string = ret_string + str(int( (round(w1, 2) * 100) )) + '퍼, ' + color_name2 + '색 ' + str(int( (round(w2, 2) * 100) )) + '퍼, '

        q.put(ret_string)
        return q
