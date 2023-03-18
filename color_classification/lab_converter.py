from skimage import color
from colormath.color_diff import delta_e_cie2000
from colormath.color_objects import LabColor

### 색상간 거리를 계산하여 가장 비슷한(거리가 가까운) 색상명으로 맵핑해주는 프로그램
# rgb 대신 Lab 색공간 이용
class ColorConverter:

    def __init__(self, colors_path=''):
        # Lab list's format: L a b <color>정
        self.COLORS_FILE_PATH = colors_path
        # color_list[0]: 색상명
        # color_list[1]: Tuple(L, a, b)
        self.color_list = []

        # clab_list 생성
        self.init_color_list()

    def set_path(self, path):
        self.COLORS_FILE_PATH = path

    def init_color_list(self):
        f = open(self.COLORS_FILE_PATH, 'r')
        self.color_list = []
        while True:
            line = f.readline()
            if line == '':
                break

            lab_color = line.split(',')
            color = lab_color[0].strip()
            l = float(lab_color[1])
            a = float(lab_color[2])
            b = float(lab_color[3])

            self.color_list.append([color, (l, a, b)])

    def get_color_list(self):
        return self.color_list

    # Lab 색공간에서 두 색의 거리를 계산하여 가장 가까운 거리의 (미리 정의된)색상으로 맵핑해준다.
    # lab: 맵핑할 색상의 Lab값 -> Tuple(L,a,b)
    def rgb_to_name(self, rgb):
        r, g, b = rgb
        # print(r,g,b)
        r = r/255
        g = g/255
        b = b/255
        # print(r,g,b)
        rgb = [[[r, g, b]]]
        lab = color.rgb2lab(rgb)
        # print("lab")
        # print(lab)
        lab = tuple([lab[0][0][0], lab[0][0][1],lab[0][0][2]])
        return self.lab_to_name(lab)

    def lab_to_name(self, lab):
        min_d = 987654321
        index = -1

        l1, a1, b1 = lab
        color1 = LabColor(lab_l=l1, lab_a=a1, lab_b=b1)
        # 최솟값(min_d)과 최솟값이 위치하는 인덱스(index) 구함
        for i in range(len(self.color_list)):
            l2, a2, b2 = self.color_list[i][1]
            color2 = LabColor(lab_l=l2, lab_a=a2, lab_b=b2)
            d = delta_e_cie2000(color1, color2)
            # min_d =min(min_d, d)
            if d < min_d:
                min_d = d
                index = i
        # 가장 근접한 색상명 리턴
        return self.color_list[index][0]
