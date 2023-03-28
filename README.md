# Histogram-equalized

1. 히스토그램 평활화란?
어두운 영상의 히스토그램을 조절해 명암분포가 균일하게 되도록 만들어주는 것이다.
저대비 영상을 고대비 영상으로 바꾸기 위해 사용된다.

히스토그램 평활화 식은 

![평활화 식](https://user-images.githubusercontent.com/89963228/228205415-3fee933e-014e-436a-a1ac-aed529ad58b5.PNG)




2. 히스토그램 평활화 코드
* python으로 코드 작성하였다. (주피터 노트북 이용)
* 컬러영상을 hsi로 변환하여 평활화하였다.
* cv2는 이미지 불러올때만 사용하였다.

<p align="center"><img src="/test_image.jpg" width="300" height="300"/></center> </p>

<p align="center">test image</p>

<p align="center"><img src="/output_image.jpg" width="550" height="450"/></p>


<p align="center">output image</p>
