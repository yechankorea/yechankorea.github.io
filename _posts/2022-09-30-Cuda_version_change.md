---
layout: single
title: "Cuda version change"
---

사용하고 싶은 cuda 다운받은 후에

윈도우+R

![image-20220930004052160](../images/2022-09-30-Cuda_version_change/image-20220930004052160.png)



고급 > 환경변수

![image-20220930004124366](../images/2022-09-30-Cuda_version_change/image-20220930004124366.png)





CUDA_PATE(편집) 에서 마지막, 원하는 버전으로 변경 예시 10.1 -> 11.7

![image-20220930004202164](../images/2022-09-30-Cuda_version_change/image-20220930004202164.png)

Path(편집) 원하는 버전 bin, libnvvp 맨위로 올리기

![image-20220930011135070](../images/2022-09-30-Cuda_version_change/image-20220930011135070.png)

파일 두개를 맨위로 이동.

프롬포트 에서 

'nvcc --version' 으로 cuda 버전확인.















