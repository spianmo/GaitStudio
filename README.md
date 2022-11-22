# GaitStudio

> 基于Kinect深度相机和MediaPipe实时推理人体各部分关节活动度，并输出PDF检测报告。
>
> Based on the Kinect depth camera and MediaPipe, real-time reasoning of the joint activity of each part of the human body, and output a detection report.

定义了一套检测规则DSL，可扩展多种检测项目。检测速率在15FPS左右，支持深度视图、RGB视图以及叠加视图的切换显示。实时展示各部位关节角度，自由调节检测BlazePose模型精度和Kinect相机参数。

**TODO:**

- [ ] 根据关节角度Detect步态相位，目前使用空间点分解加速度的方式，通过加速度滤波得出周期性的步态相位，数据来源于CV推理，不比IMU。

**技术栈：**

Python3, PySide2, MediaPipe, QT

**截图：**

![image](https://user-images.githubusercontent.com/18194268/203285437-5f7f7bdc-c197-447f-89b5-171a09473329.png)

![image](https://user-images.githubusercontent.com/18194268/203286216-0628a45d-5286-441d-b16b-0ba6b1c8aebd.png)

![image](https://user-images.githubusercontent.com/18194268/203286444-0592364b-3ebd-4152-8d46-721fa87c2fe1.png)



![6a87ea890718ca79c54a0da4ea74730](https://user-images.githubusercontent.com/18194268/203284879-9cf87e31-245a-41a2-bd1b-9a5c68371f84.jpg)
