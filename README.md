# Lane Detection

This project will implement the detection of straight and curved roads. It would be achieved by using various Computer Vision algorithms to process image and eventually lead to final output. We plan to detect lane/road in the images at initial phase and then apply it over the video.

## Abstract

In the most recent years the applications for Computer Vision have increased exponentially. One of the those applications is for a self-driving car. We propose an application that can be considered as a sub-function of a self-driving car - Lane Detection. This application can be used not only for a self-driving car but can also be integrated in some smart cars which detect lanes for driver safety and warns the driver when the car leaves a specific lane while on a highway or on a route commonly known as Lane Departure Warning(LDW) and Advanced Driver-assistance System (ADAS). The methods used to achieve the results are purely based on Computer Vision algorithms, such as Canny edge detection and Sliding Window protocol which will be discussed in detail further.

Before starting off with the implementation of the project we would like to inform the readers that the system currently cannot be implemented on real-time data because of obvious computation time. This system takes about 1 second of computation for each frame in a video. Therefore, the input video is to be passed as an argument to the system and then after traversing through all the frames of the video, the system will generate an another video as an output which will contain the lane mask.

## Libraries Used

```
- cv2
- numpy
- pickle
- glob
- argparse
```

## Built With

* [Google Colab](https://colab.research.google.com) - The platform to run the code
* [Spyder](https://www.spyder-ide.org/) - IDE - To code the project

## Report Paper

Please feel free to refer to the [Report](https://umbc.box.com/s/u5vayh4m824x9muwqde33j9dg2rdu2iv) to learn more about the algorithms of Computer Vision, how they can be used in real life situations, and view the results obtained from the project.
