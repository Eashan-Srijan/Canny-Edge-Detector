# Canny Edge Detector from Scratch

![image](https://user-images.githubusercontent.com/50113394/148710754-bc9e3a35-6268-44a5-b8ed-6f3e1a38229d.png)

## Introduction
The Canny edge detector is an edge detection operator that uses a multi-stage algorithm to detect a wide range of edges in images. In this project we will develop a Canny-Edge Detector from scratch, including functuionalities like convlution, non-max supression, simple thresholding(for threshold choosen at 25th, 50th and 75th percentile), gradient magnitude, edge angle and gradient angle computations. 

The process of Canny edge detection algorithm can be broken down to following steps:

1. Apply Gaussian filter to smooth the image in order to remove the noise.
2. Find the intensity gradients of the image using Prewitt's Operator.
3. Apply gradient magnitude thresholding or lower bound cut-off suppression to get rid of spurious response to edge detection (Non-max Supression).
4. Apply simple threshold to determine potential edges.

### Step 1: Gaussian Filtering

Gaussian filter is a filter whose impulse response is a Gaussian function (or an approximation to it, since a true Gaussian response would have infinite impulse response). Gaussian filters have the properties of having no overshoot to a step function input while minimizing the rise and fall time. It is considered the ideal time domain filter.

In the project we have used the center of the mask as the reference center. Also, if any part of the Gaussian mask (7 X 7) goes outside of the image border, we have let the output image be undefined at the location of the reference center.

<p align="center">
  <img src="https://user-images.githubusercontent.com/50113394/148710572-8b562018-3741-428b-886e-b6583e6094f1.png" />
</p>

<p align="center">
  <img src="https://user-images.githubusercontent.com/50113394/148710582-a635c938-cde7-4f7c-b1fd-ebb9df05a44c.png" />
</p>

### Step 2: Gradient Computaion

The finite differences are averaged over the 3 x 3 square so that the x and y partial derivatives are computed at the same point in the image. The magnitude and orientation of the gradient can be computed from the standard formulas for rectangular-to-polar conversion.

Here we have used the Prewitt’s operator to compute horizontal and vertical gradients. If any part of the 3 × 3 mask of the operator goes outside of the image border or lies in the undefined region of the image after Gaussian filtering, we have let the output value be undefined.

<p align="center">
  <img src="https://user-images.githubusercontent.com/50113394/148710590-a5799ae2-e316-44ab-8e81-5d6452a2b5fa.png" />
</p>

### Step 3: Non-max Supression

The magnitude image array M[i, j] will have large values where the image gradient is large, but this is not sufficient to identify the edges, since the problem of finding locations in the image array where there is rapid change has merely been transformed into the problem of finding locations in the magnitude array M[i, j] that are local maxima. Thin broad ridges n the magnitude array. 

•	Quantize angle of the gradient to one of the four sectors.
M[i,j] = Sector(Q[i,j])

<p align="center">
  <img src="https://user-images.githubusercontent.com/50113394/148710614-0a26f1d8-c348-4a83-8787-d07cb0c6a496.png" />
</p>

•	Thin magnitude image by using a 3X3 window.
N[i, j] = nms(M[i,j], e[i,j])

<p align="center">
  <img src="https://user-images.githubusercontent.com/50113394/148710629-f891605c-52c2-4455-bc82-fc5de3277c35.png" />
</p>

•	Suppress values along the line of gradient that are not peak values of a range.

•	If M[i,j] > magnitude of both neighbors along the line of the gradient given by sector values.

### Step 4: Simple Thresholding

The typical procedure used to reduce the number of false edge fragments in the non-maxima-suppressed gradient magnitude is to apply a threshold to N[i,j]. All values below the threshold are changed to zero. Here we have made use of 3 threshold values namely 25th percentile, 50th percentile and the 75th percentile. We have also excluded pixels with zero gradient magnitude when determining the percentiles

•	Simple Thresholding: - 
Use a single threshold T and keep edges with N[i,j] >= T and remove edges with N[i,j] < T.

## Outputs

1. Image 1

<p align="center">
  <img src="https://user-images.githubusercontent.com/50113394/148710719-b607a0c9-a225-4f50-9058-9b1cd390feaa.png" />
</p>

<p align="center">
  <img src="https://user-images.githubusercontent.com/50113394/148710686-72833326-f99c-4241-9ce0-91ed589559ee.png" />
</p>

2. Image 2

<p align="center">
  <img src="https://user-images.githubusercontent.com/50113394/148710733-7701e8b9-e085-48b4-b854-7be52c32f9b1.png" />
</p>

<p align="center">
  <img src="https://user-images.githubusercontent.com/50113394/148710709-fee142de-d1b7-4bcc-9c97-b543da8de385.png" />
</p>


