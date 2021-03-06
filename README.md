# Instance Normalization
This repo implements the Instance Normalization with different programming
languages or libraries: Tensorflow (Keras), Numpy, C++, Eigen, CUDA. Each sample
includes both the forward and backward passes and results are checked compared
with Keras's InstanceNormalization Instance.

## Instance Normalization Forward Pass
Suppose <img src="https://render.githubusercontent.com/render/math?math=N, C, D"> are batch, channel and feature dimensions respectively.

<img src="https://render.githubusercontent.com/render/math?math=x: (N, C, D)">

<img src="https://render.githubusercontent.com/render/math?math=\mu, \sigma ^2: (N, C, 1)">

<img src="https://render.githubusercontent.com/render/math?math=\gamma, \beta: (1, C, 1)">

---

<img src="https://render.githubusercontent.com/render/math?math=\mu = \frac{1}{D}\sum_{i=1}^{D}x_i">

<img src="https://render.githubusercontent.com/render/math?math=\sigma^2 = \frac{1}{D}\sum_{i=1}^{D}(x_i - \mu)^2">

[//]: <> (Comment: Use %2B to replace +.) 

<img src="https://render.githubusercontent.com/render/math?math=\hat{x} = \frac{x - \mu}{\sqrt{\sigma^2 %2B \epsilon}}">


<img src="https://render.githubusercontent.com/render/math?math=y=\gamma\hat{x} %2B \beta">

## Instance Normalization Backward Pass
<img src="https://render.githubusercontent.com/render/math?math=dy: (N, C, D)">

<img src="https://render.githubusercontent.com/render/math?math=d\mu, d\sigma ^2: (N, C, 1)">

<img src="https://render.githubusercontent.com/render/math?math=d\gamma, d\beta: (1, C, 1)">

---

<img src="https://render.githubusercontent.com/render/math?math=d\gamma=dy\cdot \frac{\partial y}{\partial \gamma}=\sum_{i=1}^{ND}dy\cdot\frac{x-\mu}{\sqrt{\sigma^2 %2B \epsilon}}">

<img src="https://render.githubusercontent.com/render/math?math=d\beta=dy\cdot \frac{\partial y}{\partial \beta}=\sum_{i=1}^{ND}dy">

Here we can view <img src="https://render.githubusercontent.com/render/math?math=y=\gamma\hat{x} %2B \beta"> as a composition function <img src="https://render.githubusercontent.com/render/math?math=y(i(x), \sigma^2(x), \mu(x))">, where <img src="https://render.githubusercontent.com/render/math?math=i(x)=x"> and <img src="https://render.githubusercontent.com/render/math?math=\sigma^2(x)">, <img src="https://render.githubusercontent.com/render/math?math=\mu(x)"> are defined as above. Therefore, we get:

<img src="https://render.githubusercontent.com/render/math?math=dx = \frac{\partial L}{\partial i}\cdot\frac{\partial i}{\partial x} %2B \frac{\partial L}{\partial \sigma^2}\cdot\frac{\partial \sigma^2}{\partial x} %2B \frac{\partial L}{\partial \mu}\cdot\frac{\partial \mu}{\partial x}">

* Compute <img src="https://render.githubusercontent.com/render/math?math=\frac{\partial L}{\partial i}\cdot\frac{\partial i}{\partial x}">

<img src="https://render.githubusercontent.com/render/math?math=\frac{\partial L}{\partial i} = \frac{\partial L}{\partial y}\cdot\frac{\partial y}{\partial i} = dy \cdot \gamma \cdot \frac{1}{\sqrt{\sigma^2 %2B \epsilon}}">

<img src="https://render.githubusercontent.com/render/math?math=\frac{\partial i}{\partial x}=1">

* Compute <img src="https://render.githubusercontent.com/render/math?math=\frac{\partial L}{\partial \sigma^2}\cdot\frac{\partial \sigma^2}{\partial x}">

<img src="https://render.githubusercontent.com/render/math?math=\frac{\partial L}{\partial \sigma^2} = \frac{\partial L}{\partial y}\cdot\frac{\partial y}{\partial \sigma^2} = \sum_{i=1}^{D}dy \cdot \gamma (x-\mu)(-\frac{1}{2})(\sigma^2 %2B \epsilon)^{-\frac{3}{2}}">

<img src="https://render.githubusercontent.com/render/math?math=\frac{\partial \sigma^2}{\partial x}=\frac{2}{D}(x-\mu)">

* Compute <img src="https://render.githubusercontent.com/render/math?math=\frac{\partial L}{\partial \mu}\cdot\frac{\partial \mu}{\partial x}">

<img src="https://render.githubusercontent.com/render/math?math=\frac{\partial L}{\partial \mu}=\frac{\partial L}{\partial \mu}\cdot \frac{\partial \mu}{\partial \mu} %2B \frac{\partial L}{\partial \sigma^2}\cdot \frac{\partial \sigma^2}{\partial \mu}\\=\frac{\partial L}{\partial y}\cdot \frac{\partial y}{\partial \mu}\cdot 1 %2B \frac{\partial L}{\partial \sigma^2}\cdot \frac{\partial \sigma^2}{\partial \mu}\\=\sum_{i=1}^{D}dy\cdot \gamma \cdot\frac{-1}{\sqrt{\sigma^2%2B\epsilon}}%2B\sum_{i=1}^{D}\frac{\partial L}{\partial \sigma^2}\cdot \frac{1}{D}(-2)(x-\mu)">, where the last term is 0

<img src="https://render.githubusercontent.com/render/math?math=\frac{\partial \mu}{\partial x}=\frac{1}{D}">
