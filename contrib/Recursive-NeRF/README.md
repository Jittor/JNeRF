# Recursive-NeRF: An Efficient and Dynamically Growing NeRF
This is official implementation of Recursive-NeRF: An Efficient and Dynamically Growing NeRF.

Paper link: https://ieeexplore.ieee.org/document/9909994

## Abstract
View synthesis methods using implicit continuous shape representations learned from a set of images, such as the Neural Radiance Field (NeRF) method, have gained increasing attention due to their high quality imagery and scalability to high resolution.
However, the heavy computation required by its volumetric approach prevents NeRF from being useful in practice; minutes are taken to render a single image of a few megapixels.
Now, an image of a scene can be rendered in a level-of-detail manner, so  we posit that a complicated region of the scene should be represented by a large neural network while a small neural network is capable of encoding a  simple region, enabling a balance between efficiency and quality. 
Recursive-NeRF is our embodiment of this idea, providing an efficient and adaptive rendering and training approach for NeRF.
The core of Recursive-NeRF  learns uncertainties for query coordinates, representing the quality of the predicted color and volumetric intensity at each level.
Only query coordinates with high uncertainties are forwarded to the next level to a bigger neural network with a more powerful representational  capability.
The final rendered image is a composition of results from neural networks of all levels.
Our evaluation on public datasets and a large-scale scene dataset we collected shows that Recursive-NeRF is more efficient than NeRF while providing state-of-the-art quality.