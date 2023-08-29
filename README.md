# Image Augmentor

Image Augmentor is a project that uses cv2 to create augmentations of images for deep learning purposes. It can apply various filters and transformations to images, and also supports datasets with multiple classes and bounding boxes for object detection.

## Components

The project contains five main components:

- **Image Augmentation Filters**: This component applies filters to each image, such as blur, sharpen, noise, contrast, brightness, etc. You can choose from a list of predefined filters or create your own custom ones.
- **Simple Augmentor**: This component augments a dataset of images as much as you want, by randomly applying filters and transformations such as rotation, scaling, cropping, flipping, etc. This component does not have multi-class support, so it assumes that all images belong to the same class.
- **Multi-Class Augmentor**: This component augments a dataset of images with multiple classes, by randomly applying filters and transformations to each class separately. You need to provide a label file that specifies the class of each image in the dataset.
- **Bounding Box Augmentor**: This component augments a dataset of images with bounding boxes used for object detection, by randomly applying filters and transformations to both the images and the bounding boxes. This component does not support multiple classes, so it assumes that all images have only one bounding box.
- **Multi-Class Bounding Box Augmentor**: This component augments a dataset of images with multiple classes and bounding boxes used for object detection, by randomly applying filters and transformations to each class and bounding box separately. You need to provide a label file that specifies the class and the coordinates of each bounding box in the dataset.

## Usage

[Will add Later]

## Examples

Here are some examples of how the project can be used to augment different types of images:

### Image Augmentation Filters

[Will be added after project is finished]
### Simple Augmentor

[Will be added after project is finished]

### Multi-Class Augmentor

[Will be added after project is finished]

### Bounding Box Augmentor

[Will be added after project is finished]

### Multi-Class Bounding Box Augmentor

[Will be added after project is finished]

## License

This project is licensed under the MIT License - see the LICENSE file for details.