# Sight Seer
This project uses a residual network[^1] to find a location from a single photograph. Model architecture is a ResNet18 model. Across the United States, the model achieved an accuracy of 16.26% across 55 classes.

## Dataset Details
The training dataset is comprised of twenty thousand images sourced from Google Street View. Images are broken into classes based on their geographical location, and are binned into 5 degree by 5 degree squares.


[^1]: [Wikipedia: Residual Neural Networks](https://en.wikipedia.org/wiki/Residual_neural_network)