1. getHarr

extract viola-jones harr features from the images

2. getWeakClassifier

Select best weak classifier for one feature over all images

3. boost

boosting function given examples and weights, stop until certain decrease in error is met

4. cascade

stack layers of boosted classifiers together until error < maximum error tolerated

5. run

train the cascade classifier on training samples and apply to a graduation photo for face recogniton