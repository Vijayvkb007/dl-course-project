# dl-course-project
## Models tried
1. Early Fusion
2. Late Fusion
## Model Implemented 
Early Fusion
We fuse the images together (rgb + ir) to 4-channeled images input which is sent to the model and then processed.
## Improvements which can be brought down
Image validation - RGB and IR should be of same image, if not the complete prediction is of no use.

## Bugs & Fixes

What if multiple users access it ? 

-> Queued and answered one after one

Fix:

Async calls in python!
