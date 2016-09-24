# behaviour_accelerometry
Code to run through the behaviour accelerometry machine learning models


things to consider Mon:

it seams that you are removing the following features:

"Acf.x", "Acf.y", "Acf.z", "Corr.xy", "Corr.yz", "Corr.xz"

If you are going to reprocess the data just remove them from the processing stage.

I believe that you want to only go forward with the origonal 50 or so features?
Then remove these from the processing stage, as there are NA's in these
