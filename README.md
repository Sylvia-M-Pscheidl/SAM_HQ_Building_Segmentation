# SAM-HQ-Building-Segmentation
We present a workflow for segmenting building footprints from a high resolution drone image in the city of Freetown, Sierra Leone, retraining SAM-HQ (Segment-Anything-Model High Quality) from Meta with drone imagery and OSM building polygons.

## Set Up

```
#clone official SAM-HQ repo 
git submodule add https://github.com/USER/REPO.git sam-hq

#install model from official repo
poetry add git+https://github.com/SysCV/sam-hq.git
```
