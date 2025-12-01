import rasterio
from rasterio.mask import mask
import geopandas as gpd

# 1. Load vector and match CRS
gdf = gpd.read_file("input_files/learning_area_mask.gpkg")

with rasterio.open("input_files/freetown_droneimage.tif.crdownload") as src:
    gdf = gdf.to_crs(src.crs)

    # 2. Clip the raster
    # rasterio expects a list of geometries
    out_img, out_transform = mask(src, gdf.geometry, crop=True)
    out_meta = src.meta.copy()

# 3. Update metadata
out_meta.update({
    "driver": "GTiff",
    "height": out_img.shape[1],
    "width": out_img.shape[2],
    "transform": out_transform
})

# 4. Save to disk
with rasterio.open("input_files/droneimage_learning.tif", "w", **out_meta) as dest:
    dest.write(out_img)