import os
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.features import rasterize
import geopandas as gpd
from shapely.geometry import mapping

# --- input files ---
drone_tif = "input_files/droneimage_learning.tif"
building_gpkg = "input_files/osm_buildings.gpkg"   # must be in same CRS as drone

# --- read drone image ---
src = rasterio.open(drone_tif)

# --- read building polygons ---
gdf = gpd.read_file(building_gpkg)
gdf = gdf.to_crs(src.crs)  # reproject if needed


# binary mask output file
mask_tif = "building_mask.tif"

# prepare rasterization shapes
shapes = [(mapping(geom), 1) for geom in gdf.geometry]

# rasterize polygons into array
mask_array = rasterize(
    shapes=shapes,
    out_shape=(src.height, src.width),
    transform=src.transform,
    fill=0,
    dtype="uint8"
)

# save mask aligned to drone tif
with rasterio.open(
    mask_tif, 'w',
    driver='GTiff',
    height=src.height,
    width=src.width,
    count=1,
    dtype="uint8",
    crs=src.crs,
    transform=src.transform
) as dst:
    dst.write(mask_array, 1)

print("Binary mask created:", mask_tif)



TILE_SIZE = 1024
OVERLAP = 128
STEP = TILE_SIZE - OVERLAP



def tile_raster_pair(
    image_path,
    mask_path,
    out_dir="tiles",
    tile_size=TILE_SIZE,
    step=STEP
):
    os.makedirs(f"{out_dir}/images", exist_ok=True)
    os.makedirs(f"{out_dir}/masks", exist_ok=True)

    with rasterio.open(image_path) as src_img, rasterio.open(mask_path) as src_mask:
        assert src_img.width == src_mask.width
        assert src_img.height == src_mask.height

        cols = src_img.width
        rows = src_img.height

        tile_id = 0

        # loop over grid
        for row in range(0, rows, step):
            if row + tile_size > rows:
                row = rows - tile_size  # shift to end
            for col in range(0, cols, step):
                if col + tile_size > cols:
                    col = cols - tile_size  # shift to end

                # define window
                window = Window(col_off=col, row_off=row, width=tile_size, height=tile_size)

                # read image + mask window
                img_tile = src_img.read(window=window)
                mask_tile = src_mask.read(1, window=window)

                # unique tile name
                tile_name = f"row_{row}_col_{col}"

                # --- save image tile ---
                out_img_path = f"{out_dir}/images/{tile_name}_image.tif"
                out_mask_path = f"{out_dir}/masks/{tile_name}_mask.tif"

                # tile's new transform
                transform = rasterio.windows.transform(window, src_img.transform)

                # save image tile
                with rasterio.open(
                    out_img_path, 'w',
                    driver='GTiff',
                    height=tile_size,
                    width=tile_size,
                    count=src_img.count,
                    dtype=src_img.dtypes[0],
                    crs=src_img.crs,
                    transform=transform
                ) as dst:
                    dst.write(img_tile)

                # save mask tile
                with rasterio.open(
                    out_mask_path, 'w',
                    driver='GTiff',
                    height=tile_size,
                    width=tile_size,
                    count=1,
                    dtype="uint8",
                    crs=src_mask.crs,
                    transform=transform
                ) as dst:
                    dst.write(mask_tile, 1)

                tile_id += 1

        print(f"Created {tile_id} tile pairs.")


tile_raster_pair(
    image_path=drone_tif,
    mask_path=mask_tif,
    out_dir="sam_tiles"
)




