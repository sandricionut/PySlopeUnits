from __future__ import annotations
from pathlib import Path
import numpy as np
import rasterio
from rasterio.windows import Window


def _hash_u32(x):
    z=x.astype(np.uint32,copy=False); z=z^(z>>np.uint32(16)); z=z*np.uint32(0x7FEB352D); z=z^(z>>np.uint32(15)); z=z*np.uint32(0x846CA68B); z=z^(z>>np.uint32(16)); return z


def write_hashed_display_raster(input_raster, output_raster, *, classes=255):
    input_raster=Path(input_raster); output_raster=Path(output_raster); output_raster.parent.mkdir(parents=True,exist_ok=True)
    dtype='uint8' if classes<=255 else 'uint16'
    with rasterio.open(input_raster) as src:
        profile=src.profile.copy()
        profile.pop('blockxsize', None); profile.pop('blockysize', None)
        profile.update(dtype=dtype,count=1,nodata=0,compress='DEFLATE',predictor=2,BIGTIFF='IF_SAFER')
        if src.width >= 16 and src.height >= 16:
            bx=max(16, min(512, (src.width//16)*16)); by=max(16, min(512, (src.height//16)*16))
            profile.update(tiled=True, blockxsize=bx, blockysize=by)
        else:
            profile.update(tiled=False)
        with rasterio.open(output_raster,'w',**profile) as dst:
            for _,win in src.block_windows(1):
                a=src.read(1,window=win); out=np.zeros(a.shape,dtype=np.uint16); good=a>0
                if np.any(good): out[good]=1+(_hash_u32(a[good].astype(np.uint32))%np.uint32(classes))
                dst.write(out.astype(dtype),1,window=win)
    return output_raster


def write_boundary_raster(input_raster, output_raster, *, include_nodata_edge=False):
    input_raster=Path(input_raster); output_raster=Path(output_raster); output_raster.parent.mkdir(parents=True,exist_ok=True)
    with rasterio.open(input_raster) as src:
        profile=src.profile.copy()
        profile.pop('blockxsize', None); profile.pop('blockysize', None)
        profile.update(dtype='uint8',count=1,nodata=0,compress='DEFLATE',predictor=2,BIGTIFF='IF_SAFER')
        if src.width >= 16 and src.height >= 16:
            bx=max(16, min(512, (src.width//16)*16)); by=max(16, min(512, (src.height//16)*16))
            profile.update(tiled=True, blockxsize=bx, blockysize=by)
        else:
            profile.update(tiled=False)
        with rasterio.open(output_raster,'w',**profile) as dst:
            for _,win in src.block_windows(1):
                r0=int(win.row_off); c0=int(win.col_off); h=int(win.height); w=int(win.width)
                # Always request a one-cell halo. Rasterio fills outside-raster
                # locations with zero, so the four neighbor arrays have the
                # same shape for edge and interior blocks.
                halo=src.read(
                    1,
                    window=Window(c0-1,r0-1,w+2,h+2),
                    boundless=True,
                    fill_value=0,
                )
                center=halo[1:h+1,1:w+1]
                good=center>0
                out=np.zeros(center.shape,dtype=np.uint8)
                neighbors=(
                    halo[0:h,1:w+1],
                    halo[2:h+2,1:w+1],
                    halo[1:h+1,0:w],
                    halo[1:h+1,2:w+2],
                )
                for nbr in neighbors:
                    if include_nodata_edge:
                        diff=good & (nbr!=center)
                    else:
                        diff=good & (nbr>0) & (nbr!=center)
                    out[diff]=1
                dst.write(out,1,window=win)
    return output_raster
