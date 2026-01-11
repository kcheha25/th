pip install spectral numpy matplotlib scikit-learn

import spectral as sp

img = sp.open_image("image.hdr")
cube = img.load()   # (lines, samples, bands)

print(cube.shape)

wavelengths = img.bands.centers
print(wavelengths[:10])

import matplotlib.pyplot as plt

band_idx = 100
plt.imshow(cube[:, :, band_idx], cmap='gray')
plt.title(f"{wavelengths[band_idx]:.1f} nm")
plt.colorbar()
plt.show()

dark = sp.open_image("dark.hdr").load()
white = sp.open_image("white.hdr").load()

cube_ref = (cube - dark) / (white - dark)
cube_ref = cube_ref.clip(0, 1)

y, x = 200, 300
spectrum = cube_ref[y, x, :]

plt.plot(wavelengths, spectrum)
plt.xlabel("Wavelength (nm)")
plt.ylabel("Reflectance")
plt.show()

roi = cube_ref[200:250, 300:350, :]
mean_spectrum = roi.mean(axis=(0, 1))

from sklearn.decomposition import PCA

h, w, b = cube_ref.shape
X = cube_ref.reshape(-1, b)

pca = PCA(n_components=3)
X_pca = pca.fit_transform(X)

pca_img = X_pca.reshape(h, w, 3)
sp.imshow(pca_img)


############################################3
# pip install specarray dask matplotlib scikit-learn
from specarray import SpecArray
import dask.array as da
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


# Suppose que le dossier contient image.hdr + image.raw + dark/white
data_dir = "chemin/vers/ton/dossier"

# Charge le cube Specim en utilisant Dask pour gros fichiers
hsidata = SpecArray.from_folder(data_dir, chunks=(100, 100, -1))  
# chunks=(lines, samples, bands) ; -1 signifie pas de découpage sur les bandes

cube = hsidata.capture.data  # ceci est un dask.array (lazy loading)
print(cube.shape)

wavelengths = hsidata.wavelengths.data
print(wavelengths[:10])


dark_cube = hsidata.dark.data
white_cube = hsidata.white.data

# Réflectance spectrale
cube_ref = (cube - dark_cube) / (white_cube - dark_cube)
cube_ref = da.clip(cube_ref, 0, 1)  # clip entre 0 et 1


band_idx = 100
plt.imshow(cube_ref[:, :, band_idx].compute(), cmap='gray')
plt.title(f"{wavelengths[band_idx]:.1f} nm")
plt.colorbar()
plt.show()


y, x = 200, 300
spectrum = cube_ref[y, x, :].compute()

plt.plot(wavelengths, spectrum)
plt.xlabel("Wavelength (nm)")
plt.ylabel("Reflectance")
plt.show()

roi = cube_ref[200:250, 300:350, :]
mean_spectrum = roi.mean(axis=(0,1)).compute()
plt.plot(wavelengths, mean_spectrum)
plt.title("Mean Spectrum ROI")
plt.show()


h, w, b = cube_ref.shape
X = cube_ref.reshape((h*w, b))

# Attention : Dask n'est pas compatible directement avec scikit-learn
# On convertit en numpy (peut être lent pour très gros cubes)
X_np = X.compute()

pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_np)

pca_img = X_pca.reshape(h, w, 3)
plt.imshow(pca_img / pca_img.max())
plt.title("PCA 3 composants")
plt.show()
##############################################

pip install hylite
pip install rasterio

from hylite import io
import matplotlib.pyplot as plt
import numpy as np

file_path = "chemin/vers/ton_cube_pushbroom.hdr"
cube = io.load(file_path)

print(cube.shape)
print(cube.wl[:10])

cube.quick_plot(title="Faux-RGB du cube Pushbroom")
band_idx = 50
cube.quick_plot(bands=[band_idx], title=f"Bande {band_idx} - {cube.wl[band_idx]:.1f} nm")

y, x = 100, 150
spectrum = cube.data[y, x, :]
plt.plot(cube.wl, spectrum)
plt.xlabel("Longueur d'onde (nm)")
plt.ylabel("Reflectance")
plt.title(f"Spectre pixel ({y},{x})")
plt.show()

roi = cube.data[100:120, 150:170, :]
mean_spectrum = np.mean(roi, axis=(0,1))
plt.plot(cube.wl, mean_spectrum)
plt.xlabel("Longueur d'onde (nm)")
plt.ylabel("Reflectance")
plt.title("Spectre moyen ROI")
plt.show()
