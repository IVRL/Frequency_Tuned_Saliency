import numpy as np
from PIL import Image


def srgb_to_cielab(sR, sG, sB):

	#------------------------
	# sRGB to XYZ conversion
	#------------------------
	R = sR.astype(np.float64)/255.0
	G = sG.astype(np.float64)/255.0
	B = sB.astype(np.float64)/255.0

	maskr = (R <= 0.04045)
	maskg = (G <= 0.04045)
	maskb = (B <= 0.04045)

	r, g, b = np.zeros_like(R), np.zeros_like(G), np.zeros_like(B)

	r[maskr] = R[maskr]/12.92
	g[maskg] = G[maskg]/12.92
	b[maskb] = B[maskb]/12.92

	r[~maskr] = np.power((R[~maskr]+0.055)/1.055,2.4)
	g[~maskg] = np.power((G[~maskg]+0.055)/1.055,2.4)
	b[~maskb] = np.power((B[~maskb]+0.055)/1.055,2.4)

	X = r*0.4124564 + g*0.3575761 + b*0.1804375
	Y = r*0.2126729 + g*0.7151522 + b*0.0721750
	Z = r*0.0193339 + g*0.1191920 + b*0.9503041

	#------------------------
	# XYZ to LAB conversion
	#------------------------
	epsilon = 0.008856	# actual CIE standard
	kappa   = 903.3		# actual CIE standard

	Xr = 0.950456	# reference white
	Yr = 1.0		# reference white
	Zr = 1.088754	# reference white

	xr = X/Xr
	yr = Y/Yr
	zr = Z/Zr

	maskx = (xr > epsilon)
	masky = (yr > epsilon)
	maskz = (zr > epsilon)

	fx, fy, fz = np.zeros_like(xr), np.zeros_like(yr), np.zeros_like(zr)

	fx[maskx] = np.power(xr[maskx], 1.0/3.0)
	fy[masky] = np.power(yr[masky], 1.0/3.0)
	fz[maskz] = np.power(zr[maskz], 1.0/3.0)

	fx[~maskx] = (kappa*xr[~maskx] + 16.0)/116.0
	fy[~masky] = (kappa*yr[~masky] + 16.0)/116.0
	fz[~maskz] = (kappa*zr[~maskz] + 16.0)/116.0

	lvals = 116.0*fy-16.0;
	avals = 500.0*(fx-fy);
	bvals = 200.0*(fy-fz);

	return lvals, avals, bvals


def compute_saliency_map(srgb, sigma=0):

	

	if sigma > 0:
		from scipy.ndimage import gaussian_filter
		srgb = gaussian_filter(srgb, sigma = (sigma, sigma, 0))

	sr, sg, sb = srgb[:,:,0], srgb[:,:,1], srgb[:,:,2]
	lvals, avals, bvals = srgb_to_cielab(sr, sg, sb)
	lmean, amean, bmean = lvals.mean(), avals.mean(), bvals.mean()
	salmap = (lvals-lmean)**2 + (avals-amean)**2 + (bvals-bmean)**2
	salimg = (255*salmap/salmap.max() + 0.5).astype(np.uint8) # values of salmap scaled to [0,255]
	salimg = np.stack([salimg, salimg, salimg]).transpose(1,2,0)

	return salmap, salimg



if __name__ == "__main__":

	filepath = './bee.png'
	img = Image.open(filepath).convert("RGB")
	img = np.asarray(img) # shape is H,W,C

	salmap, salimg = compute_saliency_map(img, sigma=0)

	combo = np.concatenate([img,salimg], axis=1)
	Image.fromarray(combo).save('./bee_sal.png')



