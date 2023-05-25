'''
this code is adapted from https://github.com/abdallahdib/NextFace/blob/main/sphericalharmonics.py
'''


import math
import torch


def associated_legendre_polynomial(l, m, x):
    pmm = torch.ones_like(x)
    if m > 0:
        somx2 = torch.sqrt((1 - x) * (1 + x) + 1e-6)
        fact = 1.0
        for i in range(1, m + 1):
            pmm = pmm * (-fact) * somx2
            fact += 2.0
    if l == m:
        return pmm
    pmmp1 = x * (2.0 * m + 1.0) * pmm
    if l == m + 1:
        return pmmp1
    pll = torch.zeros_like(x)
    for ll in range(m + 2, l + 1):
        pll = ((2.0 * ll - 1.0) * x * pmmp1 - (ll + m - 1.0) * pmm) / (ll - m)
        pmm = pmmp1
        pmmp1 = pll
    return pll


def normlizeSH(l, m):
    return math.sqrt((2.0 * l + 1.0) * math.factorial(l - m) / (4 * math.pi * math.factorial(l + m)))


def SH(l, m, x, y, z):
    '''
    NOTE actually we can directly compute phi from x and y using atan2,
    but we find the atan2 function leads to unstable gradient when the geometry is needed to be optimized
    thus we do not compute phi explicitly, only compute sin(phi) and cos(phi)
    '''
    z = torch.clamp(z, min=-0.99, max=0.99)
    cos_theta = z
    theta = torch.acos(cos_theta)
    sin_theta = torch.sin(theta)
    sin = y / (sin_theta + 1e-6)  # sin(phi)
    cos = x / (sin_theta + 1e-6)  # cos(phi)
    sin = torch.clamp(sin, min=-1., max=1.)
    cos = torch.clamp(cos, min=-1., max=1.)
    if m == 0:
        return normlizeSH(l, m) * associated_legendre_polynomial(l, m, z)
    elif m > 0:
        return math.sqrt(2.0) * normlizeSH(l, m) * \
                cos_nx(m, sin, cos) * associated_legendre_polynomial(l, m, z)
                # torch.cos(m * phi) * associated_legendre_polynomial(l, m, torch.cos(theta))
    else:
        return math.sqrt(2.0) * normlizeSH(l, -m) * \
                sin_nx(-m, sin, cos) * associated_legendre_polynomial(l, -m, z)
                # torch.sin(-m * phi) * associated_legendre_polynomial(l, -m, torch.cos(theta))


def SH_sphere(l, m, phi, theta):
    if m == 0:
        return normlizeSH(l, m) * associated_legendre_polynomial(l, m, torch.cos(theta))
    elif m > 0:
        return math.sqrt(2.0) * normlizeSH(l, m) * \
                torch.cos(m * phi) * associated_legendre_polynomial(l, m, torch.cos(theta))
    else:
        return math.sqrt(2.0) * normlizeSH(l, -m) * \
                torch.sin(-m * phi) * associated_legendre_polynomial(l, -m, torch.cos(theta))


def cnk(n, k):
    return math.factorial(n) / (math.factorial(k) * math.factorial(n - k))


def sin_nx(n, s, c):
    '''
    use only sin(x) and cos(x) to compute sin(nx)
    '''
    odd = list(range(1, n + 1, 2))
    res = torch.zeros_like(s)
    for i in odd:
        res += ((-1) ** ((i - 1) / 2)) * (s ** i) * (c ** (n - i)) * cnk(n, i)
    return res


def cos_nx(n, s, c):
    '''
    use only sin(x) and cos(x) to compute cos(nx)
    '''
    even = list(range(0, n + 1, 2))
    res = torch.zeros_like(s)
    for i in even:
        res += ((-1) ** (i / 2)) * (s ** i) * (c ** (n - i)) * cnk(n, i)
    return res


def sh_to_envmap(coeffs, h=64, w=64):
    '''
    input: coeffs: with size of [b,3,n]
    output: pm_sh: [b,3,h,w]
    '''    
    order = int(math.sqrt(coeffs.shape[-1]))
    theta = torch.linspace(0, math.pi, h).to(coeffs.device)  # [h] from 0 to pi
    phi = torch.linspace(0, 2 * math.pi, w).to(coeffs.device)  # [w] from 0 to 2pi
    theta = theta[..., None].repeat(1, w)  # [h,w]
    phi = phi[None, ...].repeat(h, 1)  # [h,w]
    
    x = torch.sin(theta) * torch.cos(phi)
    y = torch.sin(theta) * torch.sin(phi)
    z = torch.cos(theta)
    sh_basis = []
    for l in range(order):
        for m in range(-l, l + 1):
            sh_basis.append(SH(l, m, x, y, z))
    sh_basis = torch.stack(sh_basis, dim=-1)  # [h,w,n]

    # get pm represented by sh
    coeffs_ = coeffs[:, :, None, None, :]  # [b,3,1,1,n]
    pm_sh = torch.sum(coeffs_ * sh_basis, dim=-1)
    
    return pm_sh
