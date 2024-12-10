import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# 1. Génération du profil NACA 0012
def naca4_profile(m, p, t, num_points=150):
    # Distribution en cosinus pour une meilleure résolution au bord d'attaque
    beta = np.linspace(0, np.pi, num_points)
    x = (1 - np.cos(beta)) / 2

    yt = 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x -
                  0.3516 * x**2 + 0.2843 * x**3 - 0.1015 * x**4)
    
    yc = np.zeros_like(x)
    dyc_dx = np.zeros_like(x)
    
    if p != 0 and m != 0:
        idx1 = x < p
        idx2 = x >= p
        yc[idx1] = m / p**2 * (2 * p * x[idx1] - x[idx1]**2)
        yc[idx2] = m / (1 - p)**2 * ((1 - 2 * p) + 2 * p * x[idx2] - x[idx2]**2)
        dyc_dx[idx1] = 2 * m / p**2 * (p - x[idx1])
        dyc_dx[idx2] = 2 * m / (1 - p)**2 * (p - x[idx2])
    else:
        pass  # yc et dyc_dx restent à zéro pour un profil symétrique

    theta = np.arctan(dyc_dx)
    
    xu = x - yt * np.sin(theta)
    yu = yc + yt * np.cos(theta)
    
    xl = x + yt * np.sin(theta)
    yl = yc - yt * np.cos(theta)
    
    x_coords = np.concatenate([xu, xl[::-1]])
    y_coords = np.concatenate([yu, yl[::-1]])
    
    return x_coords, y_coords


# 2. Définition des panneaux
class Panel:
    def __init__(self, xa, ya, xb, yb):
        self.xa, self.ya = xa, ya  # Point initial du panneau
        self.xb, self.yb = xb, yb  # Point final du panneau
        self.xc = (xa + xb) / 2    # Coordonnée x du centre du panneau
        self.yc = (ya + yb) / 2    # Coordonnée y du centre du panneau
        self.length = np.hypot(xb - xa, yb - ya)  # Longueur du panneau

        # Orientation du panneau
        self.beta = np.arctan2(yb - ya, xb - xa)
        self.sigma = 0.0  # Intensité de la source
        self.vt = 0.0     # Vitesse tangentielle
        self.cp = 0.0     # Coefficient de pression

def define_panels(x, y):
    num_panels = len(x) - 1
    panels = np.empty(num_panels, dtype=object)
    for i in range(num_panels):
        panels[i] = Panel(x[i], y[i], x[i+1], y[i+1])
    return panels


# 3. Définition de l'écoulement libre
class Freestream:
    def __init__(self, u_inf=1.0, alpha=0.0):
        self.u_inf = u_inf
        self.alpha = np.radians(alpha)

# 4. Construction de la matrice du système linéaire
def source_contribution_normal(panels):
    num_panels = len(panels)
    A = np.empty((num_panels, num_panels), dtype=float)
    for i, pi in enumerate(panels):
        for j, pj in enumerate(panels):
            if i != j:
                A[i, j] = 0.5 / np.pi * integral_normal(pi, pj)
            else:
                A[i, j] = 0.5
    return A

def integral_normal(pi, pj):
    def integrand(s):
        x = pj.xa - pi.xc - np.sin(pj.beta) * s
        y = pj.ya - pi.yc + np.cos(pj.beta) * s
        return (x * np.cos(pi.beta) + y * np.sin(pi.beta)) / (x**2 + y**2)
    return np.trapz(integrand(np.linspace(0, pj.length, 100)), dx=pj.length/100)

# 5. Construction du vecteur du second membre
def build_rhs(panels, freestream):
    b = np.empty(len(panels), dtype=float)
    for i, panel in enumerate(panels):
        b[i] = -freestream.u_inf * np.cos(freestream.alpha - panel.beta)
    return b


# 7. Calcul des vitesses tangentielles et du coefficient de pression
def get_tangential_velocity(panels, freestream):
    num_panels = len(panels)
    A = np.zeros((num_panels, num_panels), dtype=float)
    for i, pi in enumerate(panels):
        for j, pj in enumerate(panels):
            if i != j:
                A[i, j] = 0.5 / np.pi * integral_tangential(pi, pj)
    b = freestream.u_inf * np.sin([freestream.alpha - panel.beta for panel in panels])
    sigma = np.array([panel.sigma for panel in panels])
    vt = np.dot(A, sigma) + b
    for i, panel in enumerate(panels):
        panel.vt = vt[i]

def integral_tangential(pi, pj):
    def integrand(s):
        x = pj.xa - pi.xc - np.sin(pj.beta) * s
        y = pj.ya - pi.yc + np.cos(pj.beta) * s
        return (y * np.cos(pi.beta) - x * np.sin(pi.beta)) / (x**2 + y**2)
    return np.trapz(integrand(np.linspace(0, pj.length, 100)), dx=pj.length/100)

# 9. Masquage des points à l'intérieur du profil
def in_airfoil(x, y, panels):
    from matplotlib.path import Path
    vertices = [(panel.xa, panel.ya) for panel in panels]
    vertices.append((panels[0].xa, panels[0].ya))  # fermer le contour
    path = Path(vertices)
    points = np.vstack((x.flatten(), y.flatten())).T
    mask = path.contains_points(points)
    return mask.reshape(x.shape)