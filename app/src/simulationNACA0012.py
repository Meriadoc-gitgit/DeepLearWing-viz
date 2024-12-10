import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path
from matplotlib.animation import FuncAnimation

import os

def naca0012(num_points=150):
    """Génère les coordonnées du profil NACA 0012."""
    c = 1.0  # longueur de corde
    t = 0.12  # épaisseur maximale (12% de la corde)

    x = np.linspace(0, c, num_points)

    # Épaisseur
    yt = 5 * t * c * (0.2969 * np.sqrt(x/c) - 0.1260*(x/c) - 0.3516*(x/c)**2 + 0.2843*(x/c)**3 - 0.1036*(x/c)**4)

    # Surfaces supérieure et inférieure
    xu = x
    yu = yt
    xl = x
    yl = -yt

    # Combiner les surfaces
    x_coords = np.concatenate([xu, xl[::-1]])
    y_coords = np.concatenate([yu, yl[::-1]])

    return x_coords, y_coords

def map_airfoil_to_grid(x_coords, y_coords, Nx, Ny):
    """Mappe les coordonnées du profil sur la grille."""
    chord_length_in_grid = Nx / 1.5 # Ajuster selon la taille souhaitée
    x_grid = x_coords * chord_length_in_grid + Nx / 4  # Positionner l'airfoil dans la grille
    y_grid = y_coords * chord_length_in_grid + Ny / 2  # Centrer verticalement
    return x_grid, y_grid

def create_airfoil_mask(x_grid, y_grid, Nx, Ny):
    """Crée un masque de la forme du profil sur la grille."""
    vertices = np.column_stack((x_grid, y_grid))
    airfoil_path = Path(vertices)
    X, Y = np.meshgrid(np.arange(Nx), np.arange(Ny))
    points = np.vstack((X.flatten(), Y.flatten())).T
    airfoil_mask = airfoil_path.contains_points(points).reshape(Ny, Nx)
    return airfoil_mask

def main():
    """Simulation Lattice Boltzmann"""

    Nx = 1000  # Increase from 800
    Ny = 400   # Increase from 200

    rho0 = 100    # densité moyenne
    tau = 0.6    # temps de collision
    Nt = 500    # nombre de pas de temps

    NL = 9
    idxs = np.arange(NL)
    cxs = np.array([0, 0, 1, 1, 1, 0, -1, -1, -1])
    cys = np.array([0, 1, 1, 0, -1, -1, -1, 0, 1])
    weights = np.array([4/9, 1/9, 1/36, 1/9, 1/36, 1/9, 1/36, 1/9, 1/36])

    F = np.ones((Ny, Nx, NL)) * rho0 / NL
    X, Y = np.meshgrid(range(Nx), range(Ny))
    F += 0.01 * np.random.randn(Ny, Nx, NL)
    F[:, :, 3] += 5 * (1 + 0.2 * np.cos(2 * np.pi * X / Nx * 4))  # flux initial augmenté
    rho = np.sum(F, axis=2)
    for i in idxs:
        F[:, :, i] *= rho0 / rho

    x_coords, y_coords = naca0012(num_points=150)
    x_grid, y_grid = map_airfoil_to_grid(x_coords, y_coords, Nx, Ny)
    airfoil_mask = create_airfoil_mask(x_grid, y_grid, Nx, Ny)

    fig, ax = plt.subplots(figsize=(6, 3), dpi=80)

    vorticity_frames = []

    for it in range(Nt):
        print(f"Step {it}/{Nt}")

        for i, cx, cy in zip(idxs, cxs, cys):
            F[:, :, i] = np.roll(F[:, :, i], cx, axis=1)
            F[:, :, i] = np.roll(F[:, :, i], cy, axis=0)

        for i in idxs:
            F[:, -1, i] = F[:, -2, i]  # bord droit ouvert

        F[:, 0, 3] = rho0 * weights[3] * (1 + 3 * cxs[3] * 0.1)  # flux entrant

        bndryF = F[airfoil_mask, :]
        bndryF = bndryF[:, [0, 5, 6, 7, 8, 1, 2, 3, 4]]
        F[airfoil_mask, :] = bndryF

        rho = np.sum(F, axis=2)
        ux = np.sum(F * cxs, axis=2) / rho
        uy = np.sum(F * cys, axis=2) / rho

        Feq = np.zeros(F.shape)
        for i, cx, cy, w in zip(idxs, cxs, cys, weights):
            Feq[:, :, i] = rho * w * (
                1 + 3 * (cx * ux + cy * uy)
                + 9 * (cx * ux + cy * uy) ** 2 / 2
                - 3 * (ux**2 + uy**2) / 2
            )
        F += -(1.0 / tau) * (F - Feq)
        F[airfoil_mask, :] = bndryF

        if it % 10 == 0:
            ux[airfoil_mask] = 0
            uy[airfoil_mask] = 0
            vorticity = (
                np.roll(ux, -1, axis=0) - np.roll(ux, 1, axis=0)
            ) - (np.roll(uy, -1, axis=1) - np.roll(uy, 1, axis=1))
            vorticity[airfoil_mask] = np.nan
            vorticity_frames.append(vorticity.copy())

    def update(frame):
        ax.set_xlim(Nx / 4 - 100, Nx / 4 + 300)  # Adjust these values as needed
        ax.set_ylim(Ny / 2 - 50, Ny / 2 + 50)

        ax.clear()
        im = ax.imshow(
            frame,
            cmap="bwr",
            interpolation="none",
            aspect="equal",
            origin="lower",
        )
        ax.tick_params(axis='both', which='major', labelsize=14)  # Increase font size

        ax.invert_yaxis()
        ax.axis("off")
        return [im]

    ani = FuncAnimation(fig, update, frames=vorticity_frames, interval=50)
    os.chdir(os.path.dirname(os.getcwd()) + "/dashboard/img")
    ani.save("naca0012_vorticity.gif", writer="pillow")
    plt.show()

if __name__ == "__main__":
    main()


