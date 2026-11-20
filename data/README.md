# Umbrella sampling data

This folder holds the PLUMED `COLVAR` files from the umbrella sampling of the
VE-cadherin dimer, for every system, together with the PLUMED inputs and the GROMACS
setup files of these simulations.

The collective variable `d1` is the distance (nm) between the centres of mass of the
two cadherin chains (atoms 1-1689 and 1690-3378). Each window restrains `d1` with a
harmonic potential with force constant κ = 500 kcal/mol/nm².
The `COLVAR` files are the raw PLUMED output, and the `#! FIELDS` header line names the
columns. Times are in ps.

## Systems

| Folder              | System                                     |
|---------------------|--------------------------------------------|
| `no_plastic`        | VE-cadherin + Ca²⁺, no nanoplastic         |
| `PS10`, `PS20`, `PS40` | VE-cadherin + Ca²⁺ + polystyrene chain of 10 / 20 / 40 monomers |
| `no_plastic_no_ca`  | VE-cadherin, no nanoplastic, no Ca²⁺       |
| `2D_BO/PS<N>`       | VE-cadherin + Ca²⁺ + polystyrene chain of N monomers |

## Layout

```
data/
├── common_setup/                   setup files shared by several systems
├── <system>/
│   ├── setup/                      system-specific setup files
│   ├── <cv>/                       grid windows 1.6, 2.1, ..., 6.6 nm
│   │   ├── COLVAR
│   │   └── plumed.dat
│   └── 1D_BUQ/
│       ├── extension_setup/        setup of the 25 ns extensions
│       └── <cv>/                   windows chosen by 1D Bayesian quadrature
│           ├── COLVAR_0-50ns       original 50 ns run
│           ├── plumed_0-50ns.dat
│           ├── COLVAR_50-75ns      25 ns extension of that run
│           └── plumed_50-75ns.dat
└── 2D_BO/PS<N>/
    ├── setup/
    └── <cv>/                       windows chosen by 2D Bayesian optimisation
        ├── COLVAR
        └── plumed.dat
```

`<cv>` is the umbrella centre in nm.

## Simulation setup files

### Shared and system-specific files

Each file is stored only once.

- **`common_setup/`** holds the files that several systems share:
  - the CHARMM27 force field (`charmm27.ff/`),
  - the protein chain topologies and position restraints
    (`topol_Protein_chain_{A,B}.itp`, `posre_Protein_chain_{A,B}.itp`),
  - the topologies and position restraints of the Ca²⁺ ions
    (`topol_Ion_chain_{C,D}.itp`, `posre_Ion_chain_{C,D}.itp`),
  - the mdp files of the setup stages (`em.mdp`, `emw.mdp`, `posre.mdp`, `md.mdp`).
- **`<system>/setup/`** holds what is specific to one system:
  - `topol.top`,
  - the start structure `box.gro` (protein, Ca²⁺ and plastic, before solvation),
  - the plastic topology (`plastic.itp`; `ps20.itp` for PS20),
  - any shared file that is different for this system.
- **Which version wins.** If a file with the same name is in both `<system>/setup/`
  and `common_setup/`, the one in `<system>/setup/` is the version for that system.
  - `no_plastic` has its own Ca²⁺ ion files (3 Ca²⁺ per ion chain instead of 2).
  - `no_plastic_no_ca` has its own protein topologies and no ion files.
  - Own mdp files: `no_plastic` and `no_plastic_no_ca` all four; `PS20` `emw.mdp`
    and `md.mdp`; `PS10` `md.mdp`. All other systems use the four mdp files in
    `common_setup/`.
- **Running `grompp`.** The `topol.top` files include their itp files by name, as in
  the original simulation folder. To run `grompp`, copy the system's `setup/` files
  and the `common_setup/` files into one folder, or add the folders to the include
  path, e.g. `include = -I../../common_setup` in the mdp file.

### How the runs were set up

1. `box.gro` → solvation and ions → energy minimisation (`em.mdp`, `emw.mdp`) →
   position-restrained equilibration (`posre.mdp`) → unrestrained equilibration
   (`md.mdp`).
2. **Umbrella windows (grid, 1D BUQ and 2D BO).** The windows continue from the end of `md` and
   have no mdp file of their own. Their `.tpr` is the md `.tpr` extended with
   `gmx convert-tpr -extend`. Each window then runs `mdrun` with its own `plumed.dat`.
3. **PS10 re-equilibration.** The PS10 equilibration was rerun from the same md `.tpr`.
   The PS10 grid windows and the PS10 1D BUQ windows start from this rerun.
4. **1D BUQ extensions.** For each system, the extensions were set up with a rebuilt
   `topol.top` and `extension.mdp` (`1D_BUQ/extension_setup/`). This setup starts from
   the last frame of each 50 ns run, with new velocities. It uses the plastic itp from
   `<system>/setup/`, and the shared itps from `common_setup/` unless
   `extension_setup/` has its own version.

The intermediate structures (solvated, minimised, equilibrated) and the per-window
output structures are not included.

## COLVAR data

### Grid windows: `<system>/<cv>/`

- 11 windows per system, 1.6-6.6 nm in steps of 0.5 nm.
- Each run is 75 ns. It starts with a short steered pull from 1.6 nm to the window
  centre (column `steer.d1_cntr`), and the rest is umbrella sampling at that centre.
- **Slower pull.** In `no_plastic` 6.1 and 6.6 and in `PS20` 4.6, 5.1 and 5.6, the pull
  is twice as slow, so these runs are 90 ns.
- **Analysis window.** The last 40 ns are used: 35-75 ns of sampling, or 50-90 ns for
  the slower-pull runs.

### 1D BUQ windows: `<system>/1D_BUQ/<cv>/`

- Present for `no_plastic`, `PS10`, `PS20` and `PS40`.
- **Two files per window.** Each window was first run for 50 ns (`COLVAR_0-50ns`,
  `plumed_0-50ns.dat`) and then extended by 25 ns (`COLVAR_50-75ns`,
  `plumed_50-75ns.dat`). The extension is a separate PLUMED run, so its time column
  restarts at 0. To join the two, add the last time of `COLVAR_0-50ns` to the
  extension's times. The extension has no pull phase.
- **Analysis window.** Over the combined 75 ns, the last 40 ns (35-75 ns) are used.
- **`PS10/1D_BUQ/3.02` and `3.24`** are single continuous 75 ns runs (`COLVAR_0-75ns`,
  `plumed_0-75ns.dat`). They were not run as 50 + 25 ns.
- **`no_plastic/1D_BUQ/3.10` is a rerun of the 3.1 nm grid window.** It used the 1D BUQ
  setup (50 + 25 ns). In the free-energy profile built from 1D BUQ + grid windows, it
  replaces `no_plastic/3.1`. In the grid-only profile, `no_plastic/3.1` is used.

### 2D BO windows: `2D_BO/PS<N>/<cv>/`

- Windows chosen by Bayesian optimisation in the two-dimensional space of plastic
  length N and COM distance.
- Each run is 50 ns, with times running from 100 to 150 ns. It starts with a steered
  pull from 1.6 nm to the window centre at a constant 0.35 nm/ns, so the pull takes
  longer for larger centres: 2.0 ns for 2.30 nm, 11.2 ns for 5.52 nm. The rest is
  umbrella sampling at that centre.
- **Analysis window.** The last 15 ns (35-50 ns) are used.
- **Excluded runs.** Only the runs used in the free-energy analysis are included. These
  runs were excluded:

| Run       | Reason                       |
|-----------|------------------------------|
| PS1 4.89  | plastic moves away           |
| PS3 1.67  | COM distance below 2 nm      |
| PS4 3.03  | plastic position incorrect   |
| PS5 4.18  | plastic moves away           |
| PS8 2.65  | excluded                     |
| PS9 5.25  | chain unfolding              |
| PS16 4.75 | plastic moves away           |
| PS17 3.28 | negative mean force          |
| PS31 5.20 | chain unfolding              |
