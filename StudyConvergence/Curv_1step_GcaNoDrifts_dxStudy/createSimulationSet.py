from pathlib import Path
import osiris_utils as ou

class SimOpts:
    def __init__(self, name, nx_values, pusher, worktree=True, sim_time="0-24:00:00"):
        self.name = name
        self.nx_values = nx_values
        self.pusher = pusher
        self.worktree = worktree
        self.sim_time = sim_time


pushers = [
    # SimOpts('gcaNoDrifts', [24, 40, 60, 80, 100, 120, 140, 160, 180], 'gca', True),
    SimOpts('gcaNoDrifts1ppc', [24, 40, 60, 80, 100, 120, 140, 160, 180], 'gca', True),
    # SimOpts('gcaCurvOnly', [24, 40, 60, 80, 100, 120, 140, 160, 180], 'gca', True),
]

FIXED_DT = 0.062831853071795867871074392497
FIXED_NX_Z = 24

INPUT_TEMPLATE = """simulation
{{
  enforce_rng_constancy  = .true.,
  disableEmfAdvance = .True.,
}}

node_conf
{{
  node_number(1:3) =  2, 2, 2,
  if_periodic(1:3) =  .true., .true., .true.,
}}

!----------spatial grid----------
grid
{{
  nx_p(1:3) =  {nx_p},
  coordinates = "cartesian",
}}

!----------time step and global data dump timestep number----------
time_step
{{
  dt     =   {dt},
  ndump  =    1,
}}

!----------restart information----------
restart
{{
  ndump_fac = 0,
  if_restart = .false.,
}}

!----------spatial limits of the simulations----------
space
{{
  xmin(1:3) =  -5., -5., -5.,
  xmax(1:3) =   5.,  5.,  5.,
  if_move(1:3) = .false., .false., .false.,
}}

!----------time limits ----------
time
{{
  tmin = 0.0d0, tmax  = {tmax},
}}

!----------field solver set up----------
el_mag_fld
{{
  ext_fld = "static",
  type_init_b(1:3) = "math func", "math func", "math func",
  init_b_mfunc(1) = "- 1000.0 * x2 / sqrt(x1*x1 + x2*x2)",
  init_b_mfunc(2) = "1000.0 * x1 / sqrt(x1*x1 + x2*x2)",
  init_b_mfunc(3) = "0.0",

  type_init_e(1:3) = "math func", "math func", "math func",
  init_e_mfunc(1) = "0.0",
  init_e_mfunc(2) = "0.0",
  init_e_mfunc(3) = "0.0",
}}

!----------boundary conditions for em-fields ----------
emf_bound
{{
  type(1:2,1) =   "open", "open",
  type(1:2,2) =   "open", "open",
  type(1:2,3) =   "open", "open",
}}

!----------diagnostic for electromagnetic fields----------
diag_emf
{{

}}

!----------number of particle species----------
particles
{{
  interpolation = "cubic",
  num_species = 1,
}}

!----------information for test species----------
species
{{
  name = "test_electrons",
  num_par_max = 6000000,
  rqm = -1.0,
  num_par_x(1:3) = 1,1,1,
  add_tag = .true.,
  push_type = {pusher},
}}

!----------inital proper velocities----------
udist {{
  ufl(1:3)=  0.05d0 , -0.05d0 , 0.0001d0 ,
  uth(1:3)=  0.001d0 , -0.001d0 , 0.001d0 ,
}}

!----------density profile for this species----------
profile
{{
  profile_type(1) = "math func",
  math_func_expr = "if( x1^2 + x2^2 < 3.5^2 && x1^2 + x2^2 > 2^2 && (x3<2.4) && (x3>(-2.4)), 1.0, 0.0)",
  den_min = 1.d-12,
}}

!----------boundary conditions for this species----------
spe_bound
{{
type(1:2,1) =    "open",    "open",
type(1:2,2) =    "open",    "open",
type(1:2,3) =    "open",    "open",
}}

!----------diagnostic for this species----------
diag_species
{{
  ndump_fac_ene = 1,
  ndump_fac_raw = 1000000,
  !ndump_fac_tracks = {ndump_fac_tracks},
  !niter_tracks = {niter_tracks},
  !file_tags = "tag_file_osiris_utils.tag",
  !ifdmp_tracks_efl(1:3) = .true., .true., .true.,
  !ifdmp_tracks_bfl(1:3) = .true., .true., .true.,
}}

!-------------smooth for currents-------------
smooth
{{
type(1:3) = "5pass", "5pass", "5pass",
}}
"""

RUNJOB_WORKTREE_TEMPLATE = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --account=f202500196cpcaa3a
#SBATCH --partition=normal-arm
#SBATCH --time={sim_time}
#SBATCH --nodes=1
#SBATCH --ntasks=8

module purge
module load OpenMPI
ml HDF5/1.12.2-gompi-2022a
module load FFTW

# ---- choose branch here ----
BRANCH="{branch}"
WTBASE="$HOME/osiris_worktrees"

WORKTREE="$WTBASE/$BRANCH"

# ---- executable + input ----
EXE="osiris-3D-${{BRANCH}}.e"
BIN="$WORKTREE/bin/$EXE"
INPUT="{input_name}"

srun "$BIN" "$INPUT"
"""

RUNJOB_LOCAL_TEMPLATE = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --account=f202500196cpcaa3a
#SBATCH --partition=normal-arm
#SBATCH --time={sim_time}
#SBATCH --nodes=1
#SBATCH --ntasks=8

module purge
module load OpenMPI
ml HDF5/1.12.2-gompi-2022a
module load FFTW

BIN="$HOME/osiris_JCDev/bin/osiris-3D-deucalion_input_disable_gca_gcaCorr.e"

srun "${{BIN}}" {input_name}
"""


def create_simulation_tree(base_dir, pushers):
    base_path = Path(base_dir)
    created_paths = []

    for sim_opts in pushers:
        sim_path = base_path / sim_opts.name
        sim_path.mkdir(parents=True, exist_ok=True)
        created_paths.append(sim_path)

        for nx in sim_opts.nx_values:
            nx_label = nx_to_label(nx)
            nx_path = sim_path / f"nx{nx_label}"
            nx_path.mkdir(parents=True, exist_ok=True)
            created_paths.append(nx_path)

    return created_paths


def nx_to_label(nx):
    if isinstance(nx, (tuple, list)):
        return "x".join(str(int(value)) for value in nx)
    return str(int(nx))


def nx_to_string(nx):
    if isinstance(nx, (tuple, list)):
        if len(nx) != 3:
            raise ValueError(f"nx must have 3 components, got {nx}")
        return ",".join(str(int(value)) for value in nx)
    value = int(nx)
    return f"{value},{value},{FIXED_NX_Z}"


def job_name_from(base_dir, sim_opts, nx):
    test_name = Path(base_dir).name.replace("_", "")
    nx_label = nx_to_label(nx)
    return f"{test_name}{sim_opts.name}nx{nx_label}"


def write_input_files(base_dir, pushers, template=INPUT_TEMPLATE):
    base_path = Path(base_dir)
    created_files = []

    for sim_opts in pushers:
        for nx in sim_opts.nx_values:
            nx_label = nx_to_label(nx)
            nx_path = base_path / sim_opts.name / f"nx{nx_label}"
            nx_path.mkdir(parents=True, exist_ok=True)

            input_file = nx_path / f"{sim_opts.name}.in"
            input_file.write_text(
                template.format(
                    nx_p=nx_to_string(nx),
                    dt=f"{FIXED_DT:.30f}",
                    tmax=f"{FIXED_DT:.30f}",
                    ndump_fac_tracks=1,
                    niter_tracks=1,
                    pusher=f'"{sim_opts.pusher}"',
                )
            )
            created_files.append(input_file)

    return created_files


def write_runjob_files(
    base_dir,
    pushers,
    worktree_template=RUNJOB_WORKTREE_TEMPLATE,
    local_template=RUNJOB_LOCAL_TEMPLATE,
):
    base_path = Path(base_dir)
    created_files = []

    for sim_opts in pushers:
        for nx in sim_opts.nx_values:
            nx_label = nx_to_label(nx)
            nx_path = base_path / sim_opts.name / f"nx{nx_label}"
            nx_path.mkdir(parents=True, exist_ok=True)

            runjob_file = nx_path / "runJob"
            template = worktree_template if sim_opts.worktree else local_template
            runjob_file.write_text(
                template.format(
                    job_name=job_name_from(base_dir, sim_opts, nx),
                    sim_time=sim_opts.sim_time,
                    branch=sim_opts.name,
                    input_name=f"{sim_opts.name}.in",
                )
            )
            created_files.append(runjob_file)

    return created_files


def write_tag_files_from_raw(
    base_dir,
    pushers,
    raw_relative_path=Path("MS/RAW/test_electrons/RAW-test_electrons-000000.h5"),
    output_name="tag_file_osiris_utils.tag",
):
    base_path = Path(base_dir)
    created_files = []

    for sim_opts in pushers:
        for nx in sim_opts.nx_values:
            nx_label = nx_to_label(nx)
            nx_path = base_path / sim_opts.name / f"nx{nx_label}"
            raw_path = nx_path / raw_relative_path

            if not raw_path.exists():
                raise FileNotFoundError(f"Raw file not found: {raw_path}")

            output_path = nx_path / output_name
            raw = ou.OsirisRawFile(raw_path)
            raw.raw_to_file_tags(output_path)
            created_files.append(output_path)

    return created_files


if __name__ == "__main__":
    root = Path("/home/exxxx5/Tese/Decks/StudyConvergence/Curv_1step_GcaNoDrifts_dxStudy")
    created_dirs = create_simulation_tree(root, pushers)
    created_files = write_input_files(root, pushers)
    created_runjobs = write_runjob_files(root, pushers)
    created_tags = write_tag_files_from_raw(root, pushers)

    for path in created_dirs:
        print(path)
    for path in created_files:
        print(path)
    for path in created_runjobs:
        print(path)
    for path in created_tags:
        print(path)
    
