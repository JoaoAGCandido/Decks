from pathlib import Path
import shutil
import osiris_utils as ou

DEFAULT_N_CELLS = 80

class SimOpts:
    def __init__(
        self,
        name,
        dtw_values,
        pusher,
        worktree=True,
        sim_time="0-24:00:00",
        branch=None,
        worktree_folder=None,
        n_cells=DEFAULT_N_CELLS,
        test="curv",
    ):
        self.name = name
        self.dtw_values = dtw_values
        self.pusher = pusher
        self.worktree = worktree
        self.sim_time = sim_time
        self.branch = branch if branch is not None else name
        self.worktree_folder = worktree_folder if worktree_folder is not None else self.branch
        self.n_cells = n_cells
        self.test = test


pushers = [
    # SimOpts('Boris', ["1000", "500", "100", "50", "10", "1", "0_1", "0_01", "0_001", "0_0001"], 'standard', False),
    # SimOpts('Gca', ["1000", "500", "100", "50", "10", "1", "0_1", "0_01", "0_001", "0_0001"], 'gca', False),
    # SimOpts('GcaCorr', ["1000", "500", "100", "50", "10", "1", "0_1"], 'gca_corr', False),
    # SimOpts('gcaCorrV4', ["1000", "500", "100", "50", "10", "1", "0_1"], 'gca_corr'),
    # SimOpts('gcaCorrV5', ["1000", "500", "100", "50", "10", "1", "0_1"], 'gca_corr'),
    # SimOpts('gcaCorrNoBoris', ["1000", "500", "100", "50", "10", "1", "0_1"], 'gca_corr'),
    # SimOpts('gcaNoDrifts_doublePrecDiag_and_Gca', ["1000", "500", "100", "50", "10", "1", "0_1", "0_01", "0_001", "0_0001"], 'gca')
    # SimOpts('GcaHighRes', ["1000", "500", "100", "50", "10", "1", "0_1", "0_01", "0_001", "0_0001"], 'gca')
    # SimOpts('gcaNoDrifts_doublePrecDiag_and_Gca_nx180', ["1000", "500", "100", "50", "10", "1", "0_1", "0_01", "0_001", "0_0001"], 'gca', branch="gcaNoDrifts_doublePrecDiag_and_Gca", worktree_folder='gcaNoDrifts_doublePrecDiag_and_Gca'),
    # SimOpts('Gcav4', ["100", "1", "0_01"], 'gca', branch="deucalion_gca", n_cells=80),
    # SimOpts('Gcav4_Mirror_Prec', ["1000", "500", "100", "50", "10", "1", "0_1", "0_01", "0_001", "0_0001"], 'gca', branch="deucalion_gca", n_cells=80, test="mirror"),
    SimOpts('Gcav6', ["1000", "500", "100", "50", "10", "1", "0_1", "0_01", "0_001", "0_0001"], 'gca', branch="deucalion_gca", n_cells=80),

]

BASE_DTW = 0.01
BASE_DT = 0.00006283185307179586476925286766559
NDUMP_TRACKS_ONE_FROM_DTW = 10.0 # means that for dtw >= 10, we dump tracks every step, for dtw < 10, we dump tracks every int(round(10/dtw)) steps
TRACKS_MODE = "per_dtw_1000"  # "per_dtw_1000" or "legacy"
RAW_SOURCE_RELATIVE_PATH = Path("MS/RAW/test_electrons/RAW-test_electrons-000000.h5")

CURV_TEMPLATE = """simulation
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
  nx_p(1:3) =  {n_cells}, {n_cells}, 24,
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
  num_par_x(1:3) = 2,2,2,
  add_tag = .true.,
  push_type = {pusher},
  active_gca_components = "EXB", "gradB", "curv", "vEgradb", "E_par_F", "vEbgradb_F", "vEvEgradb_F", "mirror_F",
  gca_pos_max_iter = 100
  gca_mirror_max_iter = 100
  gca_pos_res = 1.0e-16
  gca_mirror_tol = 1.0e-16
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
  ndump_fac_raw = 1,
  {tag_comment}ndump_fac_tracks = {ndump_fac_tracks},
  {tag_comment}niter_tracks = {niter_tracks},
  {tag_comment}file_tags = "tag_file_osiris_utils.tag",
  {tag_comment}ifdmp_tracks_efl(1:3) = .true., .true., .true.,
  {tag_comment}ifdmp_tracks_bfl(1:3) = .true., .true., .true.,
}}

!-------------smooth for currents-------------
smooth
{{
type(1:3) = "5pass", "5pass", "5pass",
}}
"""

MIRROR_TEMPLATE = """simulation
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
  nx_p(1:3) =  {n_cells}, {n_cells}, {n_cells},
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
  xmin(1:3) =  0.000d0, 0.000d0, 0.000d0,
  xmax(1:3) =  40., 40., 40.,
  if_move(1:3) = .false., .false., .false.,
}}

!----------time limits ----------
time
{{
  tmin = 0.0d0, tmax  = {tmax}, !125.66370614359172953850573533118,
}}

!----------field solver set up----------
el_mag_fld
{{
  ext_fld = "static",
  type_init_b(1:3) = "math func", "math func", "math func",
  init_b_mfunc(1) = "-0.1*x1*x3",
  init_b_mfunc(2) = "-0.1*x2*x3",
  init_b_mfunc(3) = "1000.0 + 0.1*x3*x3",

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
  num_par_x(1:3) = 2,2,2,
  add_tag = .true.,
  push_type = {pusher},
}}

!----------inital proper velocities----------
udist {{
  ufl(1:3)=  0.5d0 , 0.5d0 , 0.1d0 ,
  uth(1:3)=  0.01d0 , -0.01d0 , 0.01d0 ,
}}

!----------density profile for this species----------
profile
{{
  profile_type(1) = "math func",
  math_func_expr = "if((x1<23)&&(x1>17)&&(x2<23)&&(x2>17)&&(x3<22.5)&&(x3>21.5), 1.0, 0.0)",
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
  ndump_fac_raw = 1,
  {tag_comment}ndump_fac_tracks = {ndump_fac_tracks},
  {tag_comment}niter_tracks = {niter_tracks},
  {tag_comment}file_tags = "tag_file_osiris_utils.tag",
  {tag_comment}ifdmp_tracks_bfl(1:3) = .true., .true., .true.,
}}

!-------------smooth for currents-------------
smooth
{{
type(1:3) = "5pass", "5pass", "5pass",
}}
"""

TEST_TEMPLATES = {
    "curv": CURV_TEMPLATE,
    "mirror": MIRROR_TEMPLATE,
}

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

WORKTREE="$WTBASE/{worktree_folder}"

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

        for dtw in sim_opts.dtw_values:
            dtw_label = dtw_to_label(dtw)
            dtw_path = sim_path / f"dtw{dtw_label}"
            dtw_path.mkdir(parents=True, exist_ok=True)
            created_paths.append(dtw_path)

    return created_paths


def dtw_to_label(dtw):
    if isinstance(dtw, str):
        return dtw.replace(".", "_")
    return str(dtw).replace(".", "_")


def dt_from_label(dtw_label):
    dtw_str = dtw_to_label(dtw_label)
    dtw_value = float(dtw_str.replace("_", "."))
    return BASE_DT * (dtw_value / BASE_DTW)


def legacy_niter_tracks_from_label(dtw_label):
    dtw_str = dtw_to_label(dtw_label)
    dtw_value = float(dtw_str.replace("_", "."))
    if dtw_value >= NDUMP_TRACKS_ONE_FROM_DTW:
        return 1
    return int(round(NDUMP_TRACKS_ONE_FROM_DTW / dtw_value))


def scaled_niter_tracks_from_label(dtw_label):
    dtw_str = dtw_to_label(dtw_label)
    dtw_value = float(dtw_str.replace("_", "."))
    return int(round(1000.0 / dtw_value))


def niter_tracks_from_label(dtw_label, mode=TRACKS_MODE):
    if mode == "legacy":
        return legacy_niter_tracks_from_label(dtw_label)
    if mode == "per_dtw_1000":
        return scaled_niter_tracks_from_label(dtw_label)
    raise ValueError(f"Unknown TRACKS_MODE: {mode}")


def job_name_from(base_dir, sim_opts, dtw):
    test_name = Path(base_dir).name.replace("_", "")
    dtw_label = dtw_to_label(dtw)
    return f"{test_name}{sim_opts.name}{dtw_label}"


def template_for_test(test):
    try:
        return TEST_TEMPLATES[test]
    except KeyError:
        options = ", ".join(sorted(TEST_TEMPLATES))
        raise ValueError(f"Unknown test template: {test}. Expected one of: {options}") from None


def write_input_files(base_dir, pushers, tags=False, template=None):
    base_path = Path(base_dir)
    created_files = []

    for sim_opts in pushers:
        input_template = template if template is not None else template_for_test(sim_opts.test)
        for dtw in sim_opts.dtw_values:
            dtw_label = dtw_to_label(dtw)
            dtw_path = base_path / sim_opts.name / f"dtw{dtw_label}"
            dtw_path.mkdir(parents=True, exist_ok=True)

            input_file = dtw_path / f"{sim_opts.name}.in"
            dt_value = dt_from_label(dtw)
            input_file.write_text(
                input_template.format(
                    n_cells=sim_opts.n_cells,
                    dt=f"{dt_value:.30f}",
                    tmax=f"{dt_value:.30f}",
                    ndump_fac_tracks=1,
                    niter_tracks=1,
                    tag_comment="" if tags else "!",
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
        for dtw in sim_opts.dtw_values:
            dtw_label = dtw_to_label(dtw)
            dtw_path = base_path / sim_opts.name / f"dtw{dtw_label}"
            dtw_path.mkdir(parents=True, exist_ok=True)

            runjob_file = dtw_path / "runJob"
            template = worktree_template if sim_opts.worktree else local_template
            runjob_file.write_text(
                template.format(
                    job_name=job_name_from(base_dir, sim_opts, dtw),
                    sim_time=sim_opts.sim_time,
                    branch=sim_opts.branch,
                    worktree_folder=sim_opts.worktree_folder,
                    input_name=f"{sim_opts.name}.in",
                )
            )
            created_files.append(runjob_file)

    return created_files


def raw_source_path_for(dtw_path):
    return dtw_path / RAW_SOURCE_RELATIVE_PATH


def tag_source_path_for(source_path, sim_opts, dtw_label, dtw_path, output_name):
    if source_path is None:
        return raw_source_path_for(dtw_path)

    source_path = Path(source_path)
    if source_path.is_dir():
        return source_path / sim_opts.name / f"dtw{dtw_label}" / output_name

    return source_path


def write_tag_files(base_dir, pushers, tag_source_path=None, output_name="tag_file_osiris_utils.tag"):
    base_path = Path(base_dir)
    created_files = []

    for sim_opts in pushers:
        for dtw in sim_opts.dtw_values:
            dtw_label = dtw_to_label(dtw)
            dtw_path = base_path / sim_opts.name / f"dtw{dtw_label}"
            source_path = tag_source_path_for(
                tag_source_path,
                sim_opts,
                dtw_label,
                dtw_path,
                output_name,
            )
            output_path = dtw_path / output_name

            if not source_path.exists():
                raise FileNotFoundError(
                    f"Tag source not found: {source_path}. Run once with TAGS = False first, "
                    "or set TAG_SOURCE_PATH to an existing tag file, RAW file, or source base directory."
                )

            if source_path.suffix == ".tag":
                shutil.copyfile(source_path, output_path)
            else:
                raw = ou.OsirisRawFile(source_path)
                raw.raw_to_file_tags(output_path)

            created_files.append(output_path)

    return created_files


if __name__ == "__main__":
    root = Path("/home/exxxx5/Tese/Decks/StudyConvergence/gcaRefactor")
    TAGS = True
    TAG_SOURCE_PATH = "/home/exxxx5/Tese/Decks/StudyConvergence/gcaRefactor/Gcav4/dtw0_01/tag_file_osiris_utils.tag"
    # TAG_SOURCE_PATH = None
    created_dirs = create_simulation_tree(root, pushers)
    if TAGS:
        created_tags = write_tag_files(root, pushers, TAG_SOURCE_PATH)

    created_files = write_input_files(root, pushers, TAGS)
    created_runjobs = write_runjob_files(root, pushers)

    for path in created_dirs:
        print(path)
    for path in created_files:
        print(path)
    for path in created_runjobs:
        print(path)
    if TAGS:
        for path in created_tags:
            print(path)
        
