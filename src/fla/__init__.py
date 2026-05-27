from .constants import FPS, PX_PER_MM, TRIPOD_STANCES, TETRAPOD_STANCES, LEG_NAMES, LEG_LABELS
from .kinematics import (
    signed_angle,
    normalize_to_egocentric,
    interpolate_nans,
    get_bodypart_coords,
    get_position_diff,
)
from .locomotion import (
    get_frames_when_walking_forward,
    get_average_speed_for_ranges,
    get_raw_instantaneous_speeds,
    get_instantaneous_speeds,
)
from .gait import (
    get_gait_by_frame,
    gait_index_grabber,
    get_average_gait_index_for_ranges,
    convert_ranges_to_binary,
    get_instantaneous_speeds_and_stance,
    get_gait_probability_distribution,
)
from .steps import (
    get_swing_durations,
    get_stance_durations,
    get_swing_durations_individual,
    get_stance_durations_individual,
    get_step_lengths,
    get_period_durations,
    get_footprint_alignment,
    remove_empty_data,
)
from .angles import get_all_angles
from .fitting import linear, power_law
from .visualization import plot_swing_stance
from .embedding import (
    build_umap_dataset,
    normalize_window_vectors,
    compute_window_coloring,
    fit_umap,
    plot_umap_colored,
    plot_umap_by_group,
    plot_umap_by_individual,
)
