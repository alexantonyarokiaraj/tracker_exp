from enum import Enum

class DataArray(Enum):
    X = 0
    Y = 1
    Z = 2
    Q = 3
    trackID = 4
    true_labels_sim = 5
    true_labels_hard = 6
    ransac_labels = 7
    gmm_labels = 8
    dbscan_labels = 9
    merge_p_val =  10
    merge_cdist = 11 
    scattered_track = 12
    track_inside_volume = 13
    vertex_inside_volume = 14
    side_of_track = 15
    closest_track = 16
    end_point_above_beam_zone = 17
    old_ransac_labels = 18

class RunParameters(Enum):
    lib_path = '/home2/user/u0100486/linux/doctorate/github/tracker_exp/lib/'
    files_path = '/home2/user/u0100486/linux/doctorate/DATA/EXPERIMENTAL/e780/'
    x_conversion_factor = 2.0  # units[mm]
    y_conversion_factor = 2.0  # units[mm]
    time_per_sample = 0.08  # units [us]
    drift_velocity_volume = 1.16  # units [cm/us]
    z_conversion_factor = time_per_sample * drift_velocity_volume * 10  # units[mm]
    nbins_x = 128
    x_start_bin = 0 * x_conversion_factor
    x_end_bin = 128 * x_conversion_factor
    nbins_y = 128
    y_start_bin = 0 * y_conversion_factor
    y_end_bin = 128 * y_conversion_factor
    nbins_z = 512
    z_start_bin = 0 * z_conversion_factor
    z_end_bin = 512 * z_conversion_factor
    lookup_table = "LT_GANIL_NewCF_marine.dat"
    missing_pads_info = "HitResponses.dat"
    NB_COBO = 16
    NB_ASAD = 4
    NB_AGET = 4
    NB_CHANNEL = 68

class VolumeBoundaries(Enum):
    VOLUME_MIN = 10
    VOLUME_MAX = 246
    BEAM_ZONE_MIN = 122
    BEAM_ZONE_MAX = 132
    BEAM_CENTER = 128

class SCAN(Enum):
    N_PROC = 1
    NN_NEIGHBOR = 6
    NN_RADIUS = 20.0
    DB_MIN_SAMPLES = 6
    SENSITIVITY = 3
    EPS_THRESHOLD = 4.0
    EPS_MODE = 7.0

class Optimize(Enum):
    ALPHA = 28.5/100 #percentage
    ALPHA_RANGE_LOW = 50/100
    ALPHA_RANGE_HIGH = 50/100
    ALPHA_STEPS = 1
    BETA = 40 #mm
    BETA_RANGE_LOW = 10 #mm
    BETA_RANGE_HIGH = 100 #mm
    BETA_STEPS = 1 #mm
    BETA_FRACTION = 55/100
    BETA_RANGE_LOW_FRACTION = 1/100
    BETA_RANGE_HIGH_FRACTION = 100/100
    BETA_STEPS_FRACTION = 100
    GAMMA = 1.0/100
    P_VALUE = 0.1
    C_DIST = 15
    C_DIST_RANGE_LOW = 1
    C_DIST_RANGE_HIGH = 100

class FileNames(Enum):
    CALIBRATION_PADS = 'pad_calibration_actar.txt'
    MISSING_PADS = 'HitResponses.dat'
    CONVERSION_TABLE = 'LT_GANIL_NewCF_marine.dat'
    CONFIG_FILE_EXCEL = 'LookupTable_e780_58Ni_68Ni_Alex.xlsx'
    RANGE_ENERGY_CONVERSION_SHEET = "range_energy_he_he_cf4_mixed"

class Reference(Enum):
    RANGE_EXTEND = 40
    RANGE_BIN_SIZE = 2
    RANGE_BIN_PER = 20
    AREA_TOTAL_PAD = 4
    LINE_LENGTH_THRESHOLD = Optimize.BETA.value  # Threshold to define smaller or larger tracks
    SAVITZKY_GOLAY_WINDOW_LARGE = 7  # Window Length for Savitzky Golay Filter for large tracks
    SAVITZKY_GOLAY_WINDOW_SMALL = 5  # Window Length for Savitzky Golay Filter for small tracks
    THRESHOLD_PEAKS = 0.25

class ConversionFactors(Enum):
    DRIFT_VELOCITY = 1.16  # units [cm/us]
    Z_CONVERSION_FACTOR = DRIFT_VELOCITY * (10.0 / 1000.0)  # mm/us
    X_CONVERSION_FACTOR = 2.0  # mm
    Y_CONVERSION_FACTOR = 2.0  # mm
    NBINS_X = 128
    X_START_BIN = 0 * X_CONVERSION_FACTOR
    X_END_BIN = 128 * X_CONVERSION_FACTOR
    NBINS_Y = 128
    Y_START_BIN = 0 * Y_CONVERSION_FACTOR
    Y_END_BIN = 128 * Y_CONVERSION_FACTOR
    NBINS_Z = 28000
    Z_START_BIN = 0 * Z_CONVERSION_FACTOR
    Z_END_BIN = 28000 * Z_CONVERSION_FACTOR

class RansacParameters(Enum):
    MAX_LINES = 10
    RESIDUAL_THRESHOLD = 5.0
    N_ITERATIONS = 5000