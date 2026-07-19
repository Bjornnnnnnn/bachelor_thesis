
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# =====================================================
# Settings
# =====================================================

FORCE_THRESHOLD = 0.2      # N 
MOVEMENT_THRESHOLD = 2     # mm
TARGET_INTERVAL_S = 3    # target perturbation interval in seconds (e.g. 600 ms)

FORCE_FILE = r"C:\Users\bjorn\Downloads\results\measurement.csv"
ARUCO_FILE = r"C:\Users\bjorn\Downloads\results\video.csv"



# =====================================================
# Load force data
# =====================================================

force_data = pd.read_csv(
    FORCE_FILE,
    sep=";",
    decimal=","
)


force_time = (
    pd.to_numeric(
        force_data["time_ms"].astype(str).str.replace(",", ".", regex=False)
    )
    /
    1000
)

force = force_data["force_N"]

force = (
    force
    .astype(str)
    .str.replace(",", ".", regex=False)
    .astype(float)
)


# Remove missing values

valid_force = force.notna()

force_time = force_time[valid_force].reset_index(drop=True)
force = force[valid_force].reset_index(drop=True)



# =====================================================
# Synchronize force signal
# =====================================================

contact_indices = np.where(
    force.to_numpy() > FORCE_THRESHOLD
)[0]


if len(contact_indices) == 0:

    raise Exception(
        "No force event detected"
    )


force_start_time = (
    force_time.iloc[contact_indices[0]]
)


# Shift force time

force_time_sync = (
    force_time
    -
    force_start_time
)



# =====================================================
# Detect contact regions
# =====================================================

contact = (
    force
    >
    FORCE_THRESHOLD
)


regions = []

inside = False

start = None


for i, value in enumerate(contact):


    if value and not inside:

        start = force_time_sync.iloc[i]

        inside = True


    elif not value and inside:

        end = force_time_sync.iloc[i]

        regions.append(
            (
                start,
                end
            )
        )

        inside = False



if inside:

    regions.append(
        (
            start,
            force_time_sync.iloc[-1]
        )
    )



# =====================================================
# Load ArUco data
# =====================================================

aruco = pd.read_csv(
    ARUCO_FILE,
    sep=";",
    decimal=","
)


aruco_time = aruco["time_s"]

position = aruco["board_x_mm"]



valid_position = position.notna()


aruco_time = (
    aruco_time[valid_position]
    .reset_index(drop=True)
)

position = (
    position[valid_position]
    .reset_index(drop=True)
)



# Remove initial offset

position = (
    position
    -
    position.iloc[0]
)



# =====================================================
# Synchronize displacement
# =====================================================


movement_indices = np.where(
    np.abs(position.to_numpy())
    >
    MOVEMENT_THRESHOLD
)[0]


if len(movement_indices) == 0:

    raise Exception(
        "No movement detected in ArUco data"
    )


movement_start_time = (
    aruco_time.iloc[movement_indices[0]]
)


aruco_time_sync = (
    aruco_time
    -
    movement_start_time
)



# =====================================================
# Calculate speed
# =====================================================


speed = np.gradient(
    position,
    aruco_time_sync
)


# Smooth speed

window = 5

speed = np.convolve(
    speed,
    np.ones(window)/window,
    mode="same"
)



# =====================================================
# Plot
# =====================================================


fig, axes = plt.subplots(
    3,
    1,
    figsize=(10,8),
    sharex=True
)



# -----------------------------
# Force
# -----------------------------


axes[0].plot(
    force_time_sync,
    force
)


axes[0].axhline(
    FORCE_THRESHOLD,
    linestyle="--",
    label="Detection threshold" + f" ({FORCE_THRESHOLD} N)"
)



for start, end in regions:

    axes[0].axvspan(
        start,
        end,
        alpha=0.25
    )


axes[0].set_ylabel(
    "Force (N)"
)


axes[0].legend()

axes[0].grid(True)



# -----------------------------
# Displacement
# -----------------------------


axes[1].plot(
    aruco_time_sync,
    position
)


axes[1].set_ylabel(
    "X-displacement (mm)"
)


axes[1].grid(True)



# -----------------------------
# Speed
# -----------------------------


axes[2].plot(
    aruco_time_sync,
    speed
)


axes[2].set_ylabel(
    "Velocity (mm/s)"
)


axes[2].set_xlabel(
    "Time (s)"
)


axes[2].grid(True)



plt.tight_layout()


plt.savefig(
    "measurement_results_sync.png",
    dpi=300
)


plt.show()



#print the average length of each perturbation in seconds and the standard deviation, use the centre / peak of each perturbation to calculate the average time between perturbations and the standard deviation, and print the average speed during each perturbation and the standard deviation
# read peaks from the force data and use them to calculate the average time between perturbations and the standard deviation

# Perturbation durations (from detected regions)
durations = np.array([(end - start) for start, end in regions])
print("\nPerturbation count:", len(durations))
print("Mean duration (s):", np.mean(durations))
print("Std duration (s):", np.std(durations))

# Find peak times (centre) for each region using the force signal
peak_times = []
peak_forces = []
for start, end in regions:
    # indices in the synchronized force time
    mask = (force_time_sync >= start) & (force_time_sync <= end)
    if not mask.any():
        # no samples in region = bad = skip 
        continue
    seg_forces = force[mask]
    seg_times = force_time_sync[mask]
    # index of peak force in this segment
    idx = np.argmax(seg_forces.to_numpy())
    peak_t = seg_times.iloc[idx]
    peak_f = seg_forces.iloc[idx]
    peak_times.append(peak_t)
    peak_forces.append(peak_f)

peak_times = np.array(peak_times)
peak_forces = np.array(peak_forces)

if len(peak_times) >= 2:
    intervals = np.diff(peak_times)
    print("\nMean time between perturbation peaks (s):", np.mean(intervals))
    print("Std time between peaks (s):", np.std(intervals))
    # also report frequency (Hz)
    freqs = 1.0 / intervals
    print("Peak frequency (Hz):", np.mean(freqs), "±", np.std(freqs))
    # timing accuracy vs target interval
    mean_interval = float(np.mean(intervals))
    std_interval = float(np.std(intervals))
    bias_ms = (mean_interval - TARGET_INTERVAL_S) * 1000.0
    repeatability_ms = std_interval * 1000.0
    max_abs_error_ms = float(np.max(np.abs(intervals - TARGET_INTERVAL_S))) * 1000.0
    pct_bias = 100.0 * bias_ms / (TARGET_INTERVAL_S * 1000.0)
    print(f"Timing vs target {TARGET_INTERVAL_S*1000:.0f} ms: bias={bias_ms:.1f} ms, repeatability={repeatability_ms:.1f} ms, max_abs_error={max_abs_error_ms:.1f} ms, pct_bias={pct_bias:.1f}%")
else:
    print("\nNot enough peaks to compute inter-peak statistics")

# Average speed during each perturbation
# We need to align force_time_sync and aruco_time_sync. The mapping is:
# aruco_time_sync = (force_time_sync + force_start_time) - movement_start_time
time_offset = force_start_time - movement_start_time
avg_speeds = []
for start, end in regions:
    # map to aruco_time_sync coordinates
    a_start = start + time_offset
    a_end = end + time_offset
    mask = (aruco_time_sync >= a_start) & (aruco_time_sync <= a_end)
    if not mask.any():
        avg_speeds.append(np.nan)
        continue
    seg_speed = speed[mask]
    # use mean of absolute speed to represent average movement magnitude
    avg_speeds.append(np.nanmean(np.abs(seg_speed)))

avg_speeds = np.array(avg_speeds, dtype=float)
valid = ~np.isnan(avg_speeds)
if valid.any():
    print("\nAverage speed during perturbations (mm/s):")
    print("  count:", np.sum(valid))
    print("  mean:", np.mean(avg_speeds[valid]))
    print("  std:", np.std(avg_speeds[valid]))
else:
    print("\nNo valid speed samples inside perturbations (check synchronization)")

# Optionally, print per-perturbation summary table
print("\nPer-perturbation summary:")
for i, (start, end) in enumerate(regions):
    dur = end - start
    peak = peak_times[i] if i < len(peak_times) else None
    pf = peak_forces[i] if i < len(peak_forces) else None
    sp = avg_speeds[i] if i < len(avg_speeds) else np.nan
    # compute stroke (displacement range) during the contact region
    a_start = start + time_offset
    a_end = end + time_offset
    mask = (aruco_time_sync >= a_start) & (aruco_time_sync <= a_end)
    if mask.any():
        seg_pos = position[mask]
        stroke = float(np.max(seg_pos) - np.min(seg_pos))
    else:
        stroke = float('nan')
    print(f"#{i+1}: start={start:.3f}s end={end:.3f}s dur={dur:.3f}s peak_t={peak} peak_f={pf} avg_speed={sp} stroke_mm={stroke}")

# compute per-perturbation stroke amplitudes (displacement range during contact)
strokes = []
for start, end in regions:
    a_start = start + time_offset
    a_end = end + time_offset
    mask = (aruco_time_sync >= a_start) & (aruco_time_sync <= a_end)
    if not mask.any():
        strokes.append(np.nan)
        continue
    seg_pos = position[mask]
    strokes.append(float(np.max(seg_pos) - np.min(seg_pos)))

strokes = np.array(strokes, dtype=float)
valid_strokes = ~np.isnan(strokes)
if valid_strokes.any():
    print("\nStroke amplitude during perturbations (mm):")
    print("  count:", np.sum(valid_strokes))
    print("  mean:", np.mean(strokes[valid_strokes]))
    print("  std:", np.std(strokes[valid_strokes]))
    print("  min:", np.min(strokes[valid_strokes]))
    print("  max:", np.max(strokes[valid_strokes]))
else:
    print("\nNo valid stroke samples inside perturbations (check synchronization)")
#### Summary values #####
summary = {}

# Perturbation count
summary['perturbation_count'] = int(len(durations)) if len(durations) > 0 else 0

# Duration stats
summary['duration_mean_s'] = float(np.mean(durations)) if len(durations) > 0 else float('nan')
summary['duration_std_s'] = float(np.std(durations)) if len(durations) > 0 else float('nan')
# Contact time is defined as time with measured force > threshold (same as durations)
summary['contact_time_mean_s'] = summary['duration_mean_s']
summary['contact_time_std_s'] = summary['duration_std_s']

# Peak force range
if len(peak_forces) > 0:
    summary['peak_force_min_N'] = float(np.min(peak_forces))
    summary['peak_force_max_N'] = float(np.max(peak_forces))
else:
    summary['peak_force_min_N'] = float('nan')
    summary['peak_force_max_N'] = float('nan')

# Inter-peak interval stats
if len(peak_times) >= 2:
    summary['peak_interval_mean_s'] = float(np.mean(np.diff(peak_times)))
    summary['peak_interval_std_s'] = float(np.std(np.diff(peak_times)))
    # frequency statistics derived from inter-peak intervals
    freqs = 1.0 / np.diff(peak_times)
    summary['peak_freq_mean_hz'] = float(np.mean(freqs))
    summary['peak_freq_std_hz'] = float(np.std(freqs))
    # timing accuracy values
    summary['target_interval_s'] = float(TARGET_INTERVAL_S)
    summary['timing_bias_ms'] = float((summary['peak_interval_mean_s'] - TARGET_INTERVAL_S) * 1000.0)
    summary['timing_repeatability_ms'] = float(summary['peak_interval_std_s'] * 1000.0)
    summary['timing_max_abs_error_ms'] = float(np.max(np.abs(np.diff(peak_times) - TARGET_INTERVAL_S)) * 1000.0)
    summary['timing_pct_bias'] = float(100.0 * summary['timing_bias_ms'] / (TARGET_INTERVAL_S * 1000.0))
else:
    summary['peak_interval_mean_s'] = float('nan')
    summary['peak_interval_std_s'] = float('nan')
    summary['peak_freq_mean_hz'] = float('nan')
    summary['peak_freq_std_hz'] = float('nan')
    summary['target_interval_s'] = float(TARGET_INTERVAL_S)
    summary['timing_bias_ms'] = float('nan')
    summary['timing_repeatability_ms'] = float('nan')
    summary['timing_max_abs_error_ms'] = float('nan')
    summary['timing_pct_bias'] = float('nan')

# Displacement stats
summary['max_displacement_mm'] = float(np.max(position))
summary['min_displacement_mm'] = float(np.min(position))
# first time negative displacement observed (absolute aruco_time_sync)
neg_idx = np.where(position < 0)[0]
summary['first_negative_displacement_time_s'] = float(aruco_time_sync.iloc[neg_idx[0]]) if neg_idx.size > 0 else float('nan')

# Velocity stats
summary['max_velocity_mm_s'] = float(np.max(np.abs(speed)))
if valid.any():
    summary['avg_velocity_mean_mm_s'] = float(np.mean(avg_speeds[valid]))
    summary['avg_velocity_std_mm_s'] = float(np.std(avg_speeds[valid]))
else:
    summary['avg_velocity_mean_mm_s'] = float('nan')
    summary['avg_velocity_std_mm_s'] = float('nan')


# Print summary
print("\nSummary values:")
print(f"Perturbations: {summary['perturbation_count']}")
print(f"Duration: {summary['duration_mean_s']:.3f} ± {summary['duration_std_s']:.3f} s")

print(f"Peak force: {summary['peak_force_min_N']:.2f} to {summary['peak_force_max_N']:.2f} N")
print(f"Max displacement: {summary['max_displacement_mm']:.2f} mm")
print(f"Min displacement: {summary['min_displacement_mm']:.2f} mm (first negative at {summary['first_negative_displacement_time_s']:.3f} s)")
print(f"Max velocity: {summary['max_velocity_mm_s']:.1f} mm/s")
print(f"Avg velocity during perturbations: {summary['avg_velocity_mean_mm_s']:.3f} ± {summary['avg_velocity_std_mm_s']:.3f} mm/s")
print(f"Peak interval: {summary['peak_interval_mean_s']:.3f} ± {summary['peak_interval_std_s']:.3f} s")
print(f"Peak frequency: {summary['peak_freq_mean_hz']:.3f} ± {summary['peak_freq_std_hz']:.3f} Hz")
