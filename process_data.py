import numpy as np
import matplotlib.pyplot as plt

# Set bigger font of all matplotlib fonts
plt.rcParams.update({'font.size': 18})


data_path = "results/raw_data/"

hole_counts=[1, 3, 5, 7, 10, 12]
hole_widths=[0.1, 0.2, 0.3, 0.4]
data_tracker_interval = 0.1
repeats = [1, 2, 3]


for hole_width in hole_widths:
    m_avgs = []
    m_eims = []
    for hole_count in hole_counts:
        ms = []
        for repeat in repeats:
            # load dm_dt
            try:
                file_name = data_path + f"log_holecount_{hole_count}_holewidth_{hole_width}_repeat_{repeat}_dm_dt.npy"
                dm_dt_data = np.load(file_name, allow_pickle=True)
                print(f"Loaded {file_name}")
            except FileNotFoundError:
                print(f"File {file_name} not found")
                continue

            dm_dt = np.array([entry['dm_dt'] for entry in dm_dt_data], dtype=float)


            # integrate dm_dt to get m
            m = np.zeros_like(dm_dt)
            m[1:] = np.cumsum(np.trapz(dm_dt, dx=data_tracker_interval))

            ms.append(m)
        
        # Calculate mean and std
        m_avg = np.mean(ms)
        m_std = np.std(ms)
        m_eim = m_std / np.sqrt(len(ms))

        m_avgs.append(m_avg)
        m_eims.append(m_eim)

    # Plot m vs hole_count
    plt.errorbar(hole_counts, m_avgs, yerr=m_eims, fmt='o', label=f"Width of holes={hole_width}", capsize=5)
    plt.xlabel("Number of holes")
    plt.ylabel("Total flux of mass")
    plt.legend()
    plt.title(f"Total flux of mass through window vs number of holes in window")
plt.show()







        

