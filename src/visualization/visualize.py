import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------
df = pd.read_pickle("../../data/interim/0.1_data_processed.pkl")
df

# --------------------------------------------------------------
# Plot single columns
# --------------------------------------------------------------
set_df = df[df["set"] == 1]

plt.plot(set_df["acc_y"])

set_df["acc_y"].reset_index(drop=True)
plt.plot(set_df["acc_y"].reset_index(drop=True))

# --------------------------------------------------------------
# Plot all exercises
# --------------------------------------------------------------

for label in df["label"].unique():
    subset = df[df["label"] == label]  # bunun ne işe yaradığına bak
    plt.plot(subset["acc_y"].reset_index(drop=True), label=label)
    plt.legend()  # bunun ne işe yaradığına bak
    plt.show()  # bunun ne işe yaradığına bak

for label in df["label"].unique():
    subset = df[df["label"] == label]  # bunun ne işe yaradığına bak
    plt.plot(subset[:100]["acc_y"].reset_index(drop=True), label=label)
    plt.legend()  # bunun ne işe yaradığına bak
    plt.show()  # bunun ne işe yaradığına bak

# --------------------------------------------------------------
# Adjust plot settings
# --------------------------------------------------------------
mpl.style.use("seaborn-v0_8-deep")
mpl.rcParams["figure.figsize"] = (20, 5)
mpl.rcParams["figure.dpi"] = 100


# --------------------------------------------------------------
# Compare medium vs. heavy sets
# --------------------------------------------------------------

category_df = df.query("label == 'squat'").query("participant == 'A'").reset_index()

fig, ax = plt.subplots()
category_df.groupby(["category"])["acc_y"].plot()
ax.set_ylabel("acc y")
ax.set_xlabel("samples")
plt.legend()


# --------------------------------------------------------------
# Compare participants
# --------------------------------------------------------------

participant_df = df.query("label == 'bench'").sort_values("participant").reset_index()

fig, ax = plt.subplots()
participant_df.groupby(["participant"])["acc_y"].plot()
ax.set_ylabel("acc y")
ax.set_xlabel("samples")
plt.legend()


# --------------------------------------------------------------
# Plot multiple axis
# --------------------------------------------------------------

label = "squat"
participant = "A"
all_axis_df = (
    df.query(f"label == '{label}'")
    .query(f"participant == '{participant}'")
    .reset_index()
)

fig, ax = plt.subplots()
all_axis_df[["acc_x", "acc_y", "acc_z"]].plot(
    ax=ax
)  # iki tane köşeli parantez olunca dataframe
# çevirir ve böylece birden fazla kolonu bitek bu şekilde kullanabiliriz
ax.set_ylabel("xyz")
ax.set_xlabel("samples")
plt.legend()

# --------------------------------------------------------------
# Create a loop to plot all combinations per sensor
# --------------------------------------------------------------

for label in df["label"].unique():
    for participant in sorted(df["participant"].unique()):
        all_axis_df = (
            df.query(f"label == '{label}'")
            .query(f"participant == '{participant}'")
            .reset_index()
        )
        if len(all_axis_df) > 0:
            fig, ax = plt.subplots()
            all_axis_df[["acc_x", "acc_y", "acc_z"]].plot(ax=ax)
            ax.set_ylabel("xyz")
            ax.set_xlabel("samples")
            plt.title(f"{label}({participant})".title())
            plt.legend()


for label in df["label"].unique():
    for participant in sorted(df["participant"].unique()):
        all_axis_df = (
            df.query(f"label == '{label}'")
            .query(f"participant == '{participant}'")
            .reset_index()
        )
        if len(all_axis_df) > 0:
            fig, ax = plt.subplots()
            all_axis_df[["gyr_x", "gyr_y", "gyr_z"]].plot(ax=ax)
            ax.set_ylabel("xyz")
            ax.set_xlabel("samples")
            plt.title(f"{label}({participant})".title())
            plt.legend()

# --------------------------------------------------------------
# Combine plots in one figure
# Loop over all combinations and export for both sensors
# --------------------------------------------------------------
for label in df["label"].unique():
    for participant in sorted(df["participant"].unique()):
        all_axis_df = (
            df.query(f"label == '{label}'")
            .query(f"participant == '{participant}'")
            .reset_index()
        )
        if len(all_axis_df) > 0:
            fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(20, 10))
            all_axis_df[["acc_x", "acc_y", "acc_z"]].plot(ax=ax[0])
            all_axis_df[["gyr_x", "gyr_y", "gyr_z"]].plot(ax=ax[1])
            ax[0].legend(
                loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=3, fancybox=True
            )
            ax[1].legend(
                loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=3, fancybox=True
            )
            ax[1].set_xlabel("samples")
            plt.savefig(f"../../reports/figures/{label.title()} ({participant}).png")
            plt.show()
