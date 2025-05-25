import matplotlib.pyplot as plt
from PIL import Image
import os
import glob

measure_mapping_radar = {
    "sb": "SB",
    "sb_sbs": "SB.SBS",
    "sbs": "SBS",
    "abfta": "ABFTA",
    "sbs_abfta": "SBS.ABFTA"
}

if __name__ == '__main__':

    base_dir = "/Users/samvanspringel/Documents/School/VUB/Master 2/Jaar/Thesis/Results"

    # Measures to process (directory names)
    all_measures = ["sb", "sb_sbs", "sbs", "abfta", "sbs_abfta"]

    for measure in all_measures:
        dir_path = os.path.join(base_dir, f"{measure}_results/budget_0")

        # Find the line graph image (assuming pattern match)
        linegraph_candidates = glob.glob(os.path.join(dir_path, "pf_budgets_*.jpg"))
        if not linegraph_candidates:
            print(f"No line graph found for {measure}")
            continue
        linegraph_img = Image.open(linegraph_candidates[0])

        # Find radar images for this measure (based on ARH.{measure} pattern)
        radar_imgs = {}
        for b in [0, 2, 3, 4, 5]:
            label = measure_mapping_radar[measure]
            radar_pattern = os.path.join(dir_path, f"ARH.{label}_budget0_b{b}.jpg")
            files = glob.glob(radar_pattern)
            if files:
                radar_imgs[b] = Image.open(files[0])

        # Check that required radars are found
        if not all(k in radar_imgs for k in [0, 2, 3, 4, 5]):
            print(f"Missing radar images for {measure}")
            continue

        # Create collage layout with nested gridspec to align bottom radars
        fig = plt.figure(figsize=(12, 10))
        outer_gs = fig.add_gridspec(3, 2,
                                    width_ratios=[2, 1],
                                    height_ratios=[1, 1, 1],
                                    wspace=0.02,  # reduce horizontal gap
                                    hspace=0.02)  # reduce vertical gap

        # Line graph spans first two rows in the left column
        ax0 = fig.add_subplot(outer_gs[0:2, 0])
        ax0.imshow(linegraph_img)
        ax0.axis('off')

        # Top-right radar b0
        ax1 = fig.add_subplot(outer_gs[0, 1])
        ax1.imshow(radar_imgs[0])
        ax1.axis('off')

        # Below that radar b5
        ax2 = fig.add_subplot(outer_gs[1, 1])
        ax2.imshow(radar_imgs[5])
        ax2.axis('off')

        # Bottom row radars: b2, b3, b4, spanning the full figure width
        bottom_gs = outer_gs[2, :].subgridspec(1, 3, wspace=0.02)
        for idx, b in enumerate([2, 3, 4]):
            ax = fig.add_subplot(bottom_gs[0, idx])
            ax.imshow(radar_imgs[b])
            ax.axis('off')

        plt.tight_layout(pad=0.5, w_pad=0.1, h_pad=0.1)

        # Save result as collage
        out_path = os.path.join(base_dir, f"combined_{measure}.jpg")
        plt.savefig(out_path, bbox_inches='tight', dpi=500)
        plt.close()
        print(f"Saved collage for {measure} to {out_path}")