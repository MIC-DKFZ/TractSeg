
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join
import math

import numpy as np
import nibabel as nib
from nibabel import trackvis
from dipy.tracking.streamline import transform_streamlines
from scipy.ndimage.morphology import binary_dilation
from dipy.tracking.streamline import set_number_of_points
from dipy.tracking.streamline import length as sl_length
from dipy.tracking.streamline import Streamlines
from dipy.segment.clustering import QuickBundles
from dipy.segment.metric import AveragePointwiseEuclideanMetric
from scipy.spatial import cKDTree

from tractseg.data import dataset_specific_utils
from tractseg.libs import fiber_utils
from tractseg.libs import img_utils

import matplotlib
matplotlib.use('Agg')  # Solves error with ssh and plotting
#https://www.quora.com/If-a-Python-program-already-has-numerous-matplotlib-plot-functions-what-is-the-quickest-way-to-convert-them-all-to-a-way-where-all-the-plots-can-be-produced-as-hard-images-with-minimal-modification-of-code
import matplotlib.pyplot as plt
# Might fix problems with matplotlib over ssh (failing after connection is open for longer)
#   http://stackoverflow.com/questions/2443702/problem-running-python-matplotlib-in-background-after-ending-ssh-session
plt.ioff()


def plot_mask(renderer, mask_data, affine, x_current, y_current, orientation="axial", smoothing=10, brain_mask=None):
    from tractseg.libs import vtk_utils

    if brain_mask is not None:
        brain_mask = brain_mask.transpose(0, 2, 1)
        brain_mask = brain_mask[::-1, :, :]
        if orientation == "sagittal":
            brain_mask = brain_mask.transpose(2, 1, 0)
            brain_mask = brain_mask[::-1, :, :]
        cont_actor = vtk_utils.contour_from_roi_smooth(brain_mask, affine=affine,
                                                       color=[.9, .9, .9], opacity=.1, smoothing=30)
        cont_actor.SetPosition(x_current, y_current, 0)
        renderer.add(cont_actor)

    # 3D Bundle
    mask = mask_data
    mask = mask.transpose(0, 2, 1)
    mask = mask[::-1, :, :]
    if orientation == "sagittal":
        mask = mask.transpose(2, 1, 0)
        mask = mask[::-1, :, :]
    color = [1, .27, .18]  # red

    cont_actor = vtk_utils.contour_from_roi_smooth(mask, affine=affine,
                                                   color=color, opacity=1, smoothing=smoothing)
    cont_actor.SetPosition(x_current, y_current, 0)
    renderer.add(cont_actor)


def plot_tracts(classes, bundle_segmentations, affine, out_dir, brain_mask=None):
    """
    By default this does not work on a remote server connection (ssh -X) because -X does not support OpenGL.
    On the remote Server you can do 'export DISPLAY=":0"' .
    (you should set the value you get if you do 'echo $DISPLAY' if you
    login locally on the remote server). Then all graphics will get rendered locally and not via -X.
    (important: graphical session needs to be running on remote server (e.g. via login locally))
    (important: login needed, not just stay at login screen)

    If running on a headless server without Display using Xvfb might help:
    https://stackoverflow.com/questions/6281998/can-i-run-glu-opengl-on-a-headless-server
    """
    from dipy.viz import window
    from tractseg.libs import vtk_utils

    SMOOTHING = 10
    WINDOW_SIZE = (800, 800)
    bundles = ["CST_right", "CA", "IFO_right"]

    renderer = window.Renderer()
    renderer.projection('parallel')

    rows = len(bundles)
    X, Y, Z = bundle_segmentations.shape[:3]
    for j, bundle in enumerate(bundles):
        i = 0  #only one method

        bundle_idx = dataset_specific_utils.get_bundle_names(classes)[1:].index(bundle)
        mask_data = bundle_segmentations[:,:,:,bundle_idx]

        if bundle == "CST_right":
            orientation = "axial"
        elif bundle == "CA":
            orientation = "axial"
        elif bundle == "IFO_right":
            orientation = "sagittal"
        else:
            orientation = "axial"

        #bigger: more border
        if orientation == "axial":
            border_y = -100
        else:
            border_y = -100

        x_current = X * i  # column (width)
        y_current = rows * (Y * 2 + border_y) - (Y * 2 + border_y) * j  # row (height)  (starts from bottom)

        plot_mask(renderer, mask_data, affine, x_current, y_current,
                            orientation=orientation, smoothing=SMOOTHING, brain_mask=brain_mask)

        #Bundle label
        text_offset_top = -50
        text_offset_side = -100
        position = (0 - int(X) + text_offset_side, y_current + text_offset_top, 50)
        text_actor = vtk_utils.label(text=bundle, pos=position, scale=(6, 6, 6), color=(1, 1, 1))
        renderer.add(text_actor)

    renderer.reset_camera()
    window.record(renderer, out_path=join(out_dir, "preview.png"),
                  size=(WINDOW_SIZE[0], WINDOW_SIZE[1]), reset_camera=False, magnification=2)


def plot_tracts_matplotlib(classes, bundle_segmentations, background_img, out_dir,
                           threshold=0.001, exp_type="tract_segmentation"):

    def plot_single_tract(bg, data, orientation, bundle, exp_type):
        if orientation == "coronal":
            data = data.transpose(2, 0, 1, 3) if exp_type == "peak_regression" else data.transpose(2, 0, 1)
            data = data[::-1, :, :]
            bg = bg.transpose(2, 0, 1)[::-1, :, :]
        elif orientation == "sagittal":
            data = data.transpose(2, 1, 0, 3) if exp_type == "peak_regression" else data.transpose(2, 1, 0)
            data = data[::-1, :, :]
            bg = bg.transpose(2, 1, 0)[::-1, :, :]
        else:  # axial
            pass

        mask_voxel_coords = np.where(data != 0)
        if len(mask_voxel_coords) > 2 and len(mask_voxel_coords[2]) > 0:
            minidx = int(np.min(mask_voxel_coords[2]))
            maxidx = int(np.max(mask_voxel_coords[2])) + 1
            mean_slice = int(np.mean([minidx, maxidx]))
        else:
            mean_slice = int(bg.shape[2] / 2)
        bg = bg[:, :, mean_slice]
        # bg = matplotlib.colors.Normalize()(bg)

        # project 3D to 2D image
        if aggregation == "mean":
            data = data.mean(axis=2)
        else:
            data = data.max(axis=2)

        plt.imshow(bg, cmap="gray")
        data = np.ma.masked_where(data < 0.00001, data)
        plt.imshow(data, cmap="autumn")  # even with cmap=autumn peaks still RGB
        plt.title(bundle, fontsize=7)

    if classes.startswith("xtract"):
        bundles = ["cst_r", "cst_s_r", "ifo_r", "fx_l", "fx_r", "or_l", "fma"]
    else:
        if exp_type == "endings_segmentation":
            bundles = ["CST_right_b", "CST_right_e", "CST_s_right_b", "CST_s_right_e", "CA_b", "CA_e"]
        else:
            bundles = ["CST_right", "CST_s_right", "CA", "IFO_right", "FX_left", "FX_right", "OR_left", "CC_1"]

    if exp_type == "peak_regression":
        s = bundle_segmentations.shape
        bundle_segmentations = bundle_segmentations.reshape([s[0], s[1], s[2], int(s[3]/3), 3])
        bundles = ["CST_right", "CST_s_right", "CA", "CC_1", "AF_left"]  # can only use bundles from part1

    aggregation = "max"
    cols = 4
    rows = math.ceil(len(bundles) / cols)

    background_img = background_img[...,0]

    for j, bundle in enumerate(bundles):
        bun = bundle.lower()
        if bun.startswith("ca") or bun.startswith("fx_") or bun.startswith("or_") or \
                bun.startswith("cc_1") or bun.startswith("fma"):
            orientation = "axial"
        elif bun.startswith("ifo_") or bun.startswith("icp_") or bun.startswith("cst_s_") or \
                bun.startswith("af_"):
            bundle = bundle.replace("_s", "")
            orientation = "sagittal"
        elif bun.startswith("cst_"):
            orientation = "coronal"
        else:
            raise ValueError("invalid bundle")

        bundle_idx = dataset_specific_utils.get_bundle_names(classes)[1:].index(bundle)
        mask_data = bundle_segmentations[:, :, :, bundle_idx]
        mask_data = np.copy(mask_data)  # copy data otherwise will also threshold data outside of plot function
        # mask_data[mask_data < threshold] = 0
        mask_data[mask_data < 0.001] = 0  # higher value better for preview, otherwise half of image just red

        plt.subplot(rows, cols, j+1)
        plt.axis("off")
        plot_single_tract(background_img, mask_data, orientation, bundle, exp_type=exp_type)

    if exp_type == "tract_segmentation":
        file_name = "preview_bundle"
    elif exp_type == "endings_segmentation":
        file_name = "preview_endings"
    elif exp_type == "peak_regression":
        file_name = "preview_TOM"
    elif exp_type == "dm_regression":
        file_name = "preview_dm"
    else:
        file_name = "preview"

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(join(out_dir, file_name + ".png"), bbox_inches='tight', dpi=300)


def create_exp_plot(metrics, path, exp_name, without_first_epochs=False,
                    keys=["loss", "f1_macro"], types=["train", "validate"], selected_ax=["loss", "f1"],
                    fig_name="metrics.png"):

    colors = ["r", "g", "b", "m"]
    markers = [":", "", "--"]

    if "loss" in keys:
        min_loss_test = np.min(metrics["loss_validate"])
        min_loss_test_epoch_idx = np.argmin(metrics["loss_validate"])
        description_loss = "min loss_validate: {} (ep {})".format(round(min_loss_test, 7), min_loss_test_epoch_idx)

        if "f1_macro" in keys:
            max_f1_test = np.max(metrics["f1_macro_validate"])
            max_f1_test_epoch_idx = np.argmax(metrics["f1_macro_validate"])
            description_f1 = "max f1_macro_validate: {} (ep {})".format(round(max_f1_test, 4), max_f1_test_epoch_idx)
        elif "angle_err" in keys:
            min_angle_test = np.min(metrics["angle_err_validate"])
            min_angle_test_epoch_idx = np.argmin(metrics["angle_err_validate"])
            description_f1 = "min angle_err_validate: {} (ep {})".format(round(min_angle_test, 4), min_angle_test_epoch_idx)

        description = description_loss + " || " + description_f1
    else:
        description = "MIN not available because loss and f1_macro not in keys"

    axes = {}

    fig, ax = plt.subplots(figsize=(17, 5))
    axes["loss"] = ax

    # does not properly work with ax.twinx()
    # fig.gca().set_position((.1, .3, .8, .6))  # [left, bottom, width, height]; between 0-1: where it should be in result

    plt.grid(b=True, which='major', color='black', linestyle='-')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    ax2 = ax.twinx()  # create second scale
    axes["f1"] = ax2

    #shrink current axis by 5% to make room for legend next to it
    box = axes["loss"].get_position()
    axes["loss"].set_position([box.x0, box.y0, box.width * 0.95, box.height])
    box2 = axes["f1"].get_position()
    axes["f1"].set_position([box2.x0, box2.y0, box2.width * 0.95, box2.height])

    if without_first_epochs:
        handles = []
        for idx, key in enumerate(keys):
            for jdx, type in enumerate(types):
                name = key + "_" + type
                plt_handle, = axes[selected_ax[idx]].plot(list(range(5, len(metrics[name]))),
                                metrics[name][5:], colors[idx] + markers[jdx], label=name)
                handles.append(plt_handle)

        plt.legend(handles=handles,
                   loc=2,
                   borderaxespad=0.,
                   bbox_to_anchor=(1.03, 1))

        fig_name = fig_name

    else:
        handles = []
        for idx, key in enumerate(keys):
            for jdx, type in enumerate(types):
                name = key + "_" + type
                plt_handle, = axes[selected_ax[idx]].plot(metrics[name], colors[idx] + markers[jdx], label=name)
                handles.append(plt_handle)

        plt.legend(handles=handles,
                   loc=2,
                   borderaxespad=0.,
                   bbox_to_anchor=(1.03, 1))

        fig_name = fig_name

    fig.text(0.12, 0.95, exp_name, size=12, weight="bold")
    fig.text(0.12, 0.02, description)
    fig.savefig(join(path, fig_name), dpi=100)
    plt.close()


def plot_result_trixi(trixi, x, y, probs, loss, f1, epoch_nr):
    import torch

    x_norm = (x - x.min()) / (x.max() - x.min() + 1e-7)  # for proper plotting
    trixi.show_image_grid(torch.tensor(x_norm).float()[:5, 0:1, :, :], name="input batch",
                          title="Input batch")  # all channels of one batch

    probs_shaped = probs[:, 15:16, :, :]  # (bs, 1, x, y)
    probs_shaped_bin = (probs_shaped > 0.5).int()
    trixi.show_image_grid(probs_shaped[:5], name="predictions", title="Predictions Probmap")

    # Show GT and Prediction in one image  (bundle: CST); GREEN: GT; RED: prediction (FP); YELLOW: prediction (TP)
    combined = torch.zeros((y.shape[0], 3, y.shape[2], y.shape[3]))
    combined[:, 0:1, :, :] = probs_shaped_bin  # Red
    combined[:, 1:2, :, :] = torch.tensor(y)[:, 15:16, :, :]  # Green
    trixi.show_image_grid(combined[:5], name="predictions_combined", title="Combined")

    # #Show feature activations
    # contr_1_2 = intermediate[2].data.cpu().numpy()   # (bs, nr_feature_channels=64, x, y)
    # contr_1_2 = contr_1_2[0:1,:,:,:].transpose((1,0,2,3)) # (nr_feature_channels=64, 1, x, y)
    # contr_1_2 = (contr_1_2 - contr_1_2.min()) / (contr_1_2.max() - contr_1_2.min())
    # nvl.show_images(contr_1_2, name="contr_1_2", title="contr_1_2")
    #
    # # Show feature activations
    # contr_3_2 = intermediate[1].data.cpu().numpy()  # (bs, nr_feature_channels=64, x, y)
    # contr_3_2 = contr_3_2[0:1, :, :, :].transpose((1, 0, 2, 3))  # (nr_feature_channels=64, 1, x, y)
    # contr_3_2 = (contr_3_2 - contr_3_2.min()) / (contr_3_2.max() - contr_3_2.min())
    # nvl.show_images(contr_3_2, name="contr_3_2", title="contr_3_2")
    #
    # # Show feature activations
    # deconv_2 = intermediate[0].data.cpu().numpy()  # (bs, nr_feature_channels=64, x, y)
    # deconv_2 = deconv_2[0:1, :, :, :].transpose((1, 0, 2, 3))  # (nr_feature_channels=64, 1, x, y)
    # deconv_2 = (deconv_2 - deconv_2.min()) / (deconv_2.max() - deconv_2.min())
    # nvl.show_images(deconv_2, name="deconv_2", title="deconv_2")

    trixi.show_value(value=float(loss), counter=epoch_nr, name="loss", tag="loss")
    trixi.show_value(value=float(np.mean(f1)), counter=epoch_nr, name="f1", tag="f1")


def plot_bundles_with_metric(bundle_path, endings_path, brain_mask_path, bundle, metrics, output_path,
                             tracking_format="trk_legacy", show_color_bar=True):
    import seaborn as sns  # import in function to avoid error if not installed (this is only needed in this function)
    from dipy.viz import actor, window
    from tractseg.libs import vtk_utils

    def _add_extra_point_to_last_streamline(sl):
        # Coloring broken as soon as all streamlines have same number of points -> why???
        # Add one number to last streamline to make it have a different number
        sl[-1] = np.append(sl[-1], [sl[-1][-1]], axis=0)
        return sl

    # Settings
    NR_SEGMENTS = 100
    ANTI_INTERPOL_MULT = 1  # increase number of points to avoid interpolation to blur the colors
    algorithm = "distance_map"  # equal_dist | distance_map | cutting_plane
    # colors = np.array(sns.color_palette("coolwarm", NR_SEGMENTS))  # colormap blue to red (does not fit to colorbar)
    colors = np.array(sns.light_palette("red", NR_SEGMENTS))  # colormap only red, which fits to color_bar
    img_size = (1000, 1000)

    # Tractometry skips first and last element. Therefore we only have 98 instead of 100 elements.
    # Here we duplicate the first and last element to get back to 100 elements
    metrics = list(metrics)
    metrics = np.array([metrics[0]] + metrics + [metrics[-1]])

    metrics_max = metrics.max()
    metrics_min = metrics.min()
    if metrics_max == metrics_min:
        metrics = np.zeros(len(metrics))
    else:
        metrics = img_utils.scale_to_range(metrics, range=(0, 99))  # range needs to be same as segments in colormap

    orientation = dataset_specific_utils.get_optimal_orientation_for_bundle(bundle)

    # Load mask
    beginnings_img = nib.load(endings_path)
    beginnings = beginnings_img.get_fdata().astype(np.uint8)
    for i in range(1):
        beginnings = binary_dilation(beginnings)

    # Load trackings
    if tracking_format == "trk_legacy":
        streams, hdr = trackvis.read(bundle_path)
        streamlines = [s[0] for s in streams]
    else:
        sl_file = nib.streamlines.load(bundle_path)
        streamlines = sl_file.streamlines

    # Reduce streamline count
    streamlines = streamlines[::2]

    # Reorder to make all streamlines have same start region
    streamlines = fiber_utils.add_to_each_streamline(streamlines, 0.5)
    streamlines_new = []
    for idx, sl in enumerate(streamlines):
        startpoint = sl[0]
        # Flip streamline if not in right order
        if beginnings[int(startpoint[0]), int(startpoint[1]), int(startpoint[2])] == 0:
            sl = sl[::-1, :]
        streamlines_new.append(sl)
    streamlines = fiber_utils.add_to_each_streamline(streamlines_new, -0.5)

    if algorithm == "distance_map" or algorithm == "equal_dist":
        streamlines = fiber_utils.resample_fibers(streamlines, NR_SEGMENTS * ANTI_INTERPOL_MULT)
    elif algorithm == "cutting_plane":
        streamlines = fiber_utils.resample_to_same_distance(streamlines, max_nr_points=NR_SEGMENTS,
                                                            ANTI_INTERPOL_MULT=ANTI_INTERPOL_MULT)

    # Cut start and end by percentage
    # streamlines = FiberUtils.resample_fibers(streamlines, NR_SEGMENTS * ANTI_INTERPOL_MULT)
    # remove = int((NR_SEGMENTS * ANTI_INTERPOL_MULT) * 0.15)  # remove X% in beginning and end
    # streamlines = np.array(streamlines)[:, remove:-remove, :]
    # streamlines = list(streamlines)

    if algorithm == "equal_dist":
        segment_idxs = []
        for i in range(len(streamlines)):
            segment_idxs.append(list(range(NR_SEGMENTS * ANTI_INTERPOL_MULT)))
        segment_idxs = np.array(segment_idxs)

    elif algorithm == "distance_map":
        metric = AveragePointwiseEuclideanMetric()
        qb = QuickBundles(threshold=100., metric=metric)
        clusters = qb.cluster(streamlines)
        centroids = Streamlines(clusters.centroids)
        _, segment_idxs = cKDTree(centroids.data, 1, copy_data=True).query(streamlines, k=1)

    elif algorithm == "cutting_plane":
        streamlines_resamp = fiber_utils.resample_fibers(streamlines, NR_SEGMENTS * ANTI_INTERPOL_MULT)
        metric = AveragePointwiseEuclideanMetric()
        qb = QuickBundles(threshold=100., metric=metric)
        clusters = qb.cluster(streamlines_resamp)
        centroid = Streamlines(clusters.centroids)[0]
        # index of the middle cluster
        middle_idx = int(NR_SEGMENTS / 2) * ANTI_INTERPOL_MULT
        middle_point = centroid[middle_idx]
        segment_idxs = fiber_utils.get_idxs_of_closest_points(streamlines, middle_point)
        # Align along the middle and assign indices
        segment_idxs_eqlen = []
        for idx, sl in enumerate(streamlines):
            sl_middle_pos = segment_idxs[idx]
            before_elems = sl_middle_pos
            after_elems = len(sl) - sl_middle_pos
            base_idx = 1000  # use higher index to avoid negative numbers for area below middle
            r = range((base_idx - before_elems), (base_idx + after_elems))
            segment_idxs_eqlen.append(r)
        segment_idxs = segment_idxs_eqlen

    # Add extra point otherwise coloring BUG
    streamlines = _add_extra_point_to_last_streamline(streamlines)

    renderer = window.Renderer()
    colors_all = []  # final shape will be [nr_streamlines, nr_points, 3]
    for jdx, sl in enumerate(streamlines):
        colors_sl = []
        for idx, p in enumerate(sl):
            if idx >= len(segment_idxs[jdx]):
                seg_idx = segment_idxs[jdx][idx - 1]
            else:
                seg_idx = segment_idxs[jdx][idx]

            m = metrics[int(seg_idx / ANTI_INTERPOL_MULT)]
            color = colors[int(m)]
            colors_sl.append(color)
        colors_all.append(colors_sl)  # this can not be converted to numpy array because last element has one more elem

    sl_actor = actor.streamtube(streamlines, colors=colors_all, linewidth=0.2, opacity=1)
    renderer.add(sl_actor)

    # plot brain mask
    mask = nib.load(brain_mask_path).get_fdata().astype(np.uint8)
    cont_actor = vtk_utils.contour_from_roi_smooth(mask, affine=beginnings_img.affine, color=[.9, .9, .9], opacity=.2,
                                                   smoothing=50)
    renderer.add(cont_actor)

    if show_color_bar:
        lut_cmap = actor.colormap_lookup_table(scale_range=(metrics_min, metrics_max),
                                               hue_range=(0.0, 0.0),
                                               saturation_range=(0.0, 1.0))
        renderer.add(actor.scalar_bar(lut_cmap))

    if orientation == "sagittal":
        renderer.set_camera(position=(-412.95, -34.38, 80.15),
                            focal_point=(102.46, -16.96, -11.71),
                            view_up=(0.1806, 0.0, 0.9835))
    elif orientation == "coronal":
        renderer.set_camera(position=(-48.63, 360.31, 98.37),
                            focal_point=(-20.16, 92.89, 36.02),
                            view_up=(-0.0047, -0.2275, 0.9737))
    elif orientation == "axial":
        pass
    else:
        raise ValueError("Invalid orientation provided")

    # Use this to interatively get new camera angle
    # window.show(renderer, size=img_size, reset_camera=False)
    # print(renderer.get_camera())

    window.record(renderer, out_path=output_path, size=img_size)