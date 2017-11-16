from dipy.viz import window
from os.path import join
from tractseg.libs.VtkUtils import VtkUtils

class PlotUtils:

    @staticmethod
    def plot_tracts(bundle_segmentations, out_dir):
        '''
        By default this does not work on a remote server connection (ssh -X) because -X does not support OpenGL.
        On the remote Server you can do 'export DISPLAY=":0"' (you should set the value you get if you do 'echo $DISPLAY' if you
        login locally on the remote server). Then all graphics will get rendered locally and not via -X.
        '''

        ren = window.Renderer()

        SMOOTHING = 10
        #CST
        ren.add(VtkUtils.contour_smooth(bundle_segmentations[:,:,:,15], colors=[(0., 0., 1.)], levels=[1], opacities=[1.], smoothing=SMOOTHING))
        ren.add(VtkUtils.contour_smooth(bundle_segmentations[:,:,:,16], colors=[(0., 0., 1.)], levels=[1], opacities=[1.], smoothing=SMOOTHING))
        #CA
        ren.add(VtkUtils.contour_smooth(bundle_segmentations[:,:,:,5], colors=[(1., 0., 0.)], levels=[1], opacities=[1.], smoothing=SMOOTHING))
        #FX
        ren.add(VtkUtils.contour_smooth(bundle_segmentations[:,:,:,23], colors=[(0., 1., 0.)], levels=[1], opacities=[1.], smoothing=SMOOTHING))
        ren.add(VtkUtils.contour_smooth(bundle_segmentations[:,:,:,24], colors=[(0., 1., 0.)], levels=[1], opacities=[1.], smoothing=SMOOTHING))
        #ICP
        ren.add(VtkUtils.contour_smooth(bundle_segmentations[:, :, :, 25], colors=[(1., 1., 0.)], levels=[1], opacities=[1.],
                                        smoothing=SMOOTHING))
        ren.add(VtkUtils.contour_smooth(bundle_segmentations[:, :, :, 26], colors=[(1., 1., 0.)], levels=[1], opacities=[1.],
                                        smoothing=SMOOTHING))

        #First View (Front)
        ren.set_camera(position=(72.47, 343.04, 18.99),
                        focal_point=(71.01, 90.47, 56.05),
                        view_up=(0.03, 0.14, 0.99))
        # window.show(ren, size=(1000, 1000), reset_camera=False)
        window.record(ren, out_path=join(out_dir, "preview_front.png"), size=(600, 600))

        #Second View (Top)
        ren.set_camera(position=(69.76, 144.06, 278.23),
                       focal_point=(71.01, 90.47, 56.05),
                       view_up=(0.01, -0.97, 0.23))
        window.record(ren, out_path=join(out_dir, "preview_top.png"), size=(600, 600))

        # ren.camera_info()  #to print manually selected camera angle


# import nibabel as nib
# data = nib.load("/Volumes/E130-Personal/Wasserthal/data/HCP_example/599469/270g_125mm/tractseg_output_Lasagne/bundle_segmentations.nii.gz").get_data()
# PlotUtils.plot_tracts(data, "/Users/jakob/Downloads")