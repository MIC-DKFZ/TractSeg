#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import vtk

class VtkUtils:

    @staticmethod
    def contour_smooth(vol, voxsz=(1.0, 1.0, 1.0), affine=None, levels=[50],
                       colors=[np.array([1.0, 0.0, 0.0])], opacities=[0.5], smoothing=10):
        """ Take a volume and draw surface contours for any any number of
        thresholds (levels) where every contour has its own color and opacity

        Parameters
        ----------
        vol : (N, M, K) ndarray
            An array representing the volumetric dataset for which we will draw
            some beautiful contours .
        voxsz : (3,) array_like
            Voxel size.
        affine : None
            Not used.
        levels : array_like
            Sequence of thresholds for the contours taken from image values needs
            to be same datatype as `vol`.
        colors : (N, 3) ndarray
            RGB values in [0,1].
        opacities : array_like
            Opacities of contours.

        Returns
        -------
        vtkAssembly

        Examples
        --------
        >>> import numpy as np
        >>> from dipy.viz import fvtk
        >>> A=np.zeros((10,10,10))
        >>> A[3:-3,3:-3,3:-3]=1
        >>> r=fvtk.ren()
        >>> fvtk.add(r,fvtk.contour(A,levels=[1]))
        >>> #fvtk.show(r)

        """
        major_version = vtk.vtkVersion.GetVTKMajorVersion()

        im = vtk.vtkImageData()
        if major_version <= 5:
            im.SetScalarTypeToUnsignedChar()

        im.SetDimensions(vol.shape[0], vol.shape[1], vol.shape[2])
        # im.SetOrigin(0,0,0)
        # im.SetSpacing(voxsz[2],voxsz[0],voxsz[1])
        if major_version <= 5:
            im.AllocateScalars()
        else:
            im.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 3)

        for i in range(vol.shape[0]):
            for j in range(vol.shape[1]):
                for k in range(vol.shape[2]):
                    im.SetScalarComponentFromFloat(i, j, k, 0, vol[i, j, k])

        ass = vtk.vtkAssembly()
        # ass=[]

        for (i, l) in enumerate(levels):

            # print levels
            skinExtractor = vtk.vtkContourFilter()
            if major_version <= 5:
                skinExtractor.SetInput(im)
            else:
                skinExtractor.SetInputData(im)
            skinExtractor.SetValue(0, l)

            #
            # Smoothing
            # Taken from: https://lorensen.github.io/VTKExamples/site/Python/MeshLabelImageColor/
            #
            smoother = vtk.vtkWindowedSincPolyDataFilter()
            if vtk.VTK_MAJOR_VERSION <= 5:
                smoother.SetInput(skinExtractor.GetOutput())
            else:
                smoother.SetInputConnection(skinExtractor.GetOutputPort())
            smoother.SetNumberOfIterations(smoothing)  # 30  # this has little effect on the error!
            # smoother.BoundarySmoothingOff()
            # smoother.FeatureEdgeSmoothingOff()
            # smoother.SetFeatureAngle(120.0)
            # smoother.SetPassBand(.001)        #this increases the error a lot!
            smoother.NonManifoldSmoothingOn()
            smoother.NormalizeCoordinatesOn()
            smoother.GenerateErrorScalarsOn()
            # smoother.GenerateErrorVectorsOn()
            smoother.Update()

            skinNormals = vtk.vtkPolyDataNormals()
            skinNormals.SetInputConnection(smoother.GetOutputPort())
            skinNormals.SetFeatureAngle(60.0)

            # No Smoothing
            # skinNormals = vtk.vtkPolyDataNormals()
            # skinNormals.SetInputConnection(skinExtractor.GetOutputPort())
            # skinNormals.SetFeatureAngle(60.0)

            skinMapper = vtk.vtkPolyDataMapper()
            skinMapper.SetInputConnection(skinNormals.GetOutputPort())
            skinMapper.ScalarVisibilityOff()

            skin = vtk.vtkActor()

            skin.SetMapper(skinMapper)
            skin.GetProperty().SetOpacity(opacities[i])

            # print colors[i]
            skin.GetProperty().SetColor(colors[i][0], colors[i][1], colors[i][2])
            # skin.Update()
            ass.AddPart(skin)

            del skin
            del skinMapper
            del skinExtractor

        return ass

