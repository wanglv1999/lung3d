import os
import pathlib
import vtk
import slicer

log = open(r"C:\Users\wangl\AppData\Local\Temp\opencode\view_scene.log", "w", encoding="utf-8")
def L(*a):
    log.write(" ".join(map(str, a)) + "\n"); log.flush()

case_dir = pathlib.Path(os.environ.get("SLICER_CASE_DIR") or r"C:\Users\wangl\Desktop\case_001_S40")
try:
    ct = case_dir / "ct.nii.gz"
    if ct.exists():
        slicer.util.loadVolume(str(ct))
        L("CT loaded")

    colors = {"lung_arteries": (0.2, 0.4, 1.0), "lung_veins": (0.55, 0.1, 0.1),
              "lung_airways": (0.2, 1.0, 0.3), "lung_nodules": (1.0, 1.0, 0.1)}
    mesh_dir = case_dir / "mesh"
    if mesh_dir.is_dir():
        for p in sorted(mesh_dir.glob("*.stl")):
            m = slicer.util.loadModel(str(p))
            md = m.GetDisplayNode()
            md.SetVisibility(1)
            md.SetRepresentation(0)
            col = colors.get(p.stem, (1.0, 1.0, 0.1))
            md.SetColor(*col)
            md.SetOpacity(0.95)
            L("model loaded:", p.name, "visible:", md.GetVisibility())

    lm = slicer.app.layoutManager()
    for i in range(lm.threeDViewCount):
        lm.threeDWidget(i).threeDView().resetFocalPoint()
    L("camera fitted to all data")

    rw = lm.threeDWidget(0).threeDView().renderWindow()
    rw.Render()
    w2i = vtk.vtkWindowToImageFilter()
    w2i.SetInput(rw)
    w2i.SetInputBufferTypeToRGB()
    w2i.ReadFrontBufferOff()
    w2i.Update()
    shot = str(case_dir / "slicer_3d_debug.png")
    pw = vtk.vtkPNGWriter()
    pw.SetFileName(shot)
    pw.SetInputConnection(w2i.GetOutputPort())
    pw.Write()
    L("screenshot saved:", shot, "win size:", rw.GetSize())
    L("renderer:", rw.GetRenderers().GetNumberOfItems(), "windows:", lm.threeDViewCount)
except Exception as e:
    import traceback
    L("ERR", traceback.format_exc())
L("done")
log.close()