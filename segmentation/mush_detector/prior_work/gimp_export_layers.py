import gi, glob, os
gi.require_version('Gimp', '3.0')
from gi.repository import Gimp, Gio
src = glob.glob("/home/seth/ScrollPrizeTutorial/data/Scrolls/crushed scroll label/*.xcf") + ["/home/seth/ScrollPrizeTutorial/data/Scrolls/PHerc0125_c2be2e4_Brigitte.xcf"]
out = os.environ["OUTD"]
for p in sorted(src):
    img = Gimp.file_load(Gimp.RunMode.NONINTERACTIVE, Gio.File.new_for_path(p))
    names = [l.get_name() for l in img.get_layers()]
    print("FILE", os.path.basename(p), img.get_width(), img.get_height(), names, flush=True)
    for layer in img.get_layers():
        dup = img.duplicate()
        for l in dup.get_layers():
            if l.get_name() != layer.get_name():
                dup.remove_layer(l)
        base = os.path.splitext(os.path.basename(p))[0].replace(" ", "_")
        f = Gio.File.new_for_path(os.path.join(out, f"{base}__{layer.get_name().replace(' ','_')}.png"))
        Gimp.file_save(Gimp.RunMode.NONINTERACTIVE, dup, f, None)
        dup.delete()
    img.delete()
