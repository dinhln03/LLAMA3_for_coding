from rdkit import Chem

from AnalysisModule.routines.util import load_pkl

# logit_result = yaml_fileread("../logistic.yml")
logit_result = load_pkl("../clf3d/logistic.pkl")

"""
epg-string  --> maxscore
            --> [(f, s)] --> xx, yy, zz, [(x, y, d)] --> refcode, amine
"""
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D


def moltosvg(mol, molSize=(450, 450), kekulize=True):
    mc = Chem.Mol(mol.ToBinary())
    if kekulize:
        try:
            Chem.Kekulize(mc)
        except:
            mc = Chem.Mol(mol.ToBinary())
    if not mc.GetNumConformers():
        rdDepictor.Compute2DCoords(mc)
    drawer = rdMolDraw2D.MolDraw2DSVG(molSize[0], molSize[1])
    # drawer.DrawMolecule(mc, legend="lalala")  # legend fontsize hardcoded, too small
    drawer.DrawMolecule(mc, )
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    # It seems that the svg renderer used doesn't quite hit the spec.
    # Here are some fixes to make it work in the notebook, although I think
    # the underlying issue needs to be resolved at the generation step
    return svg.replace('svg:', '')


def plot_amine(smi):
    m = Chem.MolFromSmiles(smi)
    return moltosvg(m)


def insert_url(svg, n=12, url="https://www.google.com", urlname="ABCDEF"):
    lines = svg.split("\n")
    template = '<a xmlns="http://www.w3.org/2000/svg" xlink:href="{}" xmlns:xlink="http://www.w3.org/1999/xlink" target="__blank"><text x="150" y="400" font-size="4em" fill="black">{}</text></a>'.format(
        url, urlname)
    s = ""
    for il, l in enumerate(lines):
        if il == n:
            s += template + "\n"
        s += l + "\n"
    return s


for epg, epginfo in logit_result.items():
    if epginfo is None:
        print(epg, "info is None")
        continue
    for i, refcode in enumerate(epginfo["refcodes"]):
        a = epginfo["amines"][i]
        svg = plot_amine(a)
        url = "https://www.ccdc.cam.ac.uk/structures/Search?Ccdcid={}".format(refcode)
        # svg = insert_url(svg, urlname=refcode, url=url)
        with open("amines/{}.svg".format(refcode), "w") as f:
            f.write(svg)
