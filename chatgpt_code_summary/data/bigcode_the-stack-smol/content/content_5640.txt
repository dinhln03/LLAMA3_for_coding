import os
from dominate import document
import dominate.tags as tags
import shlex
import subprocess as sp
from tqdm.auto import tqdm

style = (
"""
#myInput {
  background-image: url('/css/searchicon.png'); /* Add a search icon to input */
  background-position: 10px 12px; /* Position the search icon */
  background-repeat: no-repeat; /* Do not repeat the icon image */
  width: 100%; /* Full-width */
  font-size: 16px; /* Increase font-size */
  padding: 12px 20px 12px 40px; /* Add some padding */
  border: 1px solid #ddd; /* Add a grey border */
  margin-bottom: 12px; /* Add some space below the input */
}

#myUL {
  /* Remove default list styling */
  list-style-type: none;
  padding: 0;
  margin: 0;
}

#myUL li a {
  border: 1px solid #ddd; /* Add a border to all links */
  margin-top: -1px; /* Prevent double borders */
  background-color: #f6f6f6; /* Grey background color */
  padding: 12px; /* Add some padding */
  text-decoration: none; /* Remove default text underline */
  font-size: 18px; /* Increase the font-size */
  color: black; /* Add a black text color */
  display: block; /* Make it into a block element to fill the whole list */
}

#myUL li a:hover:not(.header) {
  background-color: #eee; /* Add a hover effect to all links, except for headers */
}
""")

style2 = (
"""
.row {
  display: flex;
}

.column {
  flex: 33.33%;
  padding: 5px;
}
""")

def runcommand(cmd):
    p = sp.run(shlex.split(cmd), stdout=sp.PIPE, stderr=sp.PIPE)
    return p.stdout, p.stderr

def generate_html(dirname, outdir, title="images"):
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    doc = document(title=title)

    with doc.head:
        tags.style(style)

    with doc:
        with tags.ul(id="myUL"):
            for category in os.listdir(dirname):
                tags.li(tags.a(category, href=category))

    with open(os.path.join(outdir, "index.html"), 'w') as f:
        f.write(doc.render())

    pbar1 = tqdm(os.listdir(dirname), dynamic_ncols=False)
    for category in pbar1:
        pbar1.set_description(category)
        if not os.path.exists(os.path.join(outdir, category)):
            os.makedirs(os.path.join(outdir, category))

        subdoc = document(title=category)
        with subdoc.head:
            tags.style(style)

        with subdoc:
            tags.a("back", href="..")
            with tags.ul(id="myUL"):
                for subcat in os.listdir(os.path.join(dirname, category)):
                    tags.li(tags.a(subcat, href=subcat))

        with open(os.path.join(outdir, category, "index.html"), 'w') as f:
            f.write(subdoc.render())

        pbar2 = tqdm(os.listdir(os.path.join(dirname, category)), dynamic_ncols=False)
        for subcat in pbar2:
            pbar2.set_description(subcat)
            if not os.path.exists(os.path.join(outdir, category, subcat)):
                os.makedirs(os.path.join(outdir, category, subcat))

            ssubdoc = document(title=subcat)
            with ssubdoc.head:
                tags.style(style2)

            imgs = []
            pbar3 = tqdm(os.listdir(os.path.join(dirname, category, subcat)), dynamic_ncols=False)
            for img in pbar3:
                pbar3.set_description(img)
                imgpng = img.replace(".pdf", ".png")
                imgs.append(imgpng)

                runcommand(
                        "convert -density 150 {} -quality 100 {}".format(
                        os.path.join(dirname, category, subcat, img),
                        os.path.join(outdir, category, subcat, imgpng),
                    )
                )

            with ssubdoc:
                tags.a("back", href="..")
                ncols = 3
                for idx in range(0, len(imgs), ncols):
                    with tags.div(_class="row"):
                        final = idx+ncols
                        if final>len(imgs)-1:
                            final = len(imgs)-1
                        for sidx in range(idx, final):
                            with tags.div(_class="column"):
                                tags.img(
                                    src=imgs[sidx],
                                    alt=os.path.splitext(imgs[sidx])[0],
                                    style="height:500px",
                                )

            with open(os.path.join(outdir, category, subcat, "index.html"), 'w') as f:
                f.write(ssubdoc.render())
