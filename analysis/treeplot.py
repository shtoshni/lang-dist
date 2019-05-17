import argparse
from ete3 import Tree, TreeStyle, NodeStyle, faces, TextFace


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()

    # Add arguments to parser
    parser.add_argument('--tree_file',
                        help='Tree file in Newick format', type=str)

    args = parser.parse_args()
    return args



def set_face(ftype, fgcolor, fsize=25):
    face = faces.AttrFace("name", fsize=fsize, fgcolor=fgcolor,
                          ftype=ftype)
    face.margin_left = 2
    return face


def get_tree_layout():
    """Returns a layout function for tree for Indo-European langs."""
    nstyle = NodeStyle()
    nstyle["shape"] = "sphere"
    nstyle["size"] = 0

    def set_face(ftype, fgcolor, fsize=25):
        face = faces.AttrFace("name", fsize=fsize, fgcolor=fgcolor,
                              ftype=ftype)
        face.margin_left = 2
        return face

    romance = ['French', 'Italian', 'Portuguese', 'Spanish', 'Catalan',
               'Romanian']
    germanic = ['English', 'Danish', 'Dutch', 'German', 'Norwegian', 'Swedish']
    slavic = ['Croatian', 'Slovenian', 'Czech', 'Slovak', 'Macedonian',
              'Bulgarian', 'Polish', 'Russian', 'Ukrainian']
    hellenic = ['Greek']

    lang_families = ["Romance", "Germanic", "Slavic", "Hellenic"]
    lang_to_family_dict = {"Romance": romance, "Germanic": germanic,
                           "Slavic": slavic, "Hellenic": hellenic}
    ftype_dict = {"Romance": 'Helvetica', "Germanic": 'Avant Garde',
                  "Slavic": 'Courier', "Hellenic": 'Computer Modern'}
    fgcolor_dict = {"Romance": 'slateblue', "Germanic": 'seagreen',
                    "Slavic": 'saddlebrown', "Hellenic": 'darkviolet'}

    face_dict = {}
    for lang_family in lang_families:
        face_dict[lang_family] = set_face(ftype=ftype_dict[lang_family],
                                          fgcolor=fgcolor_dict[lang_family])

    def node_layout_fn(node):
        node.set_style(nstyle)

        # If the node name is a language
        node_is_lang = False
        for lang_family, lang_set in lang_to_family_dict.items():
            if node.name in lang_set:
                # print (node.name)
                faces.add_face_to_node(face_dict[lang_family], node, column=0)
                node_is_lang = True
                break

        # Otherwise node is a language family
        if not node_is_lang:
            for lang_family in lang_families:
                if node.name == lang_family:
                    node.add_face(
                        TextFace(lang_family, ftype=ftype_dict[lang_family],
                                 fgcolor=fgcolor_dict[lang_family], fsize=28),
                        position = "branch-right", column=0)
                    break

    ts = TreeStyle()
    ts.show_leaf_name = False
    ts.show_branch_length = False
    ts.show_branch_support = False
    ts.show_scale = False
    ts.layout_fn = node_layout_fn

    return ts


def main():
    args = parse_args()
    ts = get_tree_layout()
    with open(args.tree_file) as f:
        pred_tree_string = f.read().strip()

    pred_indo_euro_tree = Tree(pred_tree_string)
    pred_indo_euro_tree.show(tree_style=ts)


if __name__=='__main__':
    main()
